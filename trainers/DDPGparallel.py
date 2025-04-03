

import os
from datetime import datetime
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

import gymnasium as gym
from networks.policy import Policy
from networks.q import Q
import torch
import torch.nn.functional as F
from flashml.tools.rl import log_episode, display_episodes
from optimi import StableAdamW
from flashml.tools import OrnsteinUhlenbeckProcess

class Buffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.s = torch.zeros(max_size, state_dim, requires_grad=False)
        self.a = torch.zeros(max_size, action_dim, requires_grad=False)
        self.r = torch.zeros(max_size, 1, requires_grad=False)
        self.s_ = torch.zeros(max_size, state_dim, requires_grad=False)
        self.d = torch.zeros(max_size, 1, requires_grad=False)
        self.count = 0

    def add(self, s, a, r, s_, d):
        batch_size = s.shape[0]
        self.s[self.count:self.count+batch_size] = s
        self.a[self.count:self.count+batch_size] = a
        self.r[self.count:self.count+batch_size] = r
        self.s_[self.count:self.count+batch_size] = s_
        self.d[self.count:self.count+batch_size] = d
        self.count += batch_size

    def sample_batch(self, batch_size):
        idx = torch.randint(0, self.count, (batch_size,))
        return self.s[idx], self.a[idx], self.r[idx], self.s_[idx], self.d[idx]

class DDPG():
    def __init__(
            self,
            env_name:str,
            num_envs:int = 8,
            seed:int=0,
            max_steps:int=1e6,
            batch_size:int=100,

            gamma:float=0.99,        
            lr:float=1e-3,   
            act_noise=0.1,
            tau:float=0.005,
            init_steps=1000, # steps without updates
            update_every = 50,
            networks_dimensions = (2, 64),
            state_clipping = 3.0,       
    ):
        assert batch_size < init_steps * 5, "You must collect steps at first before updating policy"
        torch.backends.cudnn.benchmark = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(seed=seed)
        torch.cuda.manual_seed_all(seed=seed)

        self.envs = gym.make_vec(env_name, num_envs=num_envs)

        assert isinstance(self.envs.action_space, gym.spaces.Box), "DDPG works only in continuous action spaces"

        self.action_range:tuple = (torch.tensor(self.envs.action_space.low, dtype=torch.float32, device=self.device), 
                                torch.tensor(self.envs.action_space.high, dtype=torch.float32, device=self.device))
        self.action_scale = (self.action_range[1] - self.action_range[0]) / 2
        self.state_dim = self.envs.observation_space.shape[-1]
        self.action_dim = self.envs.action_space.shape[-1]
        
        
        self.ou_noise = OrnsteinUhlenbeckProcess(self.action_dim * num_envs, theta=0.15, sigma=act_noise)
        self.pi = Policy(self.state_dim, self.action_dim, "deterministic", networks_dimensions[0], networks_dimensions[1]).to(self.device)
        self.q = Q(self.state_dim, self.action_dim, networks_dimensions[0], networks_dimensions[1]).to(self.device)
        self.target_pi = Policy(self.state_dim, self.action_dim, "deterministic", networks_dimensions[0], networks_dimensions[1]).to(self.device)
        self.target_q = Q(self.state_dim, self.action_dim, networks_dimensions[0], networks_dimensions[1]).to(self.device)
        #set targ params to equal main params
        self.target_pi.load_state_dict(self.pi.state_dict())
        self.target_q.load_state_dict(self.q.state_dict())

        if not isinstance(self.pi.net[-1], torch.nn.Tanh):
            self.pi.net.append(torch.nn.Tanh())
            self.target_pi.net.append(torch.nn.Tanh())


        #set targ params to equal main params
        self.target_pi.load_state_dict(self.pi.state_dict())
        self.target_q.load_state_dict(self.q.state_dict())

        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=lr, weight_decay=0)
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=lr, weight_decay=0)
        self.max_steps = max_steps
        self.gamma = gamma
        self.epsilon=act_noise
        self.tau = tau
        self.init_steps = init_steps
        self.batch_size = batch_size
        self.state_and_reward_clipping = state_clipping
        self.update_every = update_every
        self._load_checkpoint()

    def train(self):
        steps_COUNT = 0
        steps_COUNT_forGD = 0
        steps_GD = 0
        episodes_COUNT = self.envs.num_envs

        episode_rewards = [[] for _ in range(self.envs.num_envs)]
        buffer = Buffer(self.max_steps, self.state_dim, self.action_dim)
        state_batch = torch.tensor(self.envs.reset()[0], dtype=torch.float32).clip_(-self.state_and_reward_clipping, self.state_and_reward_clipping)    
        
        while steps_COUNT < self.max_steps:
            with torch.no_grad(): 
                mu_t = self.pi.forward(state_batch.to(self.device))
                # eps = torch.randn_like(mu_t, device=self.device) * self.epsilon
                eps = torch.tensor(self.ou_noise.sample(), dtype=torch.float32).to(self.device).reshape(self.envs.num_envs, -1)
                actions_batch = (mu_t + eps).detach().clip_(-1, 1)

                    # Scale actions to environment range
                STEPS_batch = self.envs.step((actions_batch * self.action_scale).detach().cpu().numpy())  # new state, reward, done, _

                next_state_batch = torch.tensor(STEPS_batch[0], dtype=torch.float32, requires_grad=False).clip_(-self.state_and_reward_clipping, self.state_and_reward_clipping)
                rewards_batch = torch.tensor(STEPS_batch[1], dtype=torch.float32, requires_grad=False)
                dones_batch = torch.tensor(STEPS_batch[2], dtype=torch.float32, requires_grad=False) 
                
                    
                # store steps 1 by 1
                buffer.add(state_batch.view(-1, self.state_dim), actions_batch.view(-1, self.action_dim), rewards_batch.view(-1, 1), next_state_batch.view(-1, self.state_dim), dones_batch.view(-1, 1))
            
            for i in range(self.envs.num_envs):
                episode_rewards[i].append(rewards_batch[i].item())
            if dones_batch.any():
                done_indices = torch.where(dones_batch)[0]
                for idx in done_indices:
                    log_episode(
                        sum(episode_rewards[idx]), 
                        episode_length=len(episode_rewards[idx]), 
                        step=(steps_COUNT, self.max_steps),
                        other_metrics={"gd_steps": steps_GD}
                    )
                    episodes_COUNT += 1
                    episode_rewards[idx].clear()

            state_batch = next_state_batch    
            steps_COUNT += self.envs.num_envs

            # Train here----------------------------------------------------------------------------------------------------
            # sample random batch
            if steps_COUNT < self.init_steps:
                continue
            if steps_COUNT - steps_COUNT_forGD < self.update_every:
                continue
            steps_COUNT_forGD += self.update_every # at least an approx here
            steps_GD += 1



            s, a, r, s_prime, d = buffer.sample_batch(self.batch_size)
            s = s.to(self.device)
            a = a.to(self.device)
            r = r.to(self.device)
            s_prime = s_prime.to(self.device)
            d = d.to(self.device)
        
            # Compute targets for q functions ----------------------------------------------------------------------------------------------------
            with torch.no_grad():
                mu_s_prime = self.target_pi(s_prime)
                q_qtarg = self.target_q.forward(s_prime, mu_s_prime)
                y = r + self.gamma * (1 - d) * q_qtarg

            # Update Q functions----------------------------------------------------------------------------------------------------
            self.q.train()
            q_loss = (self.q(s, a) - y)**2
            self.q_optim.zero_grad()
            q_loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1)
            self.q_optim.step()

            # Update PI function-----------------------------------------------------------------------------------------------------
            self.q.eval()
            mu_s = self.pi(s)
            pi_loss = self.q(s, mu_s)
            self.pi_optim.zero_grad()
            (-pi_loss.mean()).backward()
            torch.nn.utils.clip_grad_norm_(self.pi.parameters(), 1)
            self.pi_optim.step()

            # smooth update of target networks
            with torch.no_grad():
                for targ_pi_param, pi_param in zip(self.target_pi.parameters(), self.pi.parameters()):
                    targ_pi_param.data.copy_(self.tau * pi_param.data + (1 - self.tau) * targ_pi_param.data)

                for targ_q_param, q_param in zip(self.target_q.parameters(), self.q.parameters()):
                    targ_q_param.data.copy_(self.tau * q_param.data + (1 - self.tau) * targ_q_param.data)
                   

                

        print()
        self._make_checkpoint()
        display_episodes()

    def _make_checkpoint(self):
        if not os.path.exists("./checkpoints"):
            os.mkdir("./checkpoints")

        if not os.path.exists(f"./checkpoints/{__name__}"):
            os.mkdir(f"./checkpoints/{__name__}")

        if not os.path.exists(f"./checkpoints/{__name__}/{self.envs.spec.id}"):
            os.mkdir(f"./checkpoints/{__name__}/{self.envs.spec.id}")

        ckp_code = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        torch.save(
            {
                "pi":self.pi,
                "q": self.q,
                "pi_optim":self.pi_optim,
                "q_optim":self.q_optim,
            },
            f= f"./checkpoints/{__name__}/{self.envs.spec.id}/checkpoint_{ckp_code}.pth")
        print(f"Checkpoint {ckp_code} saved (./checkpoints/{__name__}/{self.envs.spec.id})")

    def _load_checkpoint(self):
        if not os.path.exists("./checkpoints"):
            os.mkdir("./checkpoints")
        
        if not os.path.exists(f"./checkpoints/{__name__}"):
            os.mkdir(f"./checkpoints/{__name__}")

        if not os.path.exists(f"./checkpoints/{__name__}/{self.envs.spec.id}"):
            os.mkdir(f"./checkpoints/{__name__}/{self.envs.spec.id}")

        path = f"./checkpoints/{__name__}/{self.envs.spec.id}"
        checkpoint_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        
        if not checkpoint_files:
            return

        latest_checkpoint = sorted(checkpoint_files)[-1]
        checkpoint_path = os.path.join(path, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.pi = checkpoint["pi"]
        self.q = checkpoint["q"]
        self.pi_optim = checkpoint["pi_optim"]
        self.q_optim = checkpoint["q_optim"]
        print(f"Checkpoint loaded: ({latest_checkpoint})")

        for param in self.target_pi.parameters():
                param.requires_grad = False
        for param in self.target_q.parameters():
            param.requires_grad = False
