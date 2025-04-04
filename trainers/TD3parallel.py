

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
import random
import numpy as np
class Buffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.s = torch.zeros(max_size, state_dim, requires_grad=False)
        self.a = torch.zeros(max_size, action_dim, requires_grad=False)
        self.r = torch.zeros(max_size, 1, requires_grad=False)
        self.s_ = torch.zeros(max_size, state_dim, requires_grad=False)
        self.d = torch.zeros(max_size, 1, requires_grad=False)
        self.count = 0
        self.max_size = max_size

    def add(self, s, a, r, s_, d):
        batch_size = s.shape[0]
        if batch_size + self.count <= self.max_size:
            pass
        self.s[self.count:self.count+batch_size] = s
        self.a[self.count:self.count+batch_size] = a
        self.r[self.count:self.count+batch_size] = r
        self.s_[self.count:self.count+batch_size] = s_
        self.d[self.count:self.count+batch_size] = d
        self.count += batch_size

    def sample_batch(self, batch_size):
        idx = torch.randint(0, self.count, (batch_size,))
        return self.s[idx], self.a[idx], self.r[idx], self.s_[idx], self.d[idx]

class TD3():
    def __init__(
            self,
            env_name:str,
            
            num_envs:int = 8,
            seed:int=0,
            max_steps:int=1e6,
            batch_size:int=100,

            gamma:float=0.99,        
            lr:float=1e-3,   
            act_noise=0.1, # used on inference
            target_noise = 0.2, # used on target actions computation
            noise_clip =0.5,
            policy_delay = 2,
            tau:float=0.005,
            init_steps=1000, # steps without updates
            update_every = 50,
            networks_dimensions = (2, 64),
            state_clipping = 3.0,       
    ):
        assert batch_size < init_steps * 5, "You must collect steps at first before updating policy"
        self.device = "cpu"
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        random.seed(seed)   
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False  
        torch.backends.cudnn.deterministic = True

        self.envs = gym.make_vec(env_name, num_envs=num_envs)

        assert isinstance(self.envs.action_space, gym.spaces.Box), "SAC works only in continuous action spaces"

        self.action_range:tuple = (torch.tensor(self.envs.action_space.low, dtype=torch.float32, device=self.device), 
                                torch.tensor(self.envs.action_space.high, dtype=torch.float32, device=self.device))
                    
        self.action_scale = (self.action_range[1] - self.action_range[0]) / 2
        self.state_dim = self.envs.observation_space.shape[-1]
        self.action_dim = self.envs.action_space.shape[-1]
        
        

        self.pi = Policy(self.state_dim, self.action_dim, "deterministic", networks_dimensions[0], networks_dimensions[1]).to(self.device)
        self.q1 = Q(self.state_dim, self.action_dim, networks_dimensions[0], networks_dimensions[1]).to(self.device)
        self.q2 = Q(self.state_dim, self.action_dim, networks_dimensions[0], networks_dimensions[1]).to(self.device)
        self.target_pi = Policy(self.state_dim, self.action_dim, "deterministic", networks_dimensions[0], networks_dimensions[1]).to(self.device)
        self.target_q1 = Q(self.state_dim, self.action_dim, networks_dimensions[0], networks_dimensions[1]).to(self.device)
        self.target_q2 = Q(self.state_dim, self.action_dim, networks_dimensions[0], networks_dimensions[1]).to(self.device)
        
        if not isinstance(self.pi.net[-1], torch.nn.Tanh):
            self.pi.net.append(torch.nn.Tanh())
            self.target_pi.net.append(torch.nn.Tanh())

        #set targ params to equal main params
        self.target_pi.load_state_dict(self.pi.state_dict())
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())


        for param in self.target_pi.parameters():
            param.requires_grad = False
        for param in self.target_q1.parameters():
            param.requires_grad = False
        for param in self.target_q2.parameters():
            param.requires_grad = False

        self.pi_optim = StableAdamW(self.pi.parameters(), lr=lr, weight_decay=0)
        self.q1q2_optim = StableAdamW([*self.q1.parameters(), *self.q2.parameters()], lr=lr, weight_decay=0)
        self.max_steps = max_steps
        self.gamma = gamma
        self.noise_clip = noise_clip
        self.sigma_targ_act_noise = target_noise
        self.policy_delay = policy_delay
        self.eps_act_noise=act_noise
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
                eps = torch.randn_like(mu_t, device=self.device) * self.eps_act_noise
                # eps = torch.tensor(self.ou_noise.sample(), dtype=torch.float32).to(self.device).reshape(self.envs.num_envs, -1)
                actions_batch = (mu_t + eps).detach().clip_(-1, 1)

                    # Scale actions to environment range
                STEPS_batch = self.envs.step((actions_batch * self.action_scale).cpu().numpy())  # new state, reward, done, _

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
                        other_metrics={"gd_steps": steps_GD, __name__: self.envs.spec.id}
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

            # Compute target actions &  targets for q functions ----------------------------------------------------------------------------------------------------
            with torch.no_grad():
                eps = torch.randn(self.batch_size, self.action_dim, device=self.device) * self.sigma_targ_act_noise
                a_prime_s_prime = torch.clip(self.target_pi(s_prime) + torch.clip(eps, -self.noise_clip, self.noise_clip), -1, 1)
                y = r + self.gamma * (1 - d) * \
                    (torch.min(self.target_q1(s_prime, a_prime_s_prime), self.target_q2(s_prime,a_prime_s_prime)))

            # Update Q functions----------------------------------------------------------------------------------------------------
            self.q1.train()
            self.q2.train()
            q_loss = ((self.q1(s, a) - y)**2).mean() + ((self.q2(s, a) - y)**2).mean()
            self.q1q2_optim.zero_grad()
            q_loss.backward()
            self.q1q2_optim.step()

            # Update PI function-----------------------------------------------------------------------------------------------------
            if steps_GD % self.policy_delay == 0:
                self.q1.eval()
                self.q2.eval()
                pi_loss = -self.q1(s, self.pi(s))
                self.pi_optim.zero_grad()
                pi_loss.mean().backward()
                self.pi_optim.step()

                # smooth update of target networks
                with torch.no_grad():
                    for pitarg_param, pi_param in zip(self.target_pi.parameters(), self.pi.parameters()):
                        pitarg_param.data.copy_(self.tau * pi_param.data + (1 - self.tau) * pitarg_param.data)

                    for qtarg_param, q_param in zip(self.target_q1.parameters(), self.q1.parameters()):
                        qtarg_param.data.copy_(self.tau * q_param.data + (1 - self.tau) * qtarg_param.data)

                    for qtarg_param, q_param in zip(self.target_q2.parameters(), self.q2.parameters()):
                        qtarg_param.data.copy_(self.tau * q_param.data + (1 - self.tau) * qtarg_param.data)
                   
                
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
                "q1": self.q1,
                "q2": self.q2,
                "pi_optim":self.pi_optim,
                "q1q2_optim":self.q1q2_optim,
            },
            f= f"./checkpoints/{__name__}/{self.envs.spec.id}/checkpoint_{ckp_code}.pth"
        )
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
        self.q1 = checkpoint["q1"]
        self.q2 = checkpoint["q2"]
        self.pi_optim = checkpoint["pi_optim"]
        self.q1q2_optim = checkpoint["q1q2_optim"]
        print(f"Checkpoint loaded: ({latest_checkpoint})")

        for param in self.target_pi.parameters():
                param.requires_grad = False
        for param in self.target_q1.parameters():
            param.requires_grad = False
        for param in self.target_q2.parameters():
            param.requires_grad = False
