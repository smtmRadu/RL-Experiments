

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

    def add(self, s, a, r, s_, d):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.d[self.count] = d
        self.count += 1

    def sample_batch(self, batch_size):
        idx = torch.randint(0, self.count, (batch_size,))
        return self.s[idx], self.a[idx], self.r[idx], self.s_[idx], self.d[idx]
class DDPG():
    def __init__(
            self,
            env_name:str,
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

        self.env = gym.make(env_name)

        assert isinstance(self.env.action_space, gym.spaces.Box), "DDPG works only in continuous action spaces"

        self.action_range:tuple = (torch.tensor(self.env.action_space.low, dtype=torch.float32, device=self.device), 
                                torch.tensor(self.env.action_space.high, dtype=torch.float32, device=self.device))
        self.action_scale = (self.action_range[1] - self.action_range[0]) / 2

        


        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        

        self.pi = Policy(self.state_dim, self.action_dim, "deterministic", networks_dimensions[0], networks_dimensions[1]).to(self.device)
        self.q = Q(self.state_dim, self.action_dim, networks_dimensions[0], networks_dimensions[1]).to(self.device)
        self.target_pi = Policy(self.state_dim, self.action_dim, "deterministic", networks_dimensions[0], networks_dimensions[1]).to(self.device)
        self.target_q = Q(self.state_dim, self.action_dim, networks_dimensions[0], networks_dimensions[1]).to(self.device)
        if not isinstance(self.pi.net[-1], torch.nn.Tanh):
            self.pi.net.append(torch.nn.Tanh())
            self.target_pi.net.append(torch.nn.Tanh())

        #set targ params to equal main params
        self.target_pi.load_state_dict(self.pi.state_dict())
        self.target_q.load_state_dict(self.q.state_dict())

        for param in self.target_pi.parameters():
            param.requires_grad = False
        for param in self.target_q.parameters():
            param.requires_grad = False

        self.pi_optim = StableAdamW(self.pi.parameters(), lr=lr, weight_decay=0)
        self.q_optim = StableAdamW(self.q.parameters(), lr=lr, weight_decay=0)
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
        episodes_COUNT = 1

        episode_rewards = []
        buffer = Buffer(self.max_steps, self.state_dim, self.action_dim)
        state = torch.tensor(self.env.reset()[0], dtype=torch.float32, requires_grad=False).clip_(-self.state_and_reward_clipping, self.state_and_reward_clipping)
        
        while steps_COUNT < self.max_steps:
            self.pi.eval()
            with torch.no_grad(): 
                mu_t = self.pi.forward(state.to(self.device))
                # eps = torch.tensor(self.ou_noise.sample(), dtype=torch.float32).to(self.device)
                eps = self.epsilon * torch.randn(self.action_dim, device=self.device)
                action = (mu_t + eps).detach().clip_(-1, 1)

                    # Scale actions to environment range
                step = self.env.step((action * self.action_scale).detach().cpu().numpy())  # new state, reward, done, _

                next_state = torch.tensor(step[0], dtype=torch.float32, requires_grad=False).clip_(-self.state_and_reward_clipping, self.state_and_reward_clipping)
                reward = torch.tensor(step[1], dtype=torch.float32, requires_grad=False)
                done = torch.tensor(step[2], dtype=torch.float32, requires_grad=False) 
                
                buffer.add(state, action, reward, next_state, done)

            episode_rewards.append(reward.item())
            if done.item()==1:
                next_state = torch.tensor(self.env.reset()[0], dtype=torch.float32, requires_grad=False).clip_(-self.state_and_reward_clipping, self.state_and_reward_clipping)
 
                log_episode(
                    sum(episode_rewards), 
                    episode_length=len(episode_rewards), 
                    step=(steps_COUNT, self.max_steps),
                    other_metrics={"gd_steps": steps_GD, __name__: self.envs.spec.id}
                )
                episodes_COUNT += 1
                episode_rewards.clear()

            state = next_state    
            steps_COUNT += 1

            # Train here----------------------------------------------------------------------------------------------------
            # sample random batch
            if steps_COUNT < self.init_steps:
                continue
            if steps_COUNT - steps_COUNT_forGD < self.update_every:
                continue
            steps_COUNT_forGD += self.update_every
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
            self.q_optim.step()

            # Update PI function-----------------------------------------------------------------------------------------------------
            self.q.eval()
            pi_loss = -self.q(s, self.pi(s))
            self.pi_optim.zero_grad()
            pi_loss.mean().backward()
            self.pi_optim.step()

            # smooth update of target networks
            with torch.no_grad():
                for pitarg_param, pi_param in zip(self.target_pi.parameters(), self.pi.parameters()):
                    pitarg_param.data.copy_(self.tau * pi_param.data + (1 - self.tau) * pitarg_param.data)

                for qtarg_param, q_param in zip(self.target_q.parameters(), self.q.parameters()):
                    qtarg_param.data.copy_(self.tau * q_param.data + (1 - self.tau) * qtarg_param.data)
                   
                
        print()
        self._make_checkpoint()
        display_episodes()

    def _make_checkpoint(self):
        if not os.path.exists("./checkpoints"):
            os.mkdir("./checkpoints")

        if not os.path.exists(f"./checkpoints/{__name__}"):
            os.mkdir(f"./checkpoints/{__name__}")

        if not os.path.exists(f"./checkpoints/{__name__}/{self.env.spec.id}"):
            os.mkdir(f"./checkpoints/{__name__}/{self.env.spec.id}")

        ckp_code = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        torch.save(
            {
                "pi":self.pi,
                "q": self.q,
                "pi_optim":self.pi_optim,
                "q_optim":self.q_optim,
            },
            f= f"./checkpoints/{__name__}/{self.env.spec.id}/checkpoint_{ckp_code}.pth")
        print(f"Checkpoint {ckp_code} saved (./checkpoints/{__name__}/{self.env.spec.id})")

    def _load_checkpoint(self):
        if not os.path.exists("./checkpoints"):
            os.mkdir("./checkpoints")
        
        if not os.path.exists(f"./checkpoints/{__name__}"):
            os.mkdir(f"./checkpoints/{__name__}")

        if not os.path.exists(f"./checkpoints/{__name__}/{self.env.spec.id}"):
            os.mkdir(f"./checkpoints/{__name__}/{self.env.spec.id}")

        path = f"./checkpoints/{__name__}/{self.env.spec.id}"
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
