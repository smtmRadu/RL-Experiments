

import os
from datetime import datetime
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

import gymnasium as gym
from networks.policy import Policy
from networks.q import Q
from normalizers.running_normalizer import RunningNormalizer
from normalizers.rewards_normalizer import RewardsNormalizer
from torch.distributions import Categorical, Independent, Normal
import torch
import torch.nn.functional as F
from flashml.tools.rl import log_episode, display_episodes
from typing import List, Tuple
from optimi import StableAdamW
import math
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

class SAC():
    def __init__(
            self,
            env_name:str,
            num_envs:int = 8,
            seed:int=0,
            max_steps:int=1e6,
            batch_size:int=100,

            gamma:float=0.99,        
            lr:float=1e-3,   
            alpha=0.2,
            tau:float=0.005,
            init_steps=1000, # steps without updates
            update_every = 50,
            networks_dimensions = (2, 64),
            state_clipping = 5.0,       
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

        self.action_range:tuple = (torch.tensor(self.envs.action_space.low, dtype=torch.float32), 
                                torch.tensor(self.envs.action_space.high, dtype=torch.float32))
                    
        self.action_scale = (self.action_range[1] - self.action_range[0]) / 2
        self.state_dim = self.envs.observation_space.shape[-1]
        self.action_dim = self.envs.action_space.shape[-1]
        
        

        self.pi = Policy(self.state_dim, self.action_dim, "continuous", networks_dimensions[0], networks_dimensions[1]).to(self.device)
        self.q1 = Q(self.state_dim, self.action_dim, networks_dimensions[0], networks_dimensions[1]).to(self.device)
        self.q2 = Q(self.state_dim, self.action_dim, networks_dimensions[0], networks_dimensions[1]).to(self.device)
        self.target_q1 = Q(self.state_dim, self.action_dim, networks_dimensions[0], networks_dimensions[1]).to(self.device)
        self.target_q2 = Q(self.state_dim, self.action_dim, networks_dimensions[0], networks_dimensions[1]).to(self.device)
        #set targ params to equal main params
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        if type(self.pi.net[-1]) is not torch.nn.Tanh:
            self.pi.net.append(torch.nn.Tanh())

        for param in self.target_q1.parameters():
            param.requires_grad = False
        for param in self.target_q2.parameters():
            param.requires_grad = False

        self.pi_optim = StableAdamW(self.pi.parameters(), lr=lr, weight_decay=0)
        self.q1q2_optim = StableAdamW([*self.q1.parameters(), *self.q2.parameters()], lr=lr)
        self.max_steps = max_steps
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.init_steps = init_steps
        self.batch_size = batch_size
        self.state_and_reward_clipping = state_clipping
        self.update_every = update_every
        self._load_checkpoint()

    def scale_actions_for_environment(self, action:torch.Tensor):
        return action * self.action_scale 

    def train(self):
        steps_COUNT = 0
        steps_COUNT_forGD = 0
        steps_GD = 0
        episodes_COUNT = self.envs.num_envs

        episode_rewards = [[] for _ in range(self.envs.num_envs)]
        buffer = Buffer(self.max_steps, self.state_dim, self.action_dim)
        state_batch = torch.tensor(self.envs.reset()[0], dtype=torch.float32, requires_grad=False)
        
        while steps_COUNT < self.max_steps:
            state_batch_cuda = state_batch.to(self.device)

            if steps_COUNT < self.init_steps:
                # do Random actions and collect 
                actions_batch = torch.rand(size=(self.envs.num_envs,self.action_dim)).detach().cpu()
            else:
                with torch.no_grad():
                    mu_t, log_std_t = self.pi.forward(state_batch_cuda)
                    torch.clip_(log_std_t, -3, 3)
                    _dist = Normal(mu_t, log_std_t.exp())
                    actions_batch = _dist.rsample().detach().cpu()

            STEPS_batch = self.envs.step(self.scale_actions_for_environment(actions_batch).numpy()) # new state, reward, done, _ 
            next_state_batch = torch.tensor(STEPS_batch[0], dtype=torch.float32, requires_grad=False)
            rewards_batch = torch.tensor(STEPS_batch[1], dtype=torch.float32, requires_grad=False)  
            dones_batch = torch.tensor(STEPS_batch[2], dtype=torch.float32, requires_grad=False)
            
                
           # store steps 1 by 1
            with torch.no_grad():
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
                mu, log_std = self.pi(s_prime)
                torch.clip_(log_std, -3, 3)
                _dist = Normal(mu, log_std.exp())
                u_prime =_dist.rsample()
                a_tilde_prime = F.tanh(u_prime)
                log_prob = _dist.log_prob(u_prime).sum(-1) - torch.sum(2 * (math.log(2) - u_prime - F.softplus(-2*u_prime)), dim=-1)
                y = r + self.gamma * (1 - d) * \
                    (torch.minimum(self.target_q1(s_prime, a_tilde_prime), self.target_q2(s_prime,a_tilde_prime)) - self.alpha * log_prob)

            # Update Q functions----------------------------------------------------------------------------------------------------
            self.q1.train()
            self.q2.train()
            q_loss = ((self.q1(s, a) - y)**2).mean() + ((self.q2(s, a) - y)**2).mean()
            self.q1q2_optim.zero_grad()
            q_loss.backward()
            self.q1q2_optim.step()

            # Update PI function-----------------------------------------------------------------------------------------------------
            self.q1.eval()
            self.q2.eval()
            mu, log_std = self.pi(s)
            log_std = torch.clip(log_std, -3, 3)
            std = log_std.exp()
            _dist = Normal(mu, std)
            u = _dist.rsample()
            a = torch.tanh(u)
            log_pi_as =  _dist.log_prob(u).sum(-1) - torch.sum(2*(math.log(2) - u - F.softplus(-2*u)), dim=-1)
            pi_loss = torch.minimum(self.q1(s, a), self.q2(s, a)) - self.alpha * log_pi_as

            self.pi_optim.zero_grad()
            (-pi_loss).mean().backward()
            self.pi_optim.step()


            # smooth update of target networks
            with torch.no_grad():
                for targ_q1, q1 in zip(self.target_q1.parameters(), self.q1.parameters()):
                    targ_q1.data.copy_(self.tau * q1.data + (1 - self.tau) * targ_q1.data)
                for targ_q2, q2 in zip(self.target_q2.parameters(), self.q2.parameters()):
                    targ_q2.data.copy_(self.tau * q2.data + (1 - self.tau) * targ_q2.data)

                

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


