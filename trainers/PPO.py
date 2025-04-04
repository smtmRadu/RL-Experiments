

import os
from datetime import datetime

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

import gymnasium as gym
from networks.policy import Policy
from networks.value import Value
from normalizers.running_normalizer import RunningNormalizer                
from normalizers.rewards_normalizer import RewardsNormalizer
from torch.distributions import Categorical, Independent, Normal
import torch
import torch.nn.functional as F
from flashml.tools.rl import log_episode, display_episodes
from torch.optim.lr_scheduler import LinearLR
import random
import numpy as np
from optimi import StableAdamW

class PPO():
    '''
    automatic PPO trainer
    '''
    def __init__(
            self,
            env_name:str,
            seed:int=0,
            buffer_size:int=10240,
            max_steps:int=1e6,
            num_epoch:int=8,
            batch_size:int=512,

            gamma:float=0.99,        
            clip_epsilon:float=0.2,
            pi_lr:float=3e-4,
            vf_lr:float=1e-3,
            max_norm:float=0.5,
            lam:float=0.97,
            beta:float= 5e-3, # entropy bonus coeff
            lr_annealing:bool=True,

            networks_dims = (2, 64), #layers/hidden
            state_and_reward_clipping = 5.0,       
    ):
        assert buffer_size % batch_size == 0, "Batch size must divide buffer size"


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

        if isinstance(self.env.action_space, gym.spaces.Box):
            self.action_space = "continuous"
        elif isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_space = "discrete"
        elif isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
            self.action_space = "multidiscrete"
        else:
            raise EnvironmentError("What environment is this sir?")   
        self.action_range:tuple = (torch.tensor(self.env.action_space.low, dtype=torch.float32), 
                                torch.tensor(self.env.action_space.high, dtype=torch.float32)) if self.action_space == "continuous" else None
        self.action_scale = (self.action_range[1] - self.action_range[0]) / 2 if self.action_space == "continuous" else 1.0
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0] if self.action_space == "continuous"else self.env.action_space.n
        
        
        self.pi = Policy(self.state_dim, self.action_dim, self.action_space, networks_dims[0], networks_dims[1])
        self.v = Value(self.state_dim, networks_dims[0], networks_dims[1])
        self.pi_optim = StableAdamW(self.pi.parameters(), lr=pi_lr)
        self.v_optim = StableAdamW(self.v.parameters(), lr=vf_lr)
        self.lr_scheduler_pi = LinearLR(self.pi_optim, start_factor=1, end_factor=0, total_iters=max_steps // buffer_size * num_epoch * buffer_size // batch_size)
        self.lr_scheduler_v = LinearLR(self.v_optim, start_factor=1, end_factor=0, total_iters=max_steps // buffer_size * num_epoch * buffer_size // batch_size)
        self.lr_anneal = lr_annealing
        self.state_normalizer = RunningNormalizer(self.state_dim, device='cpu')     
        self.rewards_normalizer = RewardsNormalizer(gamma=gamma, device='cpu')
        self.max_steps = max_steps
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.lam = lam
        self.beta = beta
        self.clip_epsilon = clip_epsilon
        self.max_norm = max_norm
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.state_and_reward_clipping = state_and_reward_clipping
        
        self._load_checkpoint()

    def normalize_and_clip_state(self, state:torch.Tensor, update_online_normalizer:bool) -> torch.Tensor:
        if update_online_normalizer:
            self.state_normalizer.update(state)
        norm_state = self.state_normalizer.normalize(state)
        norm_clip_state = norm_state.clip(-self.state_and_reward_clipping, self.state_and_reward_clipping)
        return norm_clip_state
    
    def normalize_and_clip_reward(self, reward:torch.Tensor, dones:torch.Tensor) -> torch.Tensor:
        norm_reward = self.rewards_normalizer.normalize(reward, dones)
        norm_clip_reward = norm_reward.clip(-self.state_and_reward_clipping, self.state_and_reward_clipping)
        return norm_clip_reward
    
    def scale_actions_for_environment(self, action:torch.Tensor):
        if self.action_space != "continuous":
            return action
        return action * self.action_scale 

    
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
                "v":self.v,
                "pi_optim":self.pi_optim,
                "v_optim":self.v_optim,
                "lr_scheduler_pi": self.lr_scheduler_pi,
                "lr_scheduler_v": self.lr_scheduler_v,
                "state_normalizer":self.state_normalizer,
                "rewards_normalizer":self.rewards_normalizer,
            },
            f= f"./checkpoints/{__name__}/{self.env.spec.id}/checkpoint_{ckp_code}.pth"
        )
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
        self.v = checkpoint["v"]
        self.pi_optim = checkpoint["pi_optim"]
        self.v_optim = checkpoint["v_optim"]
        self.lr_scheduler_pi = checkpoint["lr_scheduler_pi"]
        self.lr_scheduler_v = checkpoint["lr_scheduler_v"]
        self.state_normalizer = checkpoint["state_normalizer"]
        self.rewards_normalizer = checkpoint["rewards_normalizer"]
        print(f"Checkpoint loaded: ({latest_checkpoint})")


    def collect_trajectories(self, steps_COUNT, episodes_COUNT):
        buffer = Buffer(self.buffer_size, self.state_dim, self.action_dim)
        self.v.cpu()
        self.pi.cpu()
        initial_state = torch.tensor(self.env.reset()[0], dtype=torch.float32)
        buffer.states[0] = self.normalize_and_clip_state(initial_state, update_online_normalizer=True)

        episodes_COUNT += 1
        episode_rewards = []
        for timestep_ in range(self.buffer_size):
            with torch.no_grad():
                if self.action_space == "continuous":
                    mu_t, log_std_t = self.pi.forward(buffer.states[timestep_])
                    torch.clip_(log_std_t, -3, 3)
                    _dist = Independent(Normal(mu_t, log_std_t.exp()), 1)
                elif self.action_space == "discrete":
                    logits = self.pi.forward(buffer.states[timestep_])
                    _dist = Categorical(logits=logits)    
                else:
                    raise "Bro I did not implemented for Multivariate Discrete environments"

                action = _dist.sample()
                log_probs = _dist.log_prob(action)
                # Take action and unpack data
                STEP_n = self.env.step(self.scale_actions_for_environment(action).cpu().numpy()) # new state, reward, done, _ 
                reward: torch.Tensor = torch.tensor(STEP_n[1].item(), dtype=torch.float32) if isinstance(STEP_n[1], torch.Tensor) else torch.tensor(STEP_n[1], dtype=torch.float32)  # Reward
                done = torch.ones(1) if STEP_n[2] or STEP_n[3] else torch.zeros(1)
                next_state = torch.tensor(STEP_n[0], dtype=torch.float32)
                
                
                # normalize
                next_state = self.normalize_and_clip_state(next_state, True)
                normalized_reward = self.normalize_and_clip_reward(reward, done)
                # update current tstep data
                buffer.actions[timestep_] = action if self.action_space == "continuous" else F.one_hot(action, self.action_dim)
                buffer.log_probs[timestep_] = log_probs
                buffer.next_states[timestep_] = next_state
                buffer.rewards[timestep_] = normalized_reward
                buffer.dones[timestep_] = done
                episode_rewards.append(reward.item())

                # when episode ends
                if done.item() == 1:
                    next_state = torch.tensor(self.env.reset()[0], dtype=torch.float32)
                    next_state = self.normalize_and_clip_state(next_state, True)
                    
                    log_episode(sum(episode_rewards), episode_length=len(episode_rewards), step=(steps_COUNT, self.max_steps), 
                                other_metrics= {"Policy LR" : self.lr_scheduler_pi.get_last_lr(), "Value LR" : self.lr_scheduler_v.get_last_lr(), __name__: self.envs.spec.id})

                    episodes_COUNT += 1
                    episode_rewards.clear()
                    

                if timestep_ < self.buffer_size - 1:
                    buffer.states[timestep_+1] = next_state

                steps_COUNT += 1

        return buffer, (steps_COUNT, episodes_COUNT)

    def train(self):
        steps_COUNT = 0
        episodes_COUNT = 0
        
        while steps_COUNT < self.max_steps:
            
            # Collect data ----------------------------------------------------------------------------------------------------
            buffer, (steps, episodes) = self.collect_trajectories(steps_COUNT, episodes_COUNT)
            steps_COUNT = steps
            episodes_COUNT = episodes

            # Compute Rewards-to-go ----------------------------------------------------------------------------------------------------
            self.v.to(self.device)
            self.pi.to(self.device)
            with torch.no_grad():
                values_plus_1 = self.v(torch.concat([buffer.states, buffer.next_states[-1:, :]], dim=0).to(self.device)).cpu() # T+1, 1
                buffer.values = values_plus_1[:-1]
                gae = torch.zeros(1) # on parallel it must be tensor (num_envs, 1)
                for timestep_ in reversed(range(buffer.values.size(0) - 1)):
                    delta = buffer.rewards[timestep_] + self.gamma * values_plus_1[timestep_+1] * (1 - buffer.dones[timestep_]) - values_plus_1[timestep_]
                    gae = delta + self.gamma * self.lam * (1 - buffer.dones[timestep_]) * gae
                    buffer.advantages[timestep_] = gae.item()
                    buffer.value_targets[timestep_] = gae.item() + buffer.values[timestep_].item()

            # Update policy -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
            buffer.move_to(self.device)
            for _ in range(self.num_epoch):
                num_batches = self.buffer_size // self.batch_size
                random_batch_permutations = torch.randperm(self.buffer_size, device=self.device)
                for i in range(num_batches):
                    batch_idxs = random_batch_permutations[i * self.batch_size: (i+1) * self.batch_size]
                    s_batch = buffer.states[batch_idxs].detach()

                    V_targ_batch = buffer.value_targets[batch_idxs].detach()
                    v_hat = self.v.forward(s_batch)
                    v_loss = 0.5 * (V_targ_batch - v_hat)**2

                    if self.action_space == "continuous":
                        mu, log_std = self.pi.forward(s_batch)
                        log_std = torch.clip(log_std, -3, 3)
                        _dist = Independent(Normal(mu, log_std.exp()), reinterpreted_batch_ndims=1)
                        log_pi_batch:torch.Tensor = _dist.log_prob(buffer.actions[batch_idxs].detach()).view(-1, 1) # (B, 1)
                    
                    elif self.action_space == "discrete":
                        log_probs = self.pi.forward(s_batch)
                        _dist = Categorical(logits=log_probs)
                        log_pi_batch:torch.Tensor = _dist.log_prob(torch.argmax(buffer.actions[batch_idxs], dim=-1, keepdim=True).squeeze(-1)).view(-1, 1) # (B, 1)      
                    else:
                        raise "what the fuck"
                    
                    log_pi_old_batch = buffer.log_probs[batch_idxs].detach()# (B, 1)
                    entropy_batch = _dist.entropy()
 
                    r = torch.exp(log_pi_batch - log_pi_old_batch)

                    A_batch = buffer.advantages[batch_idxs]
                    full_loss = -torch.min(r * A_batch, torch.clip(r, 1-self.clip_epsilon, 1+self.clip_epsilon) * A_batch) - self.beta * entropy_batch + v_loss
                    full_loss = full_loss.mean()
                    self.v_optim.zero_grad()
                    self.pi_optim.zero_grad()
                    full_loss.backward()
                    self.v_optim.step()
                    self.pi_optim.step()

                    if self.lr_anneal:
                        self.lr_scheduler_pi.step()
                        self.lr_scheduler_v.step()

        print()
        self._make_checkpoint()
        display_episodes()

    def test(self):
        test_env = gym.make(self.env.spec.id, render_mode="human")

        self.pi.cpu()
        self.v.cpu()
        
        episode_idx_count = 0
        # Run test episodes
        while True:
            obs, _ = test_env.reset()
            state = torch.tensor(obs, dtype=torch.float32)
            state = self.normalize_and_clip_state(state, update_online_normalizer=False)
            episode_reward = 0.0
            episode_len = 0
            done = False
            episode_idx_count += 1
            while not done:
                test_env.render()

                with torch.no_grad():
                    if self.action_space == "continuous":
                        mu, log_std = self.pi.forward(state)
                        log_std = torch.clamp(log_std, -3, 3)
                        _dist = Independent(Normal(mu, log_std.exp()), 1)
                    elif self.action_space == "discrete":
                        logits = self.pi.forward(state)
                        _dist = Categorical(logits=logits)
                    else:
                        raise NotImplementedError("Test not implemented for this action space.")

                    action = _dist.sample()

                scaled_action = self.scale_actions_for_environment(action).cpu().numpy()
                step_result = test_env.step(scaled_action)

                if len(step_result) == 5:
                    next_obs, reward, done_flag, truncated, _ = step_result
                    done = done_flag or truncated

                else:
                    next_obs, reward, done, _= step_result

                episode_reward += float(reward)
                next_state = torch.tensor(next_obs, dtype=torch.float32)
                next_state = self.normalize_and_clip_state(next_state, update_online_normalizer=False)
                state = next_state
                episode_len += 1

            print(f"[Test] Episode {episode_idx_count} [Cumulated Reward: {episode_reward}] [Episode Length: {episode_len}]")

        
class Buffer:
    def __init__(self, dim:int, state_dim:int, action_dim:int) -> None:
        self.states =        torch.zeros(dim, state_dim, requires_grad=False)
        self.next_states =   torch.zeros(dim, state_dim)
        self.actions =       torch.zeros(dim, action_dim)
        self.log_probs =     torch.zeros(dim, 1)
        self.rewards =       torch.zeros(dim, 1)
        self.dones =         torch.zeros(dim, 1)
        self.values =        torch.zeros(dim, 1)
        self.value_targets = torch.zeros(dim, 1)
        self.advantages =    torch.zeros(dim, 1)

    def move_to(self, device):
        self.states = self.states.to(device)
        self.next_states = self.next_states.to(device)
        self.actions = self.actions.to(device)
        self.log_probs = self.log_probs.to(device)
        self.rewards = self.rewards.to(device)
        self.dones = self.dones.to(device)
        self.values = self.values.to(device)
        self.value_targets = self.value_targets.to(device)
        self.advantages = self.advantages.to(device)
