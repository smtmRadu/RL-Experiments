

import os

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

import gymnasium as gym
from policy import Policy
from value import Value
from torch.optim import Adam, AdamW
from torch.distributions import MultivariateNormal, Categorical, Independent, Normal
import torch
import torch.nn.functional as F
import math
from flashml.info import log_metrics, display_metrics
from flashml.info.rl import log_episode, display_episodes
from tqdm import tqdm
from running_normalizer import RunningNormalizer
from torchinfo import summary
import inspect
from optimi import StableAdamW
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv


# This is not done yet. I remained at gae computation
class PPO():
    '''
    automatic PPO trainer. Runs only on device, no transfers.
    '''
    def __init__(
            self,
            env_name:str,
            num_envs:int = 8,
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
            target_kl=1e-2,

            max_episode_len=1000,
            horizon:int=1024,
            
            networks_dimensions = (2, 64),
            state_clipping = 5.0,       
    ):
        assert buffer_size % batch_size == 0, f"Batch size must divide buffer size (recv: {buffer_size} and {num_envs})"
        assert buffer_size % num_envs == 0, f"Num envs must divide buffer size (recv: {buffer_size} and {num_envs})"

        torch.backends.cudnn.benchmark = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(seed=seed)
        torch.cuda.manual_seed_all(seed=seed)

        self.envs = gym.make_vec(env_name, num_envs=num_envs, max_episode_steps=max_episode_len)
    
        if isinstance(self.envs.action_space, gym.spaces.Box):
            self.action_space = "continuous"
        elif isinstance(self.envs.action_space, gym.spaces.Discrete):
            self.action_space = "discrete"
        elif isinstance(self.envs.action_space, gym.spaces.MultiDiscrete):
            self.action_space = "multidiscrete"
        else:
            raise EnvironmentError("What environment is this sir?")   
        self.action_range:tuple = (torch.tensor(self.envs.action_space.low, dtype=torch.float32, device=self.device), 
                                torch.tensor(self.envs.action_space.high, dtype=torch.float32, device=self.device)) if self.action_space == "continuous" else None
        self.action_scale = (self.action_range[1] - self.action_range[0]) / 2 if self.action_space == "continuous" else 1.0
        self.state_dim = self.envs.observation_space.shape[-1]
        self.action_dim = self.envs.action_space.shape[-1] if self.action_space == "continuous"else self.envs.action_space.n
        
        

        self.pi = Policy(self.state_dim, self.action_dim, self.action_space, networks_dimensions[0], networks_dimensions[1]).to(self.device)
        self.v = Value(self.state_dim, networks_dimensions[0], networks_dimensions[1]).to(self.device)
        self.pi_optim = Adam(self.pi.parameters(), lr=pi_lr)
        self.v_optim = Adam(self.v.parameters(), lr=vf_lr)
        self.state_normalizer = RunningNormalizer(self.state_dim, device=self.device)

        self.max_steps = max_steps
        self.relative_buffer_size = buffer_size // self.envs.num_envs
        self.gamma = gamma
        self.lam = lam
        self.beta = beta
        self.horizon=horizon
        self.clip_epsilon = clip_epsilon
        self.max_norm = max_norm
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.state_clip = state_clipping

        self.last_observation_input:torch.Tensor = None # normalized s

    def normalize_and_clip_state(self, state:torch.Tensor, update_online_normalizer:bool) -> torch.Tensor:
        if update_online_normalizer:
            self.state_normalizer.update(state)
        norm_state = self.state_normalizer.normalize(state)
        norm_clip_state = norm_state.clip(-self.state_clip, self.state_clip)
        return norm_clip_state
    
    def scale_actions_for_environment(self, action:torch.Tensor):
        if self.action_space != "continuous":
            return action
        return action * self.action_scale 

    def collect_trajectories(self, steps_COUNT, episodes_COUNT):
        buffer = Buffer(self.relative_buffer_size, self.envs.num_envs, self.state_dim, self.action_dim, self.device)

        if self.last_observation_input is None:
            self.last_observation_input = torch.tensor(self.envs.reset()[0], dtype=torch.float32, device=self.device)
            self.last_observation_input = self.normalize_and_clip_state(self.last_observation_input, update_online_normalizer=True)
            buffer.states[0] = self.last_observation_input
            episodes_COUNT += self.envs.num_envs

        # shape [ENVS, T]
        episode_rewards = [[] for _ in range(self.envs.num_envs)]


        for timestep_ in range(self.relative_buffer_size):
            with torch.no_grad():
                if self.action_space == "continuous":
                    mu_t, log_std_t = self.pi.forward(self.last_observation_input)
                    torch.clip_(log_std_t, -3, 3)
                    _dist = Independent(Normal(mu_t, log_std_t.exp()), 1)
                elif self.action_space == "discrete":
                    logits = self.pi.forward(self.last_observation_input)
                    _dist = Categorical(logits=logits)    
                else:
                    raise "Bro I did not implemented for Multivariate Discrete environments"

                actions_parallel_batch = _dist.sample()
                log_probs_parallel_batch = _dist.log_prob(actions_parallel_batch)
                # Take action and unpack data
                STEPS_batch = self.envs.step(self.scale_actions_for_environment(actions_parallel_batch).cpu().numpy()) # new state, reward, done, _ 
                rewards_parallel_batch = torch.tensor(STEPS_batch[1], dtype=torch.float32, device=self.device)  
                dones_parallel_batch = torch.tensor(STEPS_batch[2], dtype=torch.float32, device=self.device)
                next_states_parallel_batch = torch.tensor(STEPS_batch[0], dtype=torch.float32, device=self.device)
                next_states_parallel_batch = self.normalize_and_clip_state(next_states_parallel_batch, True)

                # update current tstep data
                buffer.actions[timestep_] = actions_parallel_batch if self.action_space == "continuous" else F.one_hot(actions_parallel_batch, self.action_dim)
                buffer.log_probs[timestep_] = log_probs_parallel_batch.unsqueeze(-1)
                buffer.next_states[timestep_] = next_states_parallel_batch
                buffer.rewards[timestep_] = rewards_parallel_batch.unsqueeze(-1)
                buffer.dones[timestep_] = dones_parallel_batch.unsqueeze(-1)

                for i in range(self.envs.num_envs):
                    episode_rewards[i].append(rewards_parallel_batch[i].item())

                if dones_parallel_batch.any():
                    done_indices = torch.where(dones_parallel_batch)[0]
                    for idx in done_indices:
                        log_episode(
                            sum(episode_rewards[idx]), 
                            episode_length=len(episode_rewards[idx]), 
                            step=(steps_COUNT, self.max_steps)
                        )
                        episodes_COUNT += 1
                        episode_rewards[idx].clear()
                    

                if timestep_ < self.relative_buffer_size - 1:
                    self.last_observation_input = next_states_parallel_batch
                    buffer.states[timestep_+1] = next_states_parallel_batch

                steps_COUNT += self.envs.num_envs

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
            with torch.no_grad():
                values_plus_1 = self.v(torch.concat([buffer.states, buffer.next_states[-1:, :]], dim=0).to(self.device)).cpu() # T+1, 1
                buffer.values = values_plus_1[:-1]
                
                for timestep_ in range(buffer.values.size(0)):
                    discount = 1.0
                    ahat_t = 0.0
                    for i in range(timestep_, buffer.values.size(0)):
                        r_t = buffer.rewards[i]
                        V_targ_batch = values_plus_1[i]
                        v_next_t = 0.0 if buffer.dones[i].item() == 1 else values_plus_1[i+1]
                        delta = r_t + self.gamma * v_next_t - V_targ_batch
                        ahat_t += discount * delta
                        discount *= self.gamma * self.lam
                        if buffer.dones[i].item() == 1:
                            break
                        if i + 1 == min(timestep_+ self.horizon, buffer.values.size(0)):
                            break
                        
                    buffer.value_targets[timestep_] = ahat_t + buffer.values[timestep_]
                    buffer.advantages[timestep_] = ahat_t

            # Todo - verify where the fuck is the overhead, from tests it seems like it appears here
            # Update policy -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
            for k in range(self.num_epoch):
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
                    torch.nn.utils.clip_grad_norm_(self.v.parameters(), max_norm=self.max_norm)
                    torch.nn.utils.clip_grad_norm_(self.pi.parameters(), max_norm=self.max_norm)
                    self.v_optim.step()
                    self.pi_optim.step() # step is the problem overhead, verified really strong

        print()
        self._make_checkpoint(0)
        display_episodes()

    def _make_checkpoint(self, idx:int):
        if not os.path.exists("./checkpoints"):
            os.mkdir("./checkpoints")

        if not os.path.exists(f"./checkpoints/{__name__}"):
            os.mkdir(f"./checkpoints/{__name__}")

        if not os.path.exists(f"./checkpoints/{__name__}/{self.envs.spec.id}"):
            os.mkdir(f"./checkpoints/{__name__}/{self.envs.spec.id}")

        torch.save(
            {
                "pi":self.pi.state_dict(),
                "v":self.v.state_dict(),
                "pi_optim":self.pi.state_dict(),
                "v_optim":self.v_optim.state_dict(),
            },
            f= f"./checkpoints/{__name__}/{self.envs.spec.id}/checkpoint_{idx}.pth"
        )
        print(f"Checkpoint {idx} done (./checkpoints/{__name__}/{self.envs.spec.id})")

        
class Buffer:
    def __init__(self, dim:int, envs_num:int, state_dim:int, action_dim:int, device) -> None:
        self.states =        torch.zeros(dim, envs_num, state_dim, device=device)
        self.next_states =   torch.zeros(dim, envs_num, state_dim, device=device)
        self.actions =       torch.zeros(dim, envs_num, action_dim, device=device)
        self.log_probs =     torch.zeros(dim, envs_num, 1, device=device)
        self.rewards =       torch.zeros(dim, envs_num, 1, device=device)
        self.dones =         torch.zeros(dim, envs_num, 1, device=device)
        self.values =        torch.zeros(dim, envs_num, 1, device=device)
        self.value_targets = torch.zeros(dim, envs_num, 1, device=device)
        self.advantages =    torch.zeros(dim, envs_num, 1, device=device)

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
