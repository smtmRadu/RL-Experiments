import torch
import torch.nn as nn
from normalizers.running_normalizer import RunningNormalizer


class RewardsNormalizer(nn.Module):
    def __init__(self, gamma:float =0.99, eps:float = 1e-8, device= 'cpu'):
        super(RewardsNormalizer, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ret_norm = RunningNormalizer(1,eps=eps, device=device)
        self.device = device
        # these must not be saved
        self._returns = None

    def normalize(self, parallel_rewards:torch.Tensor, parallel_dones:torch.Tensor):
        '''
        parallel_rewards: (Env, 1) or (1)
        parallel_dones: (Env, 1) or (1)
        '''
        # assert parallel_rewards.size(-1) == 1, f"Parallel Rewards shape must be (Envs_num, 1) or (1) (received: {tuple(parallel_rewards.shape)})"
        # assert parallel_dones.size(-1) == 1, f"Parallel Dones shape must be (Envs_num, 1) or (1) (received: {tuple(parallel_dones.shape)})"
        
        if self._returns is None: # adapt tensor to size of parallel envs
            self._returns = torch.zeros(parallel_rewards.numel(),1, device=self.device) # keep it like this unsqueezed

        self._returns = self._returns * self.gamma * (1-parallel_dones) + parallel_rewards

        self.ret_norm.update(self._returns)
        
        if self.ret_norm.step == 0:
            return parallel_rewards
        else:
            return parallel_rewards / (self.ret_norm.m2 / (self.ret_norm.step + 1) + self.eps).sqrt()
