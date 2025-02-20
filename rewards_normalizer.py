import torch
import torch.nn as nn
from running_normalizer import RunningNormalizer


class RewardsNormalizer(nn.Module):
    def __init__(self, gamma:float =0.99, eps:float = 1e-8, device= 'cpu'):
        super(RewardsNormalizer, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ret_norm = RunningNormalizer(1,eps=eps, device=device)

        # these must not be saved
        self.returns = torch.zeros(1, 1)

    def normalize(self, parallel_rewards:torch.Tensor, parallel_dones:torch.Tensor, global_step:int):

        if global_step == 0 and parallel_rewards.numel() > 1: # adapt tensor to size of parallel envs
            self.returns = torch.zeros(parallel_rewards.numel(), 1) # keep it like this unsqueezed

        self.returns = self.returns * self.gamma * parallel_dones + parallel_rewards


        self.ret_norm.update(self.returns)
        
        if self.ret_norm.step == 0:
            return parallel_rewards
        else:
            return parallel_rewards / (self.ret_norm.m2 / (self.ret_norm.step + 1) + self.eps).sqrt()
