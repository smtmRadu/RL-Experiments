import torch
import torch.nn as nn
from typing import Sequence

class RunningNormalizer(nn.Module):
    def __init__(self, *size: Sequence[int], eps:float=1e-8, device:str = "cpu") -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(size, device=device))
        self.register_buffer("m2", torch.zeros(size, device=device))
        self.register_buffer("step", torch.tensor(0, dtype=torch.long, device=device))
        self.register_buffer("eps", torch.tensor(eps, dtype=torch.float, device=device))
    
    def normalize(self, x:torch.Tensor) -> torch.Tensor:
        if self.step == 0:
            raise "Update before normalizing sir"
        if self.step <= 1:
            return torch.clone(x)
        
        variance = self.m2 / (self.step - 1)
        variance[variance == 0] = 1

        return (x - self.mean) / (variance + self.eps).sqrt()
    
    def update(self, x: torch.Tensor) -> None:
        if x.dim() == self.mean.dim():
            self.step += 1
            delta1 = x - self.mean
            self.mean += delta1 / self.step
            delta2 = x - self.mean
            self.m2 += delta1 * delta2
        elif x.dim() == self.mean.dim() + 1:
            batch_n = x.size(0)
            batch_mean = x.mean(dim=0).to(x.device)
            batch_var = x.var(dim=0, unbiased=False).to(x.device)  
            
            total_n = self.step + batch_n
            delta = batch_mean - self.mean
            new_mean = (self.mean * self.step + batch_mean * batch_n) / total_n

            self.m2 += batch_var * batch_n + (delta ** 2) * self.step * batch_n / total_n
            
            self.mean = new_mean
            self.step = total_n
        else:
            raise ValueError(f"Too big shape (recv ({x.shape}), initial shape ({self.mean.shape}))")
    
    def to(self, device):
        self.mean = self.mean.to(device)
        self.m2 = self.m2.to(device)