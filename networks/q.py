import torch
import torch.nn as nn
from flashml.modules.Rish import Rish
from flashml.modules import RMSNorm


class Q(nn.Module):
    def __init__(self, state_dim, action_dim,layer_num:int=2, hidden_dim:int=128):
        '''
        output is 1
        '''
        super().__init__()
        layers = []
        layers.append(nn.Linear(state_dim + action_dim, hidden_dim))
        layers.append(RMSNorm(hidden_dim))
        layers.append(torch.nn.SiLU())
        
        for _ in range(layer_num - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(RMSNorm(hidden_dim))
            layers.append(torch.nn.SiLU())
            
        layers.append(nn.Linear(hidden_dim, 1))

        for i in layers:
            if isinstance(i, nn.Linear):
                nn.init.uniform_(i.weight, -3e-3, 3e-3)
                nn.init.zeros_(i.bias)
                
        self.net = nn.Sequential(*layers)

        
    
    def forward(self, s, a):
        return self.net(torch.cat([s, a], -1))   