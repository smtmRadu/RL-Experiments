import torch
import torch.nn as nn
from typing import Tuple
from flashml.modules import RMSNorm

class Policy(nn.Module):
    '''
    PI network only returns real numbers.
    '''
    def __init__(self, state_dim:int, num_actions:int, action_type:str, layer_num:int=2, hidden_dim:int=128):
        super().__init__()
        self.input_size = state_dim
        self.action_type = action_type
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(RMSNorm(hidden_dim))
        layers.append(torch.nn.SiLU())
       

        for _ in range(layer_num - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(RMSNorm(hidden_dim))
            layers.append(torch.nn.SiLU())
            
        layers.append(nn.Linear(hidden_dim, num_actions * 2) if action_type == "continuous" else nn.Linear(hidden_dim, num_actions))


        for i in layers:
            if isinstance(i, nn.Linear):
                nn.init.uniform_(i.weight, -3e-3, 3e-3)
                nn.init.zeros_(i.bias)
                
        self.net = nn.Sequential(*layers)

       

    
    def forward(self, x) -> torch.tensor | Tuple[torch.tensor, torch.tensor]:
        
        '''
        returns logits for discrete \\
        returns mu and logstd for continuous
        '''

        assert x.size(-1) == self.input_size
        output =  self.net(x)
        return torch.chunk(output, 2, -1) if self.action_type == "continuous" else output
    