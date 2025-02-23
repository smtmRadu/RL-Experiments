import torch.nn as nn
from flashml.modules.Rish import Rish

class Value(nn.Module):
    def __init__(self, state_dim, layer_num:int=2, hidden_dim:int=128):
        '''
        output is 1
        '''
        super().__init__()
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(Rish())

        for _ in range(layer_num - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(Rish())

        layers.append(nn.Linear(hidden_dim, 1))

        for i in layers:
            if isinstance(i, nn.Linear):
                nn.init.kaiming_normal_(i.weight)
                nn.init.zeros_(i.bias)
                
        self.net = nn.Sequential(*layers)

        
    
    def forward(self, x):
        return self.net(x)