import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, state_dim:int, latent_dim:int) -> None:
        super(VAE, self).__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim

        self.down1 = nn.Linear(state_dim, (state_dim + latent_dim) * 2 /3)
        self.down2 = nn.Linear((state_dim + latent_dim) * 2 /3, (state_dim + latent_dim) * 1 /3)
        self.mu = nn.Linear((state_dim + latent_dim) * 1 /3, latent_dim)
        self.logvar = nn.Linear((state_dim + latent_dim) * 1 /3, latent_dim)
        self.up1 = nn.Linear(latent_dim, (state_dim + latent_dim) * 1 /3)
        self.up2 = nn.Linear((state_dim + latent_dim) * 1 /3, (state_dim + latent_dim) * 2 /3)
        self.out = nn.Linear((state_dim + latent_dim) * 2 /3, state_dim)
        

        # Init
        torch.nn.init.kaiming_normal_(self.down1.weight)
        torch.nn.init.kaiming_normal_(self.down2.weight)
        torch.nn.init.kaiming_normal_(self.mu.weight)
        torch.nn.init.kaiming_normal_(self.logvar.weight)
        torch.nn.init.kaiming_normal_(self.up1.weight)
        torch.nn.init.kaiming_normal_(self.up2.weight)
        torch.nn.init.kaiming_normal_(self.out.weight)

    def encode(self, x:torch.Tensor):
        x = F.silu(self.down1(x))
        x = F.silu(self.down2(x))
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar
    
    def decode(self, z:torch.Tensor) -> torch.Tensor:
        x = F.silu(self.up1(z))
        x = F.silu(self.up2(x))
        x = self.out(x)

    def forward(self, x:torch.Tensor) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        # returns recon_x, mu, logvar, eps
        mu, logvar = self.encode(x)

        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * mu
        out = self.decode(z)

        return out, mu, logvar, eps
