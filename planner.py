import torch
import torch.nn as nn
from typing import Tuple


class VAE(nn.Module):
    def __init__(self, state_dim, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_log_var = nn.Linear(32, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim)
        )
    
    def encode(self, x)-> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x, n_samples=1):
        mu, log_var = self.encode(x)
        
        # Expanding mu and log_var to match the batch size (n_samples)
        mu = mu.unsqueeze(0).expand(n_samples, -1)  # Expand on the sample dimension (first dimension)
        log_var = log_var.unsqueeze(0).expand(n_samples, -1)  # Similarly expand log_var

        z = self.reparameterize(mu, log_var)  # Use i-th sample for reparameterization
        samples=self.decode(z)

        return samples, mu, log_var
