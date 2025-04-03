from trainers.PPOparallel import PPO
from trainers.SACparallel import SAC
from flashml.tools import resource_monitor
from trainers.DDPG import DDPG
from networks.policy import Policy
from torchinfo import summary
import torch
if __name__ == "__main__":
    # PPO("InvertedPendulum-v5", max_steps=100_000, buffer_size=2048, batch_size=128).train()
    SAC( "InvertedPendulum-v5", max_steps=100_000).train()
    # DDPG( "InvertedPendulum-v5", max_steps=50_000, act_noise=0.1, lr=1e-4, tau=0.001, update_every=50).train()
    #ppo = PPO(
    #    "InvertedPendulum-v5",
    #    # num_envs=16,
    #    buffer_size=1024,
    #    batch_size= 128,
    #    max_steps=16_384,
    #)
# #
    #ppo.train()

    