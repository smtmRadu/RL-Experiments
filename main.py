from trainers.PPOparallel import PPO
from trainers.SACparallel import SAC
from flashml.tools import resource_monitor
from trainers.DDPGparallel import DDPG
from networks.policy import Policy
from torchinfo import summary
from trainers.REDQparallel import REDQ
from trainers.TD3parallel import TD3
import torch
if __name__ == "__main__":
    # PPO("InvertedPendulum-v5", max_steps=100_000, buffer_size=2048, batch_size=128).train()
    # SAC( "InvertedPendulum-v5", max_steps=100_000).train()
    # DDPG( "InvertedPendulum-v5", num_envs=32, max_steps=40_000, act_noise=0.1, update_every=50).train()
    # TD3( "InvertedPendulum-v5", num_envs=8, max_steps=40_000, act_noise=0.1, update_every=50).train()
    #ppo = PPO(
    #    "InvertedPendulum-v5",
    #    # num_envs=16,
    #    buffer_size=1024,
    #    batch_size= 128,
    #    max_steps=16_384,
    #)
# #
    #ppo.train()
    REDQ( "InvertedPendulum-v5", max_steps=100_000, num_qnets_N=6, num_updates_G=3, num_qnets_subset_M=2).train() 

    