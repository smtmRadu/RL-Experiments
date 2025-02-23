from trainers.PPOparallel import PPO

if __name__ == "__main__":
    ppo = PPO(
        "InvertedPendulum-v5",
        # num_envs=16,
        buffer_size=2048,
        batch_size= 128,
        max_steps=10_000,
    )

    ppo.train()