from trainers.PPO import PPO

if __name__ == "__main__":
    ppo = PPO(
        "InvertedPendulum-v5",
        buffer_size=1024,
        batch_size= 64,
        max_steps=10_000,
        horizon=64
    )

    ppo.test()