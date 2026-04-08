from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from env.glucose_env import GlucoseEnv


def main():
    # Initialize the environment and wrap it with a Monitor for logging
    env = Monitor(GlucoseEnv(max_steps=1440))

    # Configure the DQN agent
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        verbose=1,
    )

    # Start the training process
    print("Starting training...")
    model.learn(total_timesteps=30000)
    
    # Save the trained model
    model.save("dqn_glucose_env")
    print("Model saved as dqn_glucose_env.zip")


if __name__ == "__main__":
    main()
