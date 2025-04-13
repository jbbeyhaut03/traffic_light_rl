import sys
import os
# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from src.traffic_light_env import TrafficLightEnv  # Updated to new environment file

class RewardCallback(BaseCallback):
    def __init__(self, check_freq: int, save_path: str):
        super().__init__()
        self.check_freq = check_freq
        self.save_path = save_path
        self.rewards = []
        self.all_rewards = []

    def _on_step(self) -> bool:
        # Record per-step rewards
        if "rewards" in self.locals:
            self.all_rewards.extend(self.locals["rewards"].tolist())
        
        # Plot mean reward every check_freq steps
        if self.n_calls % self.check_freq == 0:
            episode_rewards = self.locals["rewards"]
            if episode_rewards:
                self.rewards.append(np.mean(episode_rewards))
            plt.clf()
            plt.plot(self.rewards, label="Mean Reward")
            plt.xlabel("Training Step (x1000)")
            plt.ylabel("Mean Reward")
            plt.title("PPO Training Reward Curve")
            plt.legend()
            plt.savefig(os.path.join(self.save_path, "reward_curve.png"))
        return True

# Set random seed for reproducibility
np.random.seed(42)

# Instantiate the environment
env = TrafficLightEnv()
check_env(env)

# Define training parameters
timesteps = 500_000
learning_rates = [1e-4, 2e-4]  # Adjusted for new reward scale

# Ensure results directory exists
os.makedirs("results/ppo", exist_ok=True)

for lr in learning_rates:
    print(f"Training PPO with learning rate: {lr}")
    lr_save_path = os.path.join("results", "ppo", f"lr_{lr}")
    os.makedirs(lr_save_path, exist_ok=True)
    
    # Initialize PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        verbose=1,
        tensorboard_log="results/ppo/tensorboard/",
        seed=42  # Added for reproducibility
    )
    
    # Set up callback
    callback = RewardCallback(check_freq=1000, save_path=lr_save_path)
    
    # Train model
    model.learn(total_timesteps=timesteps, callback=callback)

    # Plot histogram of last 100 rewards
    rewards_to_plot = callback.all_rewards[-100:] if len(callback.all_rewards) >= 100 else callback.all_rewards
    plt.figure()
    plt.hist(rewards_to_plot, bins=20)
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Step Rewards for LR {lr}")
    plt.savefig(os.path.join(lr_save_path, "reward_histogram.png"))
    plt.close()
    
    # Save model
    model_path = os.path.join(lr_save_path, f"ppo_lr_{lr}.zip")
    model.save(model_path)
    print(f"Model saved to {model_path}")

env.close()