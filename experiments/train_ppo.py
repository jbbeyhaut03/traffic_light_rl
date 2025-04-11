import sys
import os
# Add the project root (the parent directory) to the Python path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from src.traffic_light_env import TrafficLightEnv

class RewardCallback(BaseCallback):
    def __init__(self, check_freq: int, save_path: str):
        super().__init__()
        self.check_freq = check_freq    # How often to update the reward curve.
        self.save_path = save_path      # Where to save the plot image.
        self.rewards = []               # For reward curve plotting.
        self.all_rewards = []           # NEW: To store individual episode rewards.

    def _on_step(self) -> bool:
        # Record every reward from the rollout into all_rewards.
        if "rewards" in self.locals:
            self.all_rewards.extend(self.locals["rewards"].tolist())
        
        # Update the reward curve every check_freq steps.
        if self.n_calls % self.check_freq == 0:
            episode_rewards = self.locals["rewards"]
            if episode_rewards:
                self.rewards.append(np.mean(episode_rewards))
            plt.clf()
            plt.plot(self.rewards, label="Mean Episode Reward")
            plt.xlabel("Training Step (x1000)")
            plt.ylabel("Mean Reward")
            plt.title("PPO Training Reward Curve")
            plt.legend()
            plt.savefig(os.path.join(self.save_path, "reward_curve.png"))
        return True

# Instantiate the environment.
env = TrafficLightEnv()
# Check if the environment conforms to Gymnasium's API.
check_env(env)

# Define training parameters.
timesteps = 300_000
learning_rates = [1e-4, 3e-4]

# Ensure the base directory for PPO results exists.
os.makedirs("results/ppo", exist_ok=True)

for lr in learning_rates:
    print(f"Training PPO with learning rate: {lr}")
    # Create results directory for this learning rate.
    lr_save_path = os.path.join("results", "ppo", f"lr_{lr}")
    os.makedirs(lr_save_path, exist_ok=True)
    
    # Initialize the PPO model.
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        verbose=1,
        tensorboard_log="results/ppo/tensorboard/"
    )
    
    # Set up the reward callback for visualization.
    callback = RewardCallback(check_freq=1000, save_path=lr_save_path)
    
    # Train the model.
    model.learn(total_timesteps=timesteps, callback=callback)

    # Plot a histogram of the last 100 episode rewards.
    rewards_to_plot = callback.all_rewards[-100:] if len(callback.all_rewards) >= 100 else callback.all_rewards
    plt.figure()
    plt.hist(rewards_to_plot, bins=20)
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Episode Rewards for LR {lr}")
    plt.savefig(os.path.join(lr_save_path, "reward_histogram.png"))
    plt.close()
    
    # Save the trained model as a zip file inside the learning rate folder.
    model_path = os.path.join(lr_save_path, f"ppo_lr_{lr}.zip")
    model.save(model_path)
    print(f"Model saved to {model_path}")

# Close the environment.
env.close()
