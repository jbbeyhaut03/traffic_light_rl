import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

from src.traffic_light_env import TrafficLightEnv

# ------------------------------------------------------------------------------
# Custom Callback to Record and Plot Training Rewards for DQN
# ------------------------------------------------------------------------------
class DQNRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, save_path: str):
        super().__init__()
        self.check_freq = check_freq    # Frequency for updating the plot
        self.save_path = save_path      # Path to save the reward curve plot
        self.rewards = []               # Mean rewards at check points
        self.all_rewards = []           # Accumulated rewards over training

    def _on_step(self) -> bool:
        # Record rewards from the step, if available
        if "rewards" in self.locals:
            rewards = self.locals["rewards"]
            if isinstance(rewards, np.ndarray):
                self.all_rewards.extend(rewards.tolist())
            else:
                self.all_rewards.append(rewards)

        if self.n_calls % self.check_freq == 0 and self.all_rewards:
            # Compute mean of the last 100 rewards for a smooth curve
            mean_reward = np.mean(self.all_rewards[-100:])
            self.rewards.append(mean_reward)
            plt.clf()
            plt.plot(self.rewards, label="Mean Reward (last 100 steps)")
            plt.xlabel(f"Training steps (x{self.check_freq})")
            plt.ylabel("Mean Reward")
            plt.title("DQN Training Reward Curve")
            plt.legend()
            plt.savefig(os.path.join(self.save_path, "reward_curve.png"))
        return True

# ------------------------------------------------------------------------------
# Main Training Execution for DQN
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Define two network architectures.
    architectures = {
        "64_64": [64, 64],
        "128_128": [128, 128]
    }
    
    total_timesteps = 300_000
    base_save_dir = os.path.join("results", "dqn")
    os.makedirs(base_save_dir, exist_ok=True)
    
    for arch_name, net_arch in architectures.items():
        print(f"Training DQN with network architecture: {net_arch}")
        
        # Directory to save results for current architecture.
        arch_save_path = os.path.join(base_save_dir, arch_name)
        os.makedirs(arch_save_path, exist_ok=True)
        
        # Initialize and check the environment.
        env = TrafficLightEnv()
        check_env(env)
        
        # Specify the policy network architecture.
        policy_kwargs = dict(net_arch=net_arch)
        
        # Initialize the DQN model.
        model = DQN(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=1e-4,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            target_update_interval=1000,
            verbose=1,
            tensorboard_log=os.path.join("results", "dqn", "tensorboard")
        )
        
        # Create the reward callback.
        callback = DQNRewardCallback(check_freq=1000, save_path=arch_save_path)
        
        # Train the model.
        model.learn(total_timesteps=total_timesteps, callback=callback)
        
        # Save the trained model.
        model_path = os.path.join(arch_save_path, f"dqn_arch_{arch_name}.zip")
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        env.close()
