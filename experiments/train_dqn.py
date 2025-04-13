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



class DQNRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, save_path: str):
        super().__init__()
        self.check_freq = check_freq
        self.save_path = save_path
        self.rewards = []  # Mean rewards at check points
        self.all_rewards = []  # All step-wise rewards

    def _on_step(self) -> bool:
        # Record step-wise rewards
        if "rewards" in self.locals:
            rewards = self.locals["rewards"]
            if isinstance(rewards, np.ndarray):
                self.all_rewards.extend(rewards.tolist())
            else:
                self.all_rewards.append(rewards)

        # Plot mean of last 100 rewards every check_freq steps
        if self.n_calls % self.check_freq == 0 and self.all_rewards:
            mean_reward = np.mean(self.all_rewards[-100:])
            self.rewards.append(mean_reward)
            plt.clf()
            plt.plot(self.rewards, label="Mean Reward (last 100 steps)")
            plt.xlabel(f"Training Steps (x{self.check_freq})")
            plt.ylabel("Mean Reward")
            plt.title("DQN Training Reward Curve")
            plt.legend()
            plt.savefig(os.path.join(self.save_path, "reward_curve.png"))
        return True

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Define architectures and learning rates
    architectures = {
        "64_64": [64, 64],
        "128_128": [128, 128]
    }
    learning_rates = [1e-4, 3e-4]
    total_timesteps = 500_000

    base_save_dir = os.path.join("results", "dqn")
    os.makedirs(base_save_dir, exist_ok=True)

    env = TrafficLightEnv()
    check_env(env)

    for arch_name, net_arch in architectures.items():
        for lr in learning_rates:
            print(f"Training DQN with architecture: {arch_name}, lr: {lr}")
            save_dir = os.path.join(base_save_dir, f"arch_{arch_name}_lr_{lr}")
            os.makedirs(save_dir, exist_ok=True)

            policy_kwargs = dict(net_arch=net_arch)
            model = DQN(
                "MlpPolicy",
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=lr,
                buffer_size=50000,
                learning_starts=1000,
                batch_size=64,
                target_update_interval=1000,
                exploration_fraction=0.1,
                exploration_final_eps=0.02,
                verbose=1,
                tensorboard_log=os.path.join(base_save_dir, "tensorboard"),
                seed=42
            )

            callback = DQNRewardCallback(check_freq=1000, save_path=save_dir)
            model.learn(total_timesteps=total_timesteps, callback=callback)

            model_path = os.path.join(save_dir, f"dqn_arch_{arch_name}_lr_{lr}.zip")
            model.save(model_path)
            print(f"Model saved to {model_path}")

    env.close()