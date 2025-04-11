import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from src.traffic_light_env import TrafficLightEnv

def evaluate_model(model, num_episodes=50):
    """
    Runs a trained DQN model for num_episodes and computes:
      - Average episode reward
      - Average queue length (time-averaged over episodes)
      - Total throughput (vehicles cleared)
      - Total switching (how many times the light switched)
    Also returns the queue history from the first evaluation episode.
    """
    rewards = []
    avg_queue_lengths = []
    total_throughputs = []
    total_switches = []
    representative_history = None

    for ep in range(num_episodes):
        env = TrafficLightEnv()
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_queue_sum = 0
        ep_steps = 0
        ep_throughput = 0
        ep_switches = 0

        while not done:
            # Use deterministic action selection in evaluation.
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            ep_queue_sum += np.sum(obs[:-1])  # Sum of the 4 queue lengths.
            ep_steps += 1
            ep_throughput += info.get("throughput", 0)
            ep_switches += info.get("switched", 0)

            done = terminated or truncated

        rewards.append(ep_reward)
        avg_queue_lengths.append(ep_queue_sum / ep_steps)
        total_throughputs.append(ep_throughput)
        total_switches.append(ep_switches)

        # Save the history from the first episode for visualization.
        if representative_history is None:
            representative_history = env.history.copy()

        env.close()

    metrics = {
        "average_reward": np.mean(rewards),
        "average_queue_length": np.mean(avg_queue_lengths),
        "average_throughput": np.mean(total_throughputs),
        "average_switching": np.mean(total_switches)
    }
    return metrics, rewards, representative_history

def plot_traffic_flow(history, save_path):
    """Creates and saves a plot showing traffic queue dynamics over an episode."""
    history_array = np.array(history)
    plt.figure()
    plt.plot(history_array[:, 0], label="North")
    plt.plot(history_array[:, 1], label="South")
    plt.plot(history_array[:, 2], label="East")
    plt.plot(history_array[:, 3], label="West")
    plt.xlabel("Step")
    plt.ylabel("Queue Length")
    plt.title("Traffic Queues Over Time (Evaluation Episode)")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # Define your network architectures as used during training.
    # In your training, models for architecture 64_64 and 128_128 were saved in:
    #   results/dqn/64_64/dqn_arch_64_64.zip and results/dqn/128_128/dqn_arch_128_128.zip
    architectures = ["64_64", "128_128"]
    base_model_dir = os.path.join("results", "dqn")

    for arch in architectures:
        model_file = os.path.join(base_model_dir, arch, f"dqn_arch_{arch}.zip")
        if not os.path.exists(model_file):
            print(f"Model file {model_file} does not exist. Skipping evaluation for architecture {arch}.")
            continue

        print(f"Evaluating DQN model for network architecture: {arch}")
        model = DQN.load(model_file)

        # Evaluate the model over 50 episodes.
        metrics, eval_rewards, rep_history = evaluate_model(model, num_episodes=50)

        print(f"Evaluation Metrics for DQN {arch}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        # Create evaluation output folder in results/dqn/evaluation/{arch}
        eval_save_dir = os.path.join(base_model_dir, "evaluation", arch)
        os.makedirs(eval_save_dir, exist_ok=True)

        # Plot and save reward histogram.
        plt.figure()
        plt.hist(eval_rewards, bins=20)
        plt.xlabel("Episode Reward")
        plt.ylabel("Frequency")
        plt.title(f"Evaluation Reward Histogram for DQN {arch}")
        hist_path = os.path.join(eval_save_dir, "reward_histogram.png")
        plt.savefig(hist_path)
        plt.close()
        print(f"Reward histogram saved to {hist_path}")

        # Plot and save a traffic flow plot.
        traffic_flow_path = os.path.join(eval_save_dir, "traffic_flow.png")
        plot_traffic_flow(rep_history, traffic_flow_path)
        print(f"Traffic flow plot saved to {traffic_flow_path}")
