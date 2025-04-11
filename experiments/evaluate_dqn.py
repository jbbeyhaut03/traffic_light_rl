import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from src.traffic_light_env import TrafficLightEnv

# ------------------------------------------------------------------------------
# Evaluation Function for a Trained Model
# ------------------------------------------------------------------------------
def evaluate_model(model, num_episodes=50):
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
            # Use deterministic action selection for evaluation.
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_queue_sum += np.sum(obs[:-1])  # Sum of all queue lengths.
            ep_steps += 1
            ep_throughput += info.get("throughput", 0)
            ep_switches += info.get("switched", 0)
            done = terminated or truncated
        
        rewards.append(ep_reward)
        avg_queue_lengths.append(ep_queue_sum / ep_steps)
        total_throughputs.append(ep_throughput)
        total_switches.append(ep_switches)
        
        # Save history from the first episode for plotting traffic flow.
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

# ------------------------------------------------------------------------------
# Helper Function to Plot Traffic Flow from an Evaluation Episode
# ------------------------------------------------------------------------------
def plot_traffic_flow(history, save_path):
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

# ------------------------------------------------------------------------------
# Main Evaluation Execution
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # List the network architectures you want to evaluate.
    architectures = ["64_64", "128_128"]
    base_model_dir = os.path.join("results", "dqn")
    
    for arch in architectures:
        model_file = os.path.join(base_model_dir, arch, f"dqn_arch_{arch}.zip")
        if not os.path.exists(model_file):
            print(f"Model file {model_file} does not exist. Skipping {arch} architecture.")
            continue
        
        print(f"Evaluating model for architecture: {arch}")
        # Load the trained model.
        model = DQN.load(model_file)
        
        # Evaluate the model over 50 episodes.
        metrics, eval_rewards, rep_history = evaluate_model(model, num_episodes=50)
        
        # Print evaluation metrics.
        print(f"Evaluation Metrics for {arch}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        # Plot and save reward histogram.
        plt.figure()
        plt.hist(eval_rewards, bins=20)
        plt.xlabel("Episode Reward")
        plt.ylabel("Frequency")
        plt.title(f"Evaluation Reward Histogram for DQN {arch}")
        hist_path = os.path.join(base_model_dir, arch, "reward_histogram.png")
        plt.savefig(hist_path)
        plt.close()
        print(f"Reward histogram saved to {hist_path}")
        
        # Plot and save a traffic flow plot from a representative episode.
        traffic_flow_path = os.path.join(base_model_dir, arch, "traffic_flow.png")
        plot_traffic_flow(rep_history, traffic_flow_path)
        print(f"Traffic flow plot saved to {traffic_flow_path}")
