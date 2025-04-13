import sys
import os
# Add the project root to Python path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from src.traffic_light_env import TrafficLightEnv

def evaluate_model(model, num_episodes=50):
    """
    Evaluate a trained PPO model over multiple episodes.
    Returns the average reward, average queue length, maximum queue length,
    average throughput (vehicles cleared per step), switching rate (switches per step),
    and the representative queue history from the first episode.
    """
    rewards = []
    avg_queue_lengths = []
    max_queue_lengths = []
    throughputs = []  # Now stores throughput as vehicles cleared per step.
    switch_rates = []  # Stores rate of switches (switches per step) per episode.
    representative_history = None

    for ep in range(num_episodes):
        env = TrafficLightEnv()
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_queue_sum = 0.0
        ep_steps = 0
        ep_throughput = 0
        ep_switches = 0
        ep_max_queue = 0  # Track the maximum queue length in this episode.

        while not done:
            # Use deterministic action for evaluation.
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            # Sum of the queue lengths (first four elements of the observation).
            ep_queue_sum += np.sum(obs[:-1])
            # Update the maximum queue encountered in this episode.
            ep_max_queue = max(ep_max_queue, np.max(obs[:-1]))
            ep_steps += 1
            ep_throughput += info.get("throughput", 0)
            ep_switches += info.get("switched", 0)
            done = terminated or truncated

        rewards.append(ep_reward)
        avg_queue_lengths.append(ep_queue_sum / ep_steps)
        max_queue_lengths.append(ep_max_queue)
        # Now compute throughput as vehicles cleared per step.
        throughputs.append(ep_throughput / ep_steps)
        # Compute the switch rate (switches per step) for this episode.
        switch_rates.append(ep_switches / ep_steps)
        
        if representative_history is None:
            representative_history = env.history.copy()

        env.close()

    metrics = {
        "average_reward": np.mean(rewards),
        "average_queue_length": np.mean(avg_queue_lengths),
        "max_queue_length": np.mean(max_queue_lengths),
        "average_throughput": np.mean(throughputs),
        "average_switch_rate": np.mean(switch_rates)
    }
    return metrics, rewards, representative_history

def plot_traffic_flow(history, save_path):
    """Plots and saves the traffic queue dynamics over an evaluation episode."""
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
    # Evaluate the custom PPO model for each learning rate.
    # Use strings for the learning rates to match the folder names.
    custom_learning_rates = ["0.0001", "0.0002"]
    base_model_dir = os.path.join("results", "ppo_custom")
    
    # Dictionary to collect metrics for each learning rate.
    metrics_dict = {}

    for lr in custom_learning_rates:
        model_path = os.path.join(base_model_dir, f"custom_lr_{lr}", f"custom_ppo_lr_{lr}.zip")
        
        if not os.path.exists(model_path):
            print(f"Model file {model_path} does not exist. Skipping evaluation for custom PPO with lr {lr}.")
            continue

        print(f"Evaluating custom PPO model for learning rate: {lr}")
        model = PPO.load(model_path)
        
        metrics, eval_rewards, rep_history = evaluate_model(model, num_episodes=50)
        metrics_dict[lr] = metrics

        print(f"Evaluation Metrics for custom PPO with lr {lr}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
            
        # Create directory for evaluation outputs.
        eval_save_dir = os.path.join(base_model_dir, "evaluation", f"custom_lr_{lr}")
        os.makedirs(eval_save_dir, exist_ok=True)

        # Save reward histogram.
        plt.figure()
        plt.hist(eval_rewards, bins=20)
        plt.xlabel("Episode Reward")
        plt.ylabel("Frequency")
        plt.title(f"Evaluation Reward Histogram for Custom PPO (lr={lr})")
        hist_path = os.path.join(eval_save_dir, "reward_histogram.png")
        plt.savefig(hist_path)
        plt.close()
        print(f"Reward histogram saved to {hist_path}")

        # Save traffic flow plot.
        traffic_flow_path = os.path.join(eval_save_dir, "traffic_flow.png")
        plot_traffic_flow(rep_history, traffic_flow_path)
        print(f"Traffic flow plot saved to {traffic_flow_path}")

    # -------------------------------------------------------------------------
    # Save metrics summary table (aggregated over all evaluated custom PPO models)
    # -------------------------------------------------------------------------
    if metrics_dict:
        aggregated_metrics_path = os.path.join(base_model_dir, "evaluation", "custom_ppo_metrics.txt")
        with open(aggregated_metrics_path, "w") as f:
            f.write("Custom PPO Evaluation Metrics\n")
            f.write("=" * 50 + "\n")
            f.write("lr\tAvgReward\tAvgQueue\tMaxQueue\tAvgThroughput\tAvgSwitchRate\n")
            for lr, m in metrics_dict.items():
                f.write(f"{lr}\t{m['average_reward']:.2f}\t{m['average_queue_length']:.2f}\t"
                        f"{m['max_queue_length']:.2f}\t{m['average_throughput']:.2f}\t{m['average_switch_rate']:.2f}\n")
        print(f"Metrics table saved to {aggregated_metrics_path}")
