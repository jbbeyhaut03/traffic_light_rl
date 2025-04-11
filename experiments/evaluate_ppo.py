import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from src.traffic_light_env import TrafficLightEnv

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
            # Use deterministic actions for evaluation.
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
    # Use strings that match the actual naming from training.
    # In your training, the model was saved as:
    #   results/ppo/lr_0.0001/ppo_lr_0.0001.zip
    #   results/ppo/lr_0.0003/ppo_lr_0.0003.zip
    learning_rates = ["0.0001", "0.0003"]
    
    base_model_dir = os.path.join("results", "ppo")
    for lr_str in learning_rates:
        # Build the correct file path for the saved model.
        model_file = os.path.join(base_model_dir, f"lr_{lr_str}", f"ppo_lr_{lr_str}.zip")
        if not os.path.exists(model_file):
            print(f"Model file {model_file} does not exist. Skipping evaluation for learning rate {lr_str}.")
            continue

        print(f"Evaluating PPO model for learning rate: {lr_str}")
        model = PPO.load(model_file)
        
        # Evaluate the model over 50 episodes.
        metrics, eval_rewards, rep_history = evaluate_model(model, num_episodes=50)
        
        print(f"Evaluation Metrics for PPO with lr {lr_str}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        # Create an evaluation output folder (separate from training folders)
        eval_save_dir = os.path.join(base_model_dir, "evaluation", f"lr_{lr_str}")
        os.makedirs(eval_save_dir, exist_ok=True)
        
        # Plot and save a histogram of evaluation rewards.
        plt.figure()
        plt.hist(eval_rewards, bins=20)
        plt.xlabel("Episode Reward")
        plt.ylabel("Frequency")
        plt.title(f"Evaluation Reward Histogram for PPO lr {lr_str}")
        hist_path = os.path.join(eval_save_dir, "reward_histogram.png")
        plt.savefig(hist_path)
        plt.close()
        print(f"Reward histogram saved to {hist_path}")
        
        # Plot and save a traffic flow plot from a representative evaluation episode.
        traffic_flow_path = os.path.join(eval_save_dir, "traffic_flow.png")
        plot_traffic_flow(rep_history, traffic_flow_path)
        print(f"Traffic flow plot saved to {traffic_flow_path}")
