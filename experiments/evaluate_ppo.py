import sys
import os
# Add the project root to Python path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from src.traffic_light_env import TrafficLightEnv

def evaluate_model(model, env, num_episodes=50):
    rewards = []
    avg_queue_lengths = []
    max_queue_lengths = []
    throughputs = []
    switch_rates = []
    representative_history = None

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_queue_sum = 0.0
        ep_max_queue = 0.0
        ep_throughput = 0
        ep_switches = 0
        ep_steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            queues = obs[:4]  # Queue lengths
            ep_queue_sum += np.sum(queues)
            ep_max_queue = max(ep_max_queue, np.max(queues))
            ep_throughput += info["throughput"]
            ep_switches += info["switched"]
            ep_steps += 1
            done = terminated or truncated

        rewards.append(ep_reward)
        avg_queue_lengths.append(ep_queue_sum / ep_steps)
        max_queue_lengths.append(ep_max_queue)
        throughputs.append(ep_throughput / ep_steps)  # Per step
        switch_rates.append(ep_switches / ep_steps)   # Per step
        if ep == 0:
            representative_history = env.history.copy()

    metrics = {
        "average_reward": np.mean(rewards),
        "average_queue_length": np.mean(avg_queue_lengths),
        "max_queue_length": np.mean(max_queue_lengths),
        "average_throughput": np.mean(throughputs),
        "switch_rate": np.mean(switch_rates)
    }
    return metrics, rewards, representative_history

def save_plots(history, rewards, save_dir, lr_str):
    # Queue dynamics
    history_array = np.array(history)
    plt.figure()
    plt.plot(history_array[:, 0], label="North")
    plt.plot(history_array[:, 1], label="South")
    plt.plot(history_array[:, 2], label="East")
    plt.plot(history_array[:, 3], label="West")
    plt.xlabel("Step")
    plt.ylabel("Queue Length")
    plt.title(f"Traffic Queues (lr={lr_str})")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "traffic_flow.png"))
    plt.close()

    # Reward histogram
    plt.figure()
    plt.hist(rewards, bins=20)
    plt.xlabel("Episode Reward")
    plt.ylabel("Frequency")
    plt.title(f"Reward Distribution (lr={lr_str})")
    plt.savefig(os.path.join(save_dir, "reward_histogram.png"))
    plt.close()

def save_metrics_table(metrics_dict, base_dir):
    table_path = os.path.join(base_dir, "ppo_metrics.txt")
    with open(table_path, "w") as f:
        f.write("PPO Evaluation Metrics\n")
        f.write("=" * 50 + "\n")
        f.write("lr\tReward\tQueue\tMaxQueue\tThroughput\tSwitchRate\n")
        for lr, metrics in metrics_dict.items():
            f.write(f"{lr}\t{metrics['average_reward']:.2f}\t"
                    f"{metrics['average_queue_length']:.2f}\t"
                    f"{metrics['max_queue_length']:.2f}\t"
                    f"{metrics['average_throughput']:.2f}\t"
                    f"{metrics['switch_rate']:.3f}\n")
    print(f"Metrics table saved to {table_path}")

if __name__ == "__main__":
    learning_rates = ["0.0001", "0.0002"]
    base_model_dir = "results/ppo"
    metrics_dict = {}

    env = TrafficLightEnv()
    for lr_str in learning_rates:
        model_file = os.path.join(base_model_dir, f"lr_{lr_str}", f"ppo_lr_{lr_str}.zip")
        if not os.path.exists(model_file):
            print(f"Model {model_file} does not exist. Skipping.")
            continue

        print(f"Evaluating PPO (lr={lr_str})")
        model = PPO.load(model_file)
        metrics, rewards, history = evaluate_model(model, env, num_episodes=50)

        print(f"Metrics for lr={lr_str}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.2f}")
        metrics_dict[lr_str] = metrics

        eval_save_dir = os.path.join(base_model_dir, "evaluation", f"lr_{lr_str}")
        os.makedirs(eval_save_dir, exist_ok=True)
        save_plots(history, rewards, eval_save_dir, lr_str)

    save_metrics_table(metrics_dict, base_model_dir)
    env.close()