import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Add the project root to the Python path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.traffic_light_env import TrafficLightEnv

# ------------------------------
# Discretization function (same as in training)
# ------------------------------
def discretize_state(obs):
    q_n = int(round(obs[0]))
    q_s = int(round(obs[1]))
    q_e = int(round(obs[2]))
    q_w = int(round(obs[3]))
    light = int(round(obs[4]))
    return (q_n, q_s, q_e, q_w, light)

# ------------------------------
# Load Q-table from file
# ------------------------------
qtable_path = os.path.join("results", "qtable", "qtable.pkl")
if not os.path.exists(qtable_path):
    raise FileNotFoundError(f"Q-table file not found at {qtable_path}. Please run train_qtable.py first.")

with open(qtable_path, "rb") as f:
    Q = pickle.load(f)
print(f"Loaded Q-table from {qtable_path}")

# ------------------------------
# Evaluation of Q-table Policy
# ------------------------------
def evaluate_qtable(Q, env, num_episodes=50):
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
        ep_steps = 0
        ep_max_queue = 0
        ep_throughput = 0
        ep_switches = 0

        while not done:
            state = discretize_state(obs)
            # Greedy action: if state is unseen, default to action 0.
            if state in Q:
                action = np.argmax(Q[state])
            else:
                action = 0
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            queues = obs[:-1]  # first four elements: queue lengths
            ep_queue_sum += np.sum(queues)
            ep_max_queue = max(ep_max_queue, np.max(queues))
            ep_throughput += info.get("throughput", 0)
            ep_switches += info.get("switched", 0)
            ep_steps += 1
            done = terminated or truncated

        rewards.append(ep_reward)
        avg_queue_lengths.append(ep_queue_sum / ep_steps)
        max_queue_lengths.append(ep_max_queue)
        throughputs.append(ep_throughput / ep_steps)
        switch_rates.append(ep_switches / ep_steps)
        # Save the queue history from the first episode for plotting.
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

# ------------------------------
# Run Evaluation and Save Outputs
# ------------------------------
env = TrafficLightEnv()
num_eval_episodes = 50
metrics, rewards, rep_history = evaluate_qtable(Q, env, num_episodes=num_eval_episodes)

print("Evaluation Metrics:")
for key, value in metrics.items():
    print(f"  {key}: {value:.2f}")
env.close()

# Create directory to save evaluation results.
eval_dir = os.path.join("results", "qtable", "evaluation")
os.makedirs(eval_dir, exist_ok=True)

# Save traffic flow plot (using representative queue history from episode 0)
rep_history_array = np.array(rep_history)
plt.figure()
if rep_history_array.ndim == 2 and rep_history_array.shape[1] == 4:
    plt.plot(rep_history_array[:, 0], label="North")
    plt.plot(rep_history_array[:, 1], label="South")
    plt.plot(rep_history_array[:, 2], label="East")
    plt.plot(rep_history_array[:, 3], label="West")
else:
    plt.plot(rep_history_array, label="Queue Data")
plt.xlabel("Step")
plt.ylabel("Queue Length")
plt.title("Traffic Queues Over Time (Q-learning Evaluation)")
plt.legend()
traffic_flow_path = os.path.join(eval_dir, "traffic_flow.png")
plt.savefig(traffic_flow_path)
plt.close()
print(f"Traffic flow plot saved to {traffic_flow_path}")

# Save reward histogram.
plt.figure()
plt.hist(rewards, bins=20)
plt.xlabel("Episode Reward")
plt.ylabel("Frequency")
plt.title("Reward Distribution (Q-learning Evaluation)")
reward_hist_path = os.path.join(eval_dir, "reward_histogram.png")
plt.savefig(reward_hist_path)
plt.close()
print(f"Reward histogram saved to {reward_hist_path}")

# Save evaluation metrics to a text file.
metrics_file = os.path.join(eval_dir, "qtable_metrics.txt")
with open(metrics_file, "w") as f:
    f.write("Q-learning Evaluation Metrics\n")
    f.write("=" * 50 + "\n")
    f.write("AvgReward\tAvgQueue\tMaxQueue\tThroughput\tSwitchRate\n")
    f.write(f"{metrics['average_reward']:.2f}\t{metrics['average_queue_length']:.2f}\t"
            f"{metrics['max_queue_length']:.2f}\t{metrics['average_throughput']:.2f}\t"
            f"{metrics['switch_rate']:.3f}\n")
print(f"Evaluation metrics saved to {metrics_file}")
