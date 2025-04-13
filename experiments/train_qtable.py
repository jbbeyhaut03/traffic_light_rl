import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

# Add the project root to the Python path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.traffic_light_env import TrafficLightEnv

# ------------------------------
# Discretization function
# ------------------------------
def discretize_state(obs):
    """
    Convert the observation (a numpy array of floats) into a discrete tuple.
    Observation format: [queue_N, queue_S, queue_E, queue_W, light_state]
    """
    q_n = int(round(obs[0]))
    q_s = int(round(obs[1]))
    q_e = int(round(obs[2]))
    q_w = int(round(obs[3]))
    light = int(round(obs[4]))
    return (q_n, q_s, q_e, q_w, light)

# ------------------------------
# Q-learning Hyperparameters
# ------------------------------
num_episodes = 5000          # Total episodes for training
alpha = 0.1                  # Learning rate
gamma = 0.99                 # Discount factor
epsilon_start = 1.0          # Initial exploration rate
epsilon_min = 0.1            # Minimum exploration rate

# Function for linear epsilon decay over the episodes.
def get_epsilon(episode):
    return max(epsilon_min, epsilon_start - (epsilon_start - epsilon_min) * (episode / num_episodes))

# ------------------------------
# Initialize environment and Q-table
# ------------------------------
env = TrafficLightEnv()
# The Q-table is a dict mapping state tuples to an array of Q-values (length equal to number of actions).
Q = defaultdict(lambda: np.zeros(env.action_space.n))

# Lists for tracking rewards
all_step_rewards = []      # Record each step's reward
window_rewards = []        # Average reward over a window (e.g., last 100 steps)
check_freq = 1000          # Every 1000 steps we compute the moving average
window_size = 100          # Window size for averaging step rewards
global_step = 0

print("Starting Q-learning training with step reward tracking...")

for ep in range(num_episodes):
    obs, _ = env.reset()
    state = discretize_state(obs)
    epsilon = get_epsilon(ep)
    done = False

    while not done:
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        new_obs, reward, terminated, truncated, info = env.step(action)
        new_state = discretize_state(new_obs)
        
        # Q-learning update
        td_target = reward + gamma * np.max(Q[new_state])
        td_delta = td_target - Q[state][action]
        Q[state][action] += alpha * td_delta
        
        # Record the step reward and update global step count
        all_step_rewards.append(reward)
        global_step += 1

        # Every check_freq steps, compute the average of the last window_size steps.
        if global_step % check_freq == 0 and len(all_step_rewards) >= window_size:
            avg_recent = np.mean(all_step_rewards[-window_size:])
            window_rewards.append(avg_recent)
            print(f"Global step {global_step}: Avg step reward (last {window_size}) = {avg_recent:.2f}")

        state = new_state
        done = terminated or truncated

env.close()

# ------------------------------
# Save Q-table and Training Plot (Step Reward Curve)
# ------------------------------
results_dir = os.path.join("results", "qtable")
os.makedirs(results_dir, exist_ok=True)

# Save Q-table
qtable_path = os.path.join(results_dir, "qtable.pkl")
with open(qtable_path, "wb") as f:
    pickle.dump(dict(Q), f)
print(f"Q-table saved to {qtable_path}")

# Plot and save the step reward moving average curve.
plt.figure()
plt.plot(np.arange(len(window_rewards)) * check_freq, window_rewards, label="Mean Step Reward (last 100 steps)")
plt.xlabel("Training Steps")
plt.ylabel("Mean Reward")
plt.title("Q-learning Training Step Reward Curve")
plt.legend()
plot_path = os.path.join(results_dir, "training_reward_curve.png")
plt.savefig(plot_path)
plt.close()
print(f"Training reward curve saved to {plot_path}")
