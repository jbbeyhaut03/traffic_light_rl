import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
import math

class TrafficLightEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Updated State: [queue_N, queue_S, queue_E, queue_W, light_state (0=NS green, 1=EW green), time_of_day]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([20, 20, 20, 20, 1, 1]),
            dtype=np.float32
        )
        # Action: 0 = keep current light, 1 = switch light
        self.action_space = spaces.Discrete(2)
        
        # Environment parameters
        self.max_queue = 20           # Maximum queue per direction
        self.departure_rate = 3       # Vehicles cleared per green light per step
        self.max_steps = 200          # Episode length
        
        # Time-dependent simulation parameters
        self.cycle_length = 200       # Number of steps representing one complete “day”
        
        # Base arrival probabilities (directional biases)
        # Weekdays tend to have higher traffic on North/South than East/West.
        self.base_arrivals_weekday = [0.3, 0.3, 0.2, 0.2]
        # Weekends might be lighter overall.
        self.base_arrivals_weekend = [0.2, 0.2, 0.1, 0.1]
        
        # By default use weekday scenario; will be set randomly on reset.
        self.scenario = 'weekday'
        self.base_arrivals = self.base_arrivals_weekday
        
        # State variables
        self.state = None
        self.step_count = 0
        self.last_light = None
        
        # For rendering
        self.history = []  # Tracking queue lengths for plotting

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomly select a traffic scenario for this episode.
        self.scenario = np.random.choice(['weekday', 'weekend'])
        self.base_arrivals = (self.base_arrivals_weekday if self.scenario == 'weekday' 
                              else self.base_arrivals_weekend)
        
        # Initialize queues with small random values; light state fixed (0 = NS green);
        # and initial time-of-day is 0.
        initial_queues = [np.random.randint(0, 5) for _ in range(4)]
        initial_light = 0
        time_of_day = 0.0
        self.state = np.array(initial_queues + [initial_light, time_of_day], dtype=np.float32)
        
        self.step_count = 0
        self.last_light = initial_light
        self.history = []
        return self.state, {}

    def step(self, action):
        # Unpack state: queues (first 4 elements), current light (5th element),
        # and time-of-day (6th element, though we update it every step).
        queues = self.state[:4].copy()
        current_light = int(self.state[4])
        initial_light = current_light  # For reference before applying action
        
        self.step_count += 1
        
        # --- Action Handling ---
        # If action is 1, toggle the light.
        switched = False
        if action == 1:
            current_light = 1 - current_light
            switched = True
        
        # --- Queue Updates: Arrivals ---
        new_queues = queues.copy()
        # Create a time-of-day factor using a sine wave (simulates rush-hour patterns)
        time_factor = math.sin(2 * math.pi * (self.step_count % self.cycle_length) / self.cycle_length)
        normalized_time_factor = (time_factor + 1) / 2  # Scales to range 0 to 1
        
        # Update each queue with direction-specific arrival rates,
        # adjusted by the time-of-day factor.
        for i in range(4):
            # For instance, add up to 0.2 extra arrival probability during peak hours.
            current_arrival_rate = min(self.base_arrivals[i] + 0.2 * normalized_time_factor, 1.0)
            arrivals = np.random.binomial(1, current_arrival_rate)
            new_queues[i] += arrivals
            new_queues[i] = min(new_queues[i], self.max_queue)
        
        # --- Process Departures ---
        throughput = 0  # Count of vehicles cleared this step
        # If light is NS green (0), then only North & South get to clear; otherwise East & West.
        if current_light == 0:
            served_indices = [0, 1]
        else:
            served_indices = [2, 3]
        for i in served_indices:
            cleared = min(new_queues[i], self.departure_rate)
            throughput += cleared
            new_queues[i] -= cleared
        
        # Update the normalized time-of-day (value between 0 and 1)
        time_of_day = (self.step_count % self.cycle_length) / self.cycle_length
        
        # --- Update the State ---
        # New state includes updated queues, current light state, and time-of-day.
        self.state = np.array(new_queues.tolist() + [current_light, time_of_day], dtype=np.float32)
        self.history.append(new_queues.copy())
        
        # --- Reward Calculation ---
        total_queue = sum(new_queues)
        reward = -total_queue * 0.1 + 1.0 * throughput - 0.5 * int(switched)
        
        self.last_light = current_light
        
        # --- Termination ---
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        # --- Info Dictionary (including switched flag for consistency) ---
        info = {
            "throughput": throughput,    # Vehicles cleared this step
            "switched": int(switched)      # 1 if the light switched on this step, 0 otherwise
        }
        
        return self.state, reward, terminated, truncated, info

    def render(self):
        if not self.history:
            return
        plt.ion()  # Interactive mode
        queues = np.array(self.history)
        plt.clf()
        plt.plot(queues[:, 0], label="North")
        plt.plot(queues[:, 1], label="South")
        plt.plot(queues[:, 2], label="East")
        plt.plot(queues[:, 3], label="West")
        plt.xlabel("Step")
        plt.ylabel("Queue Length")
        plt.title("Traffic Queues Over Time")
        plt.legend()
        plt.draw()
        plt.pause(0.01)

    def close(self):
        plt.close()

# Test the updated environment
if __name__ == "__main__":
    env = TrafficLightEnv()
    state, _ = env.reset()
    print(f"Scenario: {env.scenario}")  # Print which scenario was chosen for this episode.
    for _ in range(100):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        print(f"Step: {env.step_count} | Reward: {reward:.2f} | Throughput: {info['throughput']} | "
              f"Switched: {info['switched']} | State: {state}")
        env.render()
        plt.pause(0.1)
        if terminated or truncated:
            break
    plt.show()  # Keep window open at the end
    env.close()
