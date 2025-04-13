import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
import math

class TrafficLightEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([20, 20, 20, 20, 1, 1]),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)
        
        self.max_queue = 20
        self.departure_rate = 3
        self.max_steps = 200
        self.cycle_length = 200
        
        # Arrival rates for weekdays (fixed)
        self.base_arrivals = [0.3, 0.3, 0.2, 0.2]  # NS > EW
        self.arrival_buffers = [0.0, 0.0, 0.0, 0.0]  # Used for fractional arrivals
        
        self.state = None
        self.step_count = 0
        self.last_light = None
        self.history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Always use "weekday"
        self.base_arrivals = [0.3, 0.3, 0.2, 0.2]
        self.arrival_buffers = [0.0, 0.0, 0.0, 0.0]
        
        # Fixed starting queues for repeatability
        initial_queues = [2, 2, 1, 1]
        initial_light = 0
        time_of_day = 0.0
        
        self.state = np.array(initial_queues + [initial_light, time_of_day], dtype=np.float32)
        self.step_count = 0
        self.last_light = initial_light
        self.history = []
        
        return self.state, {}

    def step(self, action):
        queues = self.state[:4].copy()
        current_light = int(self.state[4])
        self.step_count += 1

        # Light switch logic
        switched = False
        if action == 1:
            current_light = 1 - current_light
            switched = True

        # Time-of-day (0 to 1 based on step in cycle)
        time_of_day = (self.step_count % self.cycle_length) / self.cycle_length
        time_factor = math.sin(2 * math.pi * time_of_day)
        normalized_time_factor = (time_factor + 1) / 2

        # --- Deterministic Arrivals via Buffer ---
        for i in range(4):
            adjusted_rate = min(self.base_arrivals[i] + 0.2 * normalized_time_factor, 1.0)
            self.arrival_buffers[i] += adjusted_rate
            arrivals = int(self.arrival_buffers[i])
            self.arrival_buffers[i] -= arrivals
            queues[i] = min(queues[i] + arrivals, self.max_queue)

        # --- Departures ---
        throughput = 0
        served_indices = [0, 1] if current_light == 0 else [2, 3]
        for i in served_indices:
            cleared = min(queues[i], self.departure_rate)
            queues[i] -= cleared
            throughput += cleared

        self.state = np.array(queues.tolist() + [current_light, time_of_day], dtype=np.float32)
        self.history.append(queues.copy())

        total_queue = sum(queues)
        reward = -0.1 * total_queue + 1.0 * throughput - 0.5 * int(switched)

        terminated = self.step_count >= self.max_steps
        truncated = False

        info = {
            "throughput": throughput,
            "switched": int(switched)
        }

        return self.state, reward, terminated, truncated, info

    def render(self):
        if not self.history:
            return
        plt.ion()
        queues = np.array(self.history)
        plt.clf()
        plt.plot(queues[:, 0], label="North")
        plt.plot(queues[:, 1], label="South")
        plt.plot(queues[:, 2], label="East")
        plt.plot(queues[:, 3], label="West")
        plt.xlabel("Step")
        plt.ylabel("Queue Length")
        plt.title("Traffic Queues Over Time (Deterministic)")
        plt.legend()
        plt.draw()
        plt.pause(0.01)

    def close(self):
        plt.close()
