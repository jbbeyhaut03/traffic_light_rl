import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt

class TrafficLightEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # State: [queue_N, queue_S, queue_E, queue_W, light_state (0=NS green, 1=EW green)]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([20, 20, 20, 20, 1]),
            dtype=np.float32
        )
        # Action: 0 = keep current light, 1 = switch light
        self.action_space = spaces.Discrete(2)
        
        # Environment parameters
        self.max_queue = 20              # Cap queue to avoid infinite growth
        self.arrival_rate = 0.3          # Probability of vehicle arriving per direction per step
        self.departure_rate = 3          # Vehicles cleared per green light per step
        self.max_steps = 200             # Episode length
        
        # State variables
        self.state = None
        self.step_count = 0
        self.last_light = None
        
        # Visualization
        self.history = []              # Track queues for plotting

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize queues to random low values, light to NS green (0)
        self.state = np.array([np.random.randint(0, 5) for _ in range(4)] + [0], dtype=np.float32)
        self.step_count = 0
        self.last_light = 0
        self.history = []
        return self.state, {}

    def step(self, action):
        # Unpack state: queues for four directions and the current light state.
        queues = self.state[:-1].copy()
        current_light = int(self.state[-1])
        initial_light = current_light  # For reference before action
        
        self.step_count += 1
        
        # --- Action handling: Switching the light ---
        switched = False
        if action == 1:
            current_light = 1 - current_light  # Toggle light (0 becomes 1, and vice versa)
            switched = True
        
        # --- Queue update: Arrivals ---
        new_queues = queues.copy()
        for i in range(4):
            arrivals = np.random.binomial(1, self.arrival_rate)
            new_queues[i] += arrivals
            new_queues[i] = min(new_queues[i], self.max_queue)
        
        # --- Departures: Clear vehicles from the queue in directions with a green light ---
        throughput = 0  # Count of vehicles cleared this step
        if current_light == 0:  # NS green, EW red
            served_indices = [0, 1]
        else:  # EW green, NS red
            served_indices = [2, 3]
        
        for i in served_indices:
            # Determine how many vehicles can actually clear (cannot clear more than present)
            cleared = min(new_queues[i], self.departure_rate)
            throughput += cleared
            new_queues[i] = new_queues[i] - cleared  # Update queue
        
        # --- Update state ---
        self.state = np.append(new_queues, current_light).astype(np.float32)
        
        # --- Reward Calculation ---
        total_queue = sum(new_queues)
        reward = - total_queue / (self.max_queue * 4)
        if total_queue < 5:
            reward += 2  # Bonus for very low total queue
        
        # Penalize switching slightly if it actually changed the light status from previous step.
        if action == 1 and self.last_light != current_light:
            reward -= 1
        
        self.last_light = current_light
        self.history.append(new_queues.copy())
        
        # --- Termination ---
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        # --- Additional Info ---
        info = {
            "throughput": throughput,    # Number of vehicles cleared during this step
            "switched": int(switched)      # Whether the light switched: 1 for yes, 0 for no
        }
        
        return self.state, reward, terminated, truncated, info

    def render(self):
        if len(self.history) == 0:
            return
        plt.ion()  # Enable interactive mode
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

# Test environment
if __name__ == "__main__":
    env = TrafficLightEnv()
    env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        print(f"Step info: throughput={info['throughput']}, switched={info['switched']}")
        env.render()
        plt.pause(0.1)  # Slow down to see updates
        if terminated or truncated:
            break
    plt.show()  # Keep window open at end
    env.close()
