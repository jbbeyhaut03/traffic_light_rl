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
        # Justification: Queue limits (0-20) balance realism and simplicity; light_state is binary (0 or 1).
        
        # Action: 0 = keep current light, 1 = switch light
        self.action_space = spaces.Discrete(2)
        # Justification: Discrete actions simplify control; "keep" vs. "switch" is intuitive for traffic lights.
        
        # Environment parameters
        self.max_queue = 20  # Cap queue to avoid infinite growth
        # Justification: Prevents state explosion, keeps environment manageable.
        self.arrival_rate = 0.3  # Probability of vehicle arriving per direction per step
        # Justification: Moderate rate simulates realistic traffic, allows queues to form/clear.
        self.departure_rate = 3  # Vehicles cleared per green light per step
        # Justification: Ensures green light has impact, balances reward dynamics.
        self.max_steps = 200  # Episode length
        # Justification: Long enough for learning, short enough for quick training.
        
        # State variables
        self.state = None
        self.step_count = 0
        self.last_light = None
        
        # Visualization
        self.history = []  # Track queues for plotting
        # Justification: Enables assessment of performance over time.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize queues to random low values, light to NS green (0)
        self.state = np.array([np.random.randint(0, 5) for _ in range(4)] + [0], dtype=np.float32)
        self.step_count = 0
        self.last_light = 0
        self.history = []
        # Justification: Random initial queues add variability; NS green as default is arbitrary but consistent.
        return self.state, {}

    def step(self, action):
        queues, light = self.state[:-1], int(self.state[-1])
        self.step_count += 1
        
        # Apply action: 0=keep, 1=switch
        if action == 1:
            light = 1 - light  # Toggle between 0 (NS green) and 1 (EW green)
        # Justification: Simple toggle reflects traffic light mechanics.
        
        # Update queues
        new_queues = queues.copy()
        # Arrivals (Poisson-like)
        for i in range(4):
            new_queues[i] += np.random.binomial(1, self.arrival_rate)
            new_queues[i] = min(new_queues[i], self.max_queue)
        # Justification: Binomial approximates vehicle arrivals; cap prevents overflow.
        
        # Departures based on light
        if light == 0:  # NS green, EW red
            new_queues[0] = max(0, new_queues[0] - self.departure_rate)  # N
            new_queues[1] = max(0, new_queues[1] - self.departure_rate)  # S
        else:  # EW green, NS red
            new_queues[2] = max(0, new_queues[2] - self.departure_rate)  # E
            new_queues[3] = max(0, new_queues[3] - self.departure_rate)  # W
        # Justification: Departure only on green directions; rate ensures progress.
        
        # Update state
        self.state = np.append(new_queues, light).astype(np.float32)
        
        # Reward
        reward = -sum(new_queues)  # -1 per waiting vehicle
        if sum(new_queues) == 0:
            reward += 10  # Bonus for clearing all queues
        if action == 1 and self.last_light != light:
            reward -= 5  # Penalty for switching
        # Justification: Negative reward incentivizes short queues; bonus rewards goal; penalty discourages frequent switches.
        
        self.last_light = light
        self.history.append(new_queues.copy())
        
        # Termination
        terminated = self.step_count >= self.max_steps
        truncated = False
        # Justification: Fixed episode length ensures training stability.
        
        return self.state, reward, terminated, truncated, {}

    def render(self):
        if len(self.history) == 0:
            return
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
        plt.pause(0.01)
        # Justification: Visualizes queue dynamics, aids assessment and debugging.

    def close(self):
        plt.close()
        # Justification: Ensures clean shutdown of visualization.

# Test environment
if __name__ == "__main__":
    env = TrafficLightEnv()
    env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        state, reward, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            break
    env.close()