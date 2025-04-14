# 🚦 Traffic Light Optimization with Reinforcement Learning

This project explores how different reinforcement learning algorithms (Q-learning, DQN, PPO) can learn to control a traffic light in a simulated intersection to optimize traffic flow, queue length, and light switching behavior.

## 📂 Repository Structure

- `experiments/`  
  ├── `train_qtable.py`, `train_dqn.py`, `train_ppo.py`, `train_ppo_custom.py`  
  └── `evaluate_qtable.py`, `evaluate_dqn.py`, `evaluate_ppo.py`, `evaluate_ppo_custom.py`  
  → Scripts for training and evaluating models.

- `results/`  
  → Contains plots, saved models, and tables generated during training and evaluation.

- `src/`  
  └── `traffic_light_env.py`  
  → Custom Gymnasium environment with structured stochastic behavior (time-of-day variation, directional biases, weekday/weekend scenarios).

- `venv/`  
  → Python virtual environment (not tracked by Git if `.gitignore` is set correctly).

- `.gitignore`  
  → Ignores cache files, virtual environments, and logs.

- `README.md`  
  → Project overview and setup instructions.

## 📈 Key Insight

We initially experimented with a fully deterministic environment but found it encouraged reward exploitation and unrealistic behavior. We reverted to a structured yet stochastic environment, where agents learn under time-varying and directionally biased traffic, achieving more meaningful, robust, and interpretable outcomes.

## ⚙️ How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/traffic_light_rl.git
   cd traffic_light_rl

2. Install requirements:
    python -m venv venv
    source venv/bin/activate       # On Windows: venv\Scripts\activate
    pip install -r requirements.txt

3. Train a model:
    python train_dqn.py

4. Evaluate a model:
    python evaluate_dqn.py

## 🧪 Requirements

All Python dependencies are listed in requirements.txt. Run the following to generate or update:

    pip freeze > requirements.txt

Make sure you are inside your virtual environmet before running the command

## 📚 Report

All experiments, results, and insights are detailed in the final report:
📄 Traffic Light Optimization with Reinforcement Learning.pdf

## 🧠 Author

Juan Bautista Beyhaut
Bachelors in Data & Business Analytics - IE University
