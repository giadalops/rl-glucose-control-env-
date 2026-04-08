# rl-glucose-control-env

A small educational project designed to simulate a blood glucose control environment using the **Gymnasium** interface to train Reinforcement Learning (RL) agents.

## Objective
The agent's goal is to maintain blood glucose within a safe physiological range (Time In Range) by regulating discrete insulin doses while compensating for external disturbances caused by meals.

## Project Structure
- `env/`: **Gymnasium environment** implementation (observation/action spaces and rewards).
- `models/`: Simplified **glucose-insulin dynamic models** (the mathematical simulation).
- `utils/`: **Meal management** and scenario disturbance logic.
- `train.py`: Training script using the **DQN** (Deep Q-Network) algorithm from Stable Baselines3.
- `test.py`: **Evaluation script** to run the trained model, calculate metrics, and generate plots.

## Key Metrics
The agent is evaluated based on:
* **TIR (Time In Range):** Percentage of time glucose stays between 70 and 180 mg/dL.
* **Hypoglycemia Avoidance:** Minimizing time spent below 70 mg/dL.
* **Hyperglycemia Control:** Minimizing spikes above 180 mg/dL.

## Installation
```bash
pip install -r requirements.txt
