import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

from env.glucose_env import GlucoseEnv


def compute_metrics(history: list[dict]) -> dict:
    """Calculates clinical performance metrics from the simulation history."""
    glucose = np.array([h["G"] for h in history], dtype=float)

    # Time In Range (TIR): percentage of time between 70 and 180 mg/dL
    tir = np.mean((glucose >= 70.0) & (glucose <= 180.0)) * 100.0
    hypo = np.sum(glucose < 70.0)
    hyper = np.sum(glucose > 180.0)

    return {
        "TIR_percent": tir,
        "hypoglycemia_minutes": int(hypo),
        "hyperglycemia_minutes": int(hyper),
        "mean_glucose": float(np.mean(glucose)),
        "min_glucose": float(np.min(glucose)),
        "max_glucose": float(np.max(glucose)),
    }


def run_episode(model, env):
    """Runs a single full-day simulation using the trained agent."""
    obs, _ = env.reset()
    done = False

    while not done:
        # Use deterministic=True for evaluation to get the best learned policy
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

    return env.history


def plot_results(history: list[dict]):
    """Generates visual plots for glucose levels, insulin actions, and meal impacts."""
    t = np.array([h["t"] for h in history], dtype=float)
    glucose = np.array([h["G"] for h in history], dtype=float)
    insulin = np.array([h["u"] for h in history], dtype=float)
    meal = np.array([h["meal"] for h in history], dtype=float)

    # --- Glucose Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(t, glucose, label="Glucose [mg/dL]", color="blue")
    plt.axhline(70.0, color="red", linestyle="--", label="Hypoglycemia Threshold")
    plt.axhline(180.0, color="orange", linestyle="--", label="Hyperglycemia Threshold")
    plt.axhline(110.0, color="green", linestyle=":", label="Target")
    plt.xlabel("Time [min]")
    plt.ylabel("Glucose [mg/dL]")
    plt.title("Daily Glucose Trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig("glucose_trajectory.png", dpi=150)
    plt.show()

    # --- Insulin Action Plot ---
    plt.figure(figsize=(10, 4))
    plt.step(t, insulin, where="post", label="Insulin Dosage", color="purple")
    plt.xlabel("Time [min]")
    plt.ylabel("Insulin units/min")
    plt.title("Agent Control Actions (Insulin)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("insulin_actions.png", dpi=150)
    plt.show()

    # --- Meal Disturbance Plot ---
    plt.figure(figsize=(10, 4))
    plt.plot(t, meal, label="Meal impact on glucose", color="brown")
    plt.xlabel("Time [min]")
    plt.ylabel("Glucose change rate")
    plt.title("External Meal Disturbances")
    plt.legend()
    plt.tight_layout()
    plt.savefig("meal_disturbance.png", dpi=150)
    plt.show()


def main():
    # Load environment and the pre-trained model
    env = GlucoseEnv(max_steps=1440)
    model = DQN.load("dqn_glucose_env")

    # Execute simulation
    history = run_episode(model, env)
    metrics = compute_metrics(history)

    # Output results
    print("\n=== Evaluation metrics ===")
    for k, v in metrics.items():
        # Formatting for readability
        label = k.replace("_", " ").capitalize()
        print(f"{label}: {v:.2f}" if isinstance(v, float) else f"{label}: {v}")

    plot_results(history)


if __name__ == "__main__":
    main()
