import numpy as np
import gymnasium as gym
from gymnasium import spaces

from models.bergman_minimal import GlucoseInsulinModel
from utils.meals import MealScenario


class GlucoseEnv(gym.Env):
    """
    Minimal Gymnasium environment for glycemic control.

    Observation:
        [G, X, time_of_day_norm]

    Discrete Action:
        0 -> no insulin
        1 -> little amount
        2 -> average amount
        3 -> high amount
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps: int = 1440):
        super().__init__()

        self.model = GlucoseInsulinModel()
        self.meals = MealScenario.default()

        self.max_steps = max_steps
        self.dt = 1.0  # 1 minuto

        self.action_map = {
            0: 0.0,
            1: 0.5,
            2: 1.0,
            3: 2.0,
        }

        self.action_space = spaces.Discrete(len(self.action_map))

        low = np.array([20.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([600.0, 10.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.G = None
        self.X = None
        self.t = None

        self.history = []

    def _get_obs(self) -> np.ndarray:
        time_norm = (self.t % 1440) / 1440.0
        return np.array([self.G, self.X, time_norm], dtype=np.float32)

    def _compute_reward(self) -> float:
        """
        Reward shaping:
        - massimo vicino a 110 mg/dL
        - forti penalità in ipo e iper
        """
        target = 110.0
        reward = -abs(self.G - target) / 50.0

        if 70.0 <= self.G <= 180.0:
            reward += 1.0
        if self.G < 70.0:
            reward -= 2.0 + (70.0 - self.G) / 20.0
        if self.G > 180.0:
            reward -= 1.5 + (self.G - 180.0) / 50.0

        return float(reward)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.G = float(self.np_random.uniform(100.0, 180.0))
        self.X = 0.0
        self.t = 0
        self.history = []

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: int):
        if action not in self.action_map:
            raise ValueError(f"Azione non valida: {action}")

        u = self.action_map[action]
        D = self.meals.disturbance(self.t)

        self.G, self.X = self.model.step(
            G=self.G,
            X=self.X,
            u=u,
            D=D,
            dt=self.dt,
        )

        self.t += 1

        reward = self._compute_reward()

        terminated = bool(self.G < 40.0 or self.G > 400.0)
        truncated = bool(self.t >= self.max_steps)

        obs = self._get_obs()

        self.history.append(
            {
                "t": self.t,
                "G": self.G,
                "X": self.X,
                "u": u,
                "meal": D,
                "reward": reward,
            }
        )

        info = {
            "glucose": self.G,
            "insulin_effect": self.X,
            "meal_disturbance": D,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        print(f"t={self.t:4d} min | G={self.G:7.2f} mg/dL | X={self.X:6.3f}")

    def close(self):
        pass
