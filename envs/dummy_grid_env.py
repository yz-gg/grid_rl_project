from typing import Any, Dict, Optional, Tuple

import numpy as np


class DummyGridEnv:
    """A small Gym-like environment for validating SAC control flow.

    Observation: [df, V, P, Q, I]
    Action: delta Vref in [action_low, action_high]
    """

    obs_dim = 5
    action_dim = 1

    def __init__(
        self,
        max_steps: int = 200,
        disturbance_step: int = 50,
        disturbance_magnitude: float = 0.35,
        action_low: float = -0.02,
        action_high: float = 0.02,
        reward_coefficients: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.max_steps = int(max_steps)
        self.disturbance_step = int(disturbance_step)
        self.disturbance_magnitude = float(disturbance_magnitude)
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.rng = np.random.default_rng(seed)

        reward_coefficients = reward_coefficients or {}
        self.kf = float(reward_coefficients.get("kf", 20.0))
        self.kv = float(reward_coefficients.get("kv", 10.0))
        self.ka = float(reward_coefficients.get("ka", 0.5))
        self.ks = float(reward_coefficients.get("ks", 0.2))

        self.step_count = 0
        self.prev_action = 0.0
        self.df = 0.0
        self.V = 1.0
        self.P = 0.55
        self.Q = 0.18
        self.I = 0.58
        self.load_disturbance = 0.0

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.prev_action = 0.0
        self.load_disturbance = 0.0

        self.df = float(self.rng.normal(0.0, 0.005))
        self.V = float(1.0 + self.rng.normal(0.0, 0.003))
        self.P = float(0.55 + self.rng.normal(0.0, 0.005))
        self.Q = float(0.18 + self.rng.normal(0.0, 0.003))
        self.I = float(np.sqrt(self.P**2 + self.Q**2) / max(self.V, 0.2))
        return self._get_obs()

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        action_value = self._clip_action(action)
        previous_action = self.prev_action

        if self.step_count >= self.disturbance_step:
            self.load_disturbance = self.disturbance_magnitude
        else:
            self.load_disturbance = 0.0

        load = self.load_disturbance

        # Simplified coupled dynamics: load worsens df/V, positive delta Vref
        # partially compensates frequency deviation but shifts voltage upward.
        df_noise = float(self.rng.normal(0.0, 0.0015))
        v_noise = float(self.rng.normal(0.0, 0.0010))
        self.df = float(0.92 * self.df - 0.035 * load + 0.55 * action_value + df_noise)

        target_voltage = 1.0 + action_value - 0.035 * load - 0.12 * abs(self.df)
        self.V = float(0.88 * self.V + 0.12 * target_voltage + v_noise)

        self.P = float(0.55 + 0.65 * load + 1.8 * abs(self.df) + self.rng.normal(0.0, 0.004))
        self.Q = float(
            0.18
            + 0.40 * load
            + 2.5 * max(0.0, 1.0 - self.V)
            + 0.4 * abs(action_value)
            + self.rng.normal(0.0, 0.003)
        )
        self.I = float(np.sqrt(self.P**2 + self.Q**2) / max(self.V, 0.2))

        reward = -(
            self.kf * self.df**2
            + self.kv * (self.V - 1.0) ** 2
            + self.ka * action_value**2
            + self.ks * (action_value - previous_action) ** 2
        )

        self.prev_action = action_value
        self.step_count += 1

        done, reason = self._is_done()
        info = {
            "step": self.step_count,
            "load_disturbance": load,
            "clipped_action": action_value,
            "done_reason": reason,
        }
        return self._get_obs(), float(reward), done, info

    def sample_action(self) -> np.ndarray:
        return self.rng.uniform(self.action_low, self.action_high, size=(self.action_dim,)).astype(np.float32)

    def _clip_action(self, action: Any) -> float:
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.size == 0:
            raise ValueError("Action must contain one scalar value.")
        return float(np.clip(action_array[0], self.action_low, self.action_high))

    def _get_obs(self) -> np.ndarray:
        return np.array([self.df, self.V, self.P, self.Q, self.I], dtype=np.float32)

    def _is_done(self) -> Tuple[bool, str]:
        if self.step_count >= self.max_steps:
            return True, "max_steps"
        if abs(self.df) > 1.0:
            return True, "frequency_limit"
        if self.V < 0.85:
            return True, "voltage_low"
        if self.V > 1.15:
            return True, "voltage_high"
        return False, ""
