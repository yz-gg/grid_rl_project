from typing import Dict, Optional

import numpy as np
import torch


class RolloutBuffer:
    """On-policy trajectory buffer with GAE advantage computation."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        capacity: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
        seed: Optional[int] = None,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.capacity = int(capacity)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.device = device
        self.rng = np.random.default_rng(seed)

        self.obs_buf = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.action_buf = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.reward_buf = np.zeros(self.capacity, dtype=np.float32)
        self.done_buf = np.zeros(self.capacity, dtype=np.float32)
        self.log_prob_buf = np.zeros((self.capacity, 1), dtype=np.float32)
        self.value_buf = np.zeros((self.capacity, 1), dtype=np.float32)
        self.return_buf = np.zeros((self.capacity, 1), dtype=np.float32)
        self.advantage_buf = np.zeros((self.capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.full = False

    def add(self, obs, action, reward: float, done: bool, log_prob: float, value: float) -> None:
        if self.ptr >= self.capacity:
            raise ValueError("RolloutBuffer is full. Call reset() before adding more samples.")

        self.obs_buf[self.ptr] = np.asarray(obs, dtype=np.float32).reshape(self.obs_dim)
        self.action_buf[self.ptr] = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        self.reward_buf[self.ptr] = float(reward)
        self.done_buf[self.ptr] = float(done)
        self.log_prob_buf[self.ptr] = float(log_prob)
        self.value_buf[self.ptr] = float(value)

        self.ptr += 1
        self.full = self.ptr == self.capacity

    def compute_returns_and_advantages(self, last_value: float, last_done: bool) -> None:
        if self.ptr == 0:
            raise ValueError("Cannot compute advantages for an empty rollout.")

        last_gae = 0.0
        next_value = float(last_value)
        next_non_terminal = 1.0 - float(last_done)

        for step in reversed(range(self.ptr)):
            if step == self.ptr - 1:
                bootstrap_value = next_value
                non_terminal = next_non_terminal
            else:
                bootstrap_value = float(self.value_buf[step + 1, 0])
                non_terminal = 1.0 - float(self.done_buf[step])

            delta = (
                float(self.reward_buf[step])
                + self.gamma * bootstrap_value * non_terminal
                - float(self.value_buf[step, 0])
            )
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            self.advantage_buf[step, 0] = last_gae

        self.return_buf[: self.ptr] = self.advantage_buf[: self.ptr] + self.value_buf[: self.ptr]

        advantages = self.advantage_buf[: self.ptr]
        advantage_mean = advantages.mean()
        advantage_std = advantages.std()
        self.advantage_buf[: self.ptr] = (advantages - advantage_mean) / (advantage_std + 1e-8)

    def get_batches(self, batch_size: int, device: Optional[str] = None):
        if self.ptr == 0:
            raise ValueError("Cannot sample batches from an empty rollout.")

        target_device = device or self.device
        indices = self.rng.permutation(self.ptr)
        for start in range(0, self.ptr, batch_size):
            batch_indices = indices[start : start + batch_size]
            yield self._to_tensor_batch(batch_indices, target_device)

    def reset(self) -> None:
        self.ptr = 0
        self.full = False

    def _to_tensor_batch(self, indices, device: str) -> Dict[str, torch.Tensor]:
        return {
            "obs": torch.as_tensor(self.obs_buf[indices], dtype=torch.float32, device=device),
            "actions": torch.as_tensor(self.action_buf[indices], dtype=torch.float32, device=device),
            "old_log_probs": torch.as_tensor(self.log_prob_buf[indices], dtype=torch.float32, device=device),
            "returns": torch.as_tensor(self.return_buf[indices], dtype=torch.float32, device=device),
            "advantages": torch.as_tensor(self.advantage_buf[indices], dtype=torch.float32, device=device),
        }

    def __len__(self) -> int:
        return self.ptr
