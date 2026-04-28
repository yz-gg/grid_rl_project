from typing import Dict, Optional

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size replay buffer backed by preallocated numpy arrays."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        capacity: int,
        device: str = "cpu",
        seed: Optional[int] = None,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.capacity = int(capacity)
        self.device = device
        self.rng = np.random.default_rng(seed)

        self.obs_buf = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.action_buf = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs_buf = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.done_buf = np.zeros((self.capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward: float, next_obs, done: bool) -> None:
        self.obs_buf[self.ptr] = np.asarray(obs, dtype=np.float32).reshape(self.obs_dim)
        self.action_buf[self.ptr] = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        self.reward_buf[self.ptr] = float(reward)
        self.next_obs_buf[self.ptr] = np.asarray(next_obs, dtype=np.float32).reshape(self.obs_dim)
        self.done_buf[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: Optional[str] = None) -> Dict[str, torch.Tensor]:
        if self.size < batch_size:
            raise ValueError(f"Cannot sample batch_size={batch_size} from buffer size={self.size}.")

        target_device = device or self.device
        indices = self.rng.integers(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs_buf[indices], dtype=torch.float32, device=target_device),
            "actions": torch.as_tensor(self.action_buf[indices], dtype=torch.float32, device=target_device),
            "rewards": torch.as_tensor(self.reward_buf[indices], dtype=torch.float32, device=target_device),
            "next_obs": torch.as_tensor(self.next_obs_buf[indices], dtype=torch.float32, device=target_device),
            "dones": torch.as_tensor(self.done_buf[indices], dtype=torch.float32, device=target_device),
        }

    def __len__(self) -> int:
        return self.size
