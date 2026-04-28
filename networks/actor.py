from typing import Iterable, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def _build_mlp(input_dim: int, hidden_dims: Iterable[int], output_dim: int) -> nn.Sequential:
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(nn.ReLU())
        last_dim = int(hidden_dim)
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


def _bound_tensor(value: float, action_dim: int) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).flatten()
    if tensor.numel() == 1:
        tensor = tensor.repeat(action_dim)
    if tensor.numel() != action_dim:
        raise ValueError("Action bound size must be 1 or action_dim.")
    return tensor.view(1, action_dim)


class Actor(nn.Module):
    """Tanh-squashed Gaussian policy with action rescaling."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: float,
        action_high: float,
        hidden_dims: Tuple[int, ...] = (256, 256),
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.epsilon = float(epsilon)
        self.backbone = _build_mlp(obs_dim, hidden_dims, hidden_dims[-1])
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)

        low = _bound_tensor(action_low, action_dim)
        high = _bound_tensor(action_high, action_dim)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        raw_action = normal.rsample()
        squashed_action = torch.tanh(raw_action)
        action = squashed_action * self.action_scale + self.action_bias

        log_prob = normal.log_prob(raw_action)
        log_prob -= torch.log(self.action_scale * (1.0 - squashed_action.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        deterministic_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, deterministic_action
