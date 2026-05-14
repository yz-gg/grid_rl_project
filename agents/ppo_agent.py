from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
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


class PPOActorCritic(nn.Module):
    """Actor-critic network for continuous-action PPO."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: float,
        action_high: float,
        hidden_dims: Tuple[int, ...] = (256, 256),
        initial_log_std: float = -0.5,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.epsilon = float(epsilon)

        self.actor_backbone = _build_mlp(obs_dim, hidden_dims, hidden_dims[-1])
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(initial_log_std)))

        self.critic = _build_mlp(obs_dim, hidden_dims, 1)

        low = _bound_tensor(action_low, action_dim)
        high = _bound_tensor(action_high, action_dim)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.actor_backbone(obs)
        mean = self.mean_layer(features)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX).expand_as(mean)
        value = self.critic(obs)
        return mean, log_std, value

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self.forward(obs)
        normal = Normal(mean, log_std.exp())
        raw_action = normal.rsample()
        action, log_prob = self._squash_action_and_log_prob(normal, raw_action)
        deterministic_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, value, deterministic_action

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self.forward(obs)
        normal = Normal(mean, log_std.exp())
        raw_action = self._inverse_squash(actions)
        squashed_action = torch.tanh(raw_action)

        log_prob = normal.log_prob(raw_action)
        log_prob -= torch.log(self.action_scale * (1.0 - squashed_action.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy, value

    def _squash_action_and_log_prob(self, normal: Normal, raw_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        squashed_action = torch.tanh(raw_action)
        action = squashed_action * self.action_scale + self.action_bias

        log_prob = normal.log_prob(raw_action)
        log_prob -= torch.log(self.action_scale * (1.0 - squashed_action.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def _inverse_squash(self, actions: torch.Tensor) -> torch.Tensor:
        squashed = (actions - self.action_bias) / self.action_scale
        squashed = torch.clamp(squashed, -1.0 + self.epsilon, 1.0 - self.epsilon)
        return 0.5 * (torch.log1p(squashed) - torch.log1p(-squashed))


class PPOAgent:
    """Single-agent PPO implementation for continuous control."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: float,
        action_high: float,
        device: str = "cpu",
        hidden_dims: Tuple[int, ...] = (256, 256),
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: Optional[float] = 0.5,
        initial_log_std: float = -0.5,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.device = torch.device(device)
        self.clip_range = float(clip_range)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = max_grad_norm

        hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.policy = PPOActorCritic(
            obs_dim,
            action_dim,
            action_low,
            action_high,
            hidden_dims,
            initial_log_std=initial_log_std,
        ).to(self.device)

        actor_params = (
            list(self.policy.actor_backbone.parameters())
            + list(self.policy.mean_layer.parameters())
            + [self.policy.log_std]
        )
        self.optimizer = optim.Adam(
            [
                {"params": actor_params, "lr": actor_lr},
                {"params": self.policy.critic.parameters(), "lr": critic_lr},
            ]
        )

    def select_action(self, obs, evaluate: bool = False):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, self.obs_dim)
        with torch.no_grad():
            action, log_prob, value, deterministic_action = self.policy.sample(obs_tensor)
            selected = deterministic_action if evaluate else action

        return (
            selected.cpu().numpy().reshape(self.action_dim).astype(np.float32),
            float(log_prob.cpu().item()),
            float(value.cpu().item()),
        )

    def get_value(self, obs) -> float:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, self.obs_dim)
        with torch.no_grad():
            _, _, value = self.policy.forward(obs_tensor)
        return float(value.cpu().item())

    def update(self, rollout_buffer, batch_size: int, update_epochs: int) -> Dict[str, float]:
        metrics = []
        for _ in range(update_epochs):
            for batch in rollout_buffer.get_batches(batch_size, self.device):
                new_log_probs, entropy, values = self.policy.evaluate_actions(batch["obs"], batch["actions"])
                log_ratio = new_log_probs - batch["old_log_probs"]
                ratio = log_ratio.exp()

                unclipped_policy_loss = ratio * batch["advantages"]
                clipped_policy_loss = (
                    torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                    * batch["advantages"]
                )
                policy_loss = -torch.min(unclipped_policy_loss, clipped_policy_loss).mean()
                value_loss = F.mse_loss(values, batch["returns"])
                entropy_loss = entropy.mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), float(self.max_grad_norm))
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean()

                metrics.append(
                    {
                        "policy_loss": float(policy_loss.detach().cpu().item()),
                        "value_loss": float(value_loss.detach().cpu().item()),
                        "entropy": float(entropy_loss.detach().cpu().item()),
                        "approx_kl": float(approx_kl.detach().cpu().item()),
                        "clip_fraction": float(clip_fraction.detach().cpu().item()),
                    }
                )

        return {
            key: float(np.mean([metric[key] for metric in metrics]))
            for key in metrics[0]
        }

    def save_model(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "action_low": self.action_low,
            "action_high": self.action_high,
            "clip_range": self.clip_range,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
        }
        torch.save(checkpoint, path)

    def load_model(self, path, load_optimizers: bool = True) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        if load_optimizers and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
