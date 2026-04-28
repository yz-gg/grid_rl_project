from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from networks.actor import Actor
from networks.critic import Critic


class SACAgent:
    """Single-agent Soft Actor-Critic implementation."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: float,
        action_high: float,
        device: str = "cpu",
        hidden_dims: Tuple[int, ...] = (256, 256),
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        automatic_entropy_tuning: bool = True,
        target_entropy: Optional[float] = None,
        initial_alpha: float = 0.2,
        grad_clip_norm: Optional[float] = 10.0,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.automatic_entropy_tuning = bool(automatic_entropy_tuning)
        self.target_entropy = -float(action_dim) if target_entropy is None else float(target_entropy)
        self.grad_clip_norm = grad_clip_norm

        hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.actor = Actor(obs_dim, action_dim, action_low, action_high, hidden_dims).to(self.device)
        self.critic1 = Critic(obs_dim, action_dim, hidden_dims).to(self.device)
        self.critic2 = Critic(obs_dim, action_dim, hidden_dims).to(self.device)
        self.target_critic1 = Critic(obs_dim, action_dim, hidden_dims).to(self.device)
        self.target_critic2 = Critic(obs_dim, action_dim, hidden_dims).to(self.device)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr,
        )

        initial_log_alpha = float(np.log(initial_alpha))
        self.log_alpha = torch.tensor(initial_log_alpha, dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, obs, evaluate: bool = False) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, self.obs_dim)
        with torch.no_grad():
            action, _, deterministic_action = self.actor.sample(obs_tensor)
            selected = deterministic_action if evaluate else action
        return selected.cpu().numpy().reshape(self.action_dim).astype(np.float32)

    def update(self, replay_buffer, batch_size: int) -> Dict[str, float]:
        batch = replay_buffer.sample(batch_size, self.device)
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_obs)
            target_q1 = self.target_critic1(next_obs, next_actions)
            target_q2 = self.target_critic2(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_probs
            td_target = rewards + (1.0 - dones) * self.gamma * target_q

        current_q1 = self.critic1(obs, actions)
        current_q2 = self.critic2(obs, actions)
        critic_loss = F.mse_loss(current_q1, td_target) + F.mse_loss(current_q2, td_target)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._clip_gradients([self.critic1, self.critic2])
        self.critic_optimizer.step()

        self._set_critic_requires_grad(False)
        sampled_actions, log_probs, _ = self.actor.sample(obs)
        q1_pi = self.critic1(obs, sampled_actions)
        q2_pi = self.critic2(obs, sampled_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_probs - min_q_pi).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self._clip_gradients([self.actor])
        self.actor_optimizer.step()
        self._set_critic_requires_grad(True)

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.zeros(1, device=self.device)

        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)

        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "alpha_loss": float(alpha_loss.detach().cpu().item()),
            "alpha": float(self.alpha.detach().cpu().item()),
        }

    def save_model(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "action_low": self.action_low,
            "action_high": self.action_high,
            "gamma": self.gamma,
            "tau": self.tau,
            "target_entropy": self.target_entropy,
            "automatic_entropy_tuning": self.automatic_entropy_tuning,
        }
        torch.save(checkpoint, path)

    def load_model(self, path, load_optimizers: bool = True) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.target_critic1.load_state_dict(checkpoint.get("target_critic1", checkpoint["critic1"]))
        self.target_critic2.load_state_dict(checkpoint.get("target_critic2", checkpoint["critic2"]))

        with torch.no_grad():
            self.log_alpha.copy_(checkpoint.get("log_alpha", self.log_alpha.detach()).to(self.device))

        if load_optimizers:
            if "actor_optimizer" in checkpoint:
                self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            if "critic_optimizer" in checkpoint:
                self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            if "alpha_optimizer" in checkpoint:
                self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for source_param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * source_param.data)

    def _set_critic_requires_grad(self, requires_grad: bool) -> None:
        for critic in (self.critic1, self.critic2):
            for param in critic.parameters():
                param.requires_grad = requires_grad

    def _clip_gradients(self, modules) -> None:
        if self.grad_clip_norm is None:
            return
        parameters = []
        for module in modules:
            parameters.extend(list(module.parameters()))
        torch.nn.utils.clip_grad_norm_(parameters, float(self.grad_clip_norm))
