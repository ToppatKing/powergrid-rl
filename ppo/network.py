"""Actor-Critic neural networks for PPO.

Architecture:
    Shared trunk: Linear(obs_dim, 256) → Tanh → Linear(256, 256) → Tanh

    Actor head:   Linear(256, action_dim) → μ  (unbounded)
                  log_std: learnable parameter vector (independent of obs)

    Critic head:  Linear(256, 1)          → V(s)

Policy: Diagonal Gaussian  π(a|s) = N(μ(s), diag(exp(log_std)²))

The diagonal Gaussian lets the agent output smooth continuous setpoints
for each generator while learning per-dimension exploration variance.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    """Orthogonal weight initialisation — standard PPO practice."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    """Shared-trunk Actor-Critic network.

    Args:
        obs_dim:    Dimensionality of the observation vector.
        action_dim: Dimensionality of the action vector.
        hidden_dim: Width of the shared hidden layers.
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX =  0.5

    def __init__(
        self,
        obs_dim: int = 20,
        action_dim: int = 5,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        # ── Shared feature extractor ─────────────────────────────────────────
        self.trunk = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )

        # ── Actor head (policy mean) ─────────────────────────────────────────
        self.actor_mean = _layer_init(
            nn.Linear(hidden_dim, action_dim), std=0.01
        )
        # Learnable log-std (state-independent, one per action dimension)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # ── Critic head (value function) ─────────────────────────────────────
        self.critic = _layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    # ── Forward passes ────────────────────────────────────────────────────────

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute state value V(s).

        Args:
            obs: Tensor of shape ``(batch, obs_dim)`` or ``(obs_dim,)``.

        Returns:
            Value tensor of shape ``(batch, 1)``.
        """
        return self.critic(self.trunk(obs))

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action and compute log-probability + value.

        Args:
            obs:    Observation tensor ``(batch, obs_dim)``.
            action: If provided, compute log_prob of this action instead
                    of sampling. Used during the PPO update step.

        Returns:
            Tuple of:
            - action:   Sampled (or provided) action ``(batch, action_dim)``
            - log_prob: Sum of log-probs over action dims ``(batch,)``
            - entropy:  Entropy of the Gaussian distribution ``(batch,)``
            - value:    Critic value ``(batch, 1)``
        """
        features = self.trunk(obs)
        mean = self.actor_mean(features)

        # Clamp log_std to prevent numerical instability
        log_std = torch.clamp(self.actor_log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp().expand_as(mean)

        dist = Normal(mean, std)

        if action is None:
            action = dist.sample()

        # Sum log-probs across action dimensions (independent Gaussians)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(features)

        return action, log_prob, entropy, value

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the mean action (no sampling) for deterministic evaluation.

        Args:
            obs: Observation tensor ``(obs_dim,)`` or ``(1, obs_dim)``.

        Returns:
            Action tensor clamped to [-1, 1].
        """
        with torch.no_grad():
            features = self.trunk(obs)
            mean = self.actor_mean(features)
        return mean.clamp(-1.0, 1.0)
