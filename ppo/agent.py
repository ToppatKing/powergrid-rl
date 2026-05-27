"""Proximal Policy Optimisation (PPO-Clip) — implemented from scratch.

The PPO update minimises:

    L(θ) = E_t[ −min( r_t(θ) Â_t,  clip(r_t(θ), 1−ε, 1+ε) Â_t ) ]
           + c_v · (V_θ(s_t) − R_t)²
           − c_e · H[π_θ(·|s_t)]

where:
    r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)   (importance weight)
    Â_t     = GAE advantage estimate
    c_v     = value loss coefficient
    c_e     = entropy bonus coefficient

References:
    Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
    https://arxiv.org/abs/1707.06347
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ppo.buffer import RolloutBuffer
from ppo.network import ActorCritic


@dataclass
class PPOConfig:
    """All PPO hyperparameters in one place."""

    # ── Rollout ───────────────────────────────────────────────────────────────
    rollout_steps: int   = 2048    # steps collected per update
    num_envs:      int   = 1       # parallel environments (1 = single env)

    # ── Optimisation ─────────────────────────────────────────────────────────
    num_epochs:        int   = 4       # PPO epochs over each rollout
    num_mini_batches:  int   = 8       # mini-batches per epoch
    learning_rate:     float = 3e-4
    anneal_lr:         bool  = True    # linearly decay lr to 0

    # ── PPO objectives ────────────────────────────────────────────────────────
    gamma:        float = 0.99     # discount factor
    gae_lambda:   float = 0.95     # GAE λ
    clip_coef:    float = 0.2      # ε in clipped surrogate
    value_coef:   float = 0.5      # critic loss weight
    entropy_coef: float = 0.01     # entropy bonus weight
    max_grad_norm:float = 0.5      # gradient clipping

    # ── Clipping ──────────────────────────────────────────────────────────────
    clip_value_loss: bool  = True   # also clip value function updates

    # ── Network ───────────────────────────────────────────────────────────────
    hidden_dim: int = 256


@dataclass
class PPOUpdateStats:
    """Statistics from one PPO update cycle."""
    policy_loss:   float
    value_loss:    float
    entropy:       float
    approx_kl:     float
    clip_fraction: float
    explained_var: float


class PPOAgent:
    """PPO agent that wraps the Actor-Critic network and implements the
    clipped surrogate update.

    Args:
        obs_dim:    Observation space dimensionality.
        action_dim: Action space dimensionality.
        config:     :class:`PPOConfig` hyperparameters.
        device:     Torch device.
        total_steps:Total training steps (needed for lr annealing).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: PPOConfig | None = None,
        device: torch.device | str = "cpu",
        total_steps: int = 1_000_000,
    ) -> None:
        self.config = config or PPOConfig()
        self.device = torch.device(device)
        self.total_steps = total_steps
        self._update_count = 0

        # ── Network ───────────────────────────────────────────────────────────
        self.network = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=self.config.hidden_dim,
        ).to(self.device)

        # ── Optimizer ────────────────────────────────────────────────────────
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5,
        )

        # ── Rollout buffer ────────────────────────────────────────────────────
        self.buffer = RolloutBuffer(
            buffer_size=self.config.rollout_steps,
            obs_dim=obs_dim,
            action_dim=action_dim,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            device=self.device,
        )

        # Total number of gradient updates that will happen
        self._total_updates = (
            total_steps
            // (self.config.rollout_steps * self.config.num_envs)
        )

    # ── Action selection (during rollout collection) ──────────────────────────

    @torch.no_grad()
    def get_action(
        self, obs: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        """Sample an action from the policy.

        Args:
            obs: Current observation array ``(obs_dim,)``.

        Returns:
            Tuple of (action, log_prob, value).
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t, log_prob_t, _, value_t = self.network.get_action_and_value(obs_t)

        action = action_t.squeeze(0).cpu().numpy()
        # Clamp to valid action range
        action = np.clip(action, -1.0, 1.0)
        log_prob = float(log_prob_t.item())
        value = float(value_t.item())
        return action, log_prob, value

    @torch.no_grad()
    def get_deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        """Return mean action (no stochasticity) for evaluation."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.network.get_deterministic_action(obs_t)
        return np.clip(action.squeeze(0).cpu().numpy(), -1.0, 1.0)

    @torch.no_grad()
    def get_value(self, obs: np.ndarray) -> float:
        """Return the critic's value estimate for a given observation."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.network.get_value(obs_t).item())

    # ── PPO update ────────────────────────────────────────────────────────────

    def update(self, last_obs: np.ndarray, last_done: bool) -> PPOUpdateStats:
        """Run GAE then perform PPO update epochs on the collected rollout.

        Args:
            last_obs:  Observation after the last environment step (for
                       bootstrapping the value).
            last_done: Whether the environment was done after the last step.

        Returns:
            :class:`PPOUpdateStats` with training diagnostics.
        """
        # ── Learning rate annealing ────────────────────────────────────────
        if self.config.anneal_lr:
            frac = 1.0 - self._update_count / max(1, self._total_updates)
            new_lr = self.config.learning_rate * frac
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr

        # ── Bootstrap value for the last observation ───────────────────────
        last_value = self.get_value(last_obs)

        # ── Compute GAE advantages and discounted returns ──────────────────
        self.buffer.compute_returns_and_advantages(
            last_value=last_value, last_done=last_done
        )

        # ── Accumulate stats over all epochs ──────────────────────────────
        policy_losses, value_losses, entropies = [], [], []
        approx_kls, clip_fracs = [], []

        cfg = self.config

        for _ in range(cfg.num_epochs):
            for mb in self.buffer.get_mini_batches(cfg.num_mini_batches):
                obs_b       = mb["obs"]
                actions_b   = mb["actions"]
                old_lp_b    = mb["old_log_probs"]
                advantages_b= mb["advantages"]
                returns_b   = mb["returns"]

                # Forward pass
                _, new_log_probs, entropy, new_values = (
                    self.network.get_action_and_value(obs_b, actions_b)
                )
                new_values = new_values.squeeze(-1)

                # ── Policy loss (clipped surrogate) ────────────────────────
                log_ratio = new_log_probs - old_lp_b
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()

                pg_loss_unclipped = -advantages_b * ratio
                pg_loss_clipped   = -advantages_b * torch.clamp(
                    ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef
                )
                policy_loss = torch.max(pg_loss_unclipped, pg_loss_clipped).mean()

                # ── Value loss ─────────────────────────────────────────────
                if cfg.clip_value_loss:
                    # Clip value update for stability.
                    # Use old_values from the mini-batch dict — these are
                    # correctly indexed by the shuffled batch indices,
                    # unlike a raw slice of self.buffer.values[:size].
                    old_values = mb["old_values"]
                    v_clipped = old_values + torch.clamp(
                        new_values - old_values,
                        -cfg.clip_coef,
                        cfg.clip_coef,
                    )
                    vf_loss1 = (new_values   - returns_b).pow(2)
                    vf_loss2 = (v_clipped    - returns_b).pow(2)
                    value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
                else:
                    value_loss = 0.5 * (new_values - returns_b).pow(2).mean()

                # ── Total loss ─────────────────────────────────────────────
                loss = (
                    policy_loss
                    + cfg.value_coef  * value_loss
                    - cfg.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), cfg.max_grad_norm
                )
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())
                approx_kls.append(approx_kl)
                clip_fracs.append(clip_frac)

        # ── Explained variance (critic quality metric) ─────────────────────
        y_true = self.buffer.returns[: self.buffer.size]
        y_pred = self.buffer.values[: self.buffer.size]
        var_y = np.var(y_true)
        explained_var = float(
            1.0 - np.var(y_true - y_pred) / (var_y + 1e-8)
        )

        self.buffer.reset()
        self._update_count += 1

        return PPOUpdateStats(
            policy_loss=float(np.mean(policy_losses)),
            value_loss=float(np.mean(value_losses)),
            entropy=float(np.mean(entropies)),
            approx_kl=float(np.mean(approx_kls)),
            clip_fraction=float(np.mean(clip_fracs)),
            explained_var=explained_var,
        )

    # ── Checkpointing ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save network weights and optimizer state."""
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "update_count": self._update_count,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load network weights and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._update_count = checkpoint.get("update_count", 0)
