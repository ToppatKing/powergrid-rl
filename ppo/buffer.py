"""Rollout buffer with Generalised Advantage Estimation (GAE-λ).

The buffer stores a fixed-length rollout of transitions, then computes:

    advantages_t = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}

where δ_t = r_t + γ V(s_{t+1}) - V(s_t)  is the TD residual.

GAE interpolates between:
    λ=0  →  TD(0) — low variance, high bias
    λ=1  →  Monte Carlo returns — zero bias, high variance

The buffer also computes normalised advantages (zero mean, unit variance)
per mini-batch, which is critical for PPO training stability.
"""

from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    """Fixed-size rollout buffer for on-policy PPO.

    Args:
        buffer_size: Number of environment steps per rollout.
        obs_dim:     Dimensionality of observations.
        action_dim:  Dimensionality of actions.
        gamma:       Discount factor γ.
        gae_lambda:  GAE λ parameter.
        device:      Torch device for returned tensors.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device | str = "cpu",
    ) -> None:
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = torch.device(device)

        # Pre-allocate numpy arrays for efficiency
        self.obs          = np.zeros((buffer_size, obs_dim),     dtype=np.float32)
        self.actions      = np.zeros((buffer_size, action_dim),  dtype=np.float32)
        self.log_probs    = np.zeros(buffer_size,                dtype=np.float32)
        self.rewards      = np.zeros(buffer_size,                dtype=np.float32)
        self.values       = np.zeros(buffer_size,                dtype=np.float32)
        self.dones        = np.zeros(buffer_size,                dtype=np.float32)
        self.advantages   = np.zeros(buffer_size,                dtype=np.float32)
        self.returns      = np.zeros(buffer_size,                dtype=np.float32)

        self._ptr = 0
        self._full = False

    # ── Writing ───────────────────────────────────────────────────────────────

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        """Store one transition.

        Args:
            obs:      Current observation.
            action:   Action taken.
            log_prob: Log-probability of the action under the policy.
            reward:   Scalar reward received.
            value:    Critic's value estimate V(s).
            done:     Whether the episode ended after this step.
        """
        assert self._ptr < self.buffer_size, "Buffer is full; call compute_returns first."

        self.obs[self._ptr]       = obs
        self.actions[self._ptr]   = action
        self.log_probs[self._ptr] = log_prob
        self.rewards[self._ptr]   = reward
        self.values[self._ptr]    = value
        self.dones[self._ptr]     = float(done)
        self._ptr += 1

        if self._ptr == self.buffer_size:
            self._full = True

    # ── GAE computation ───────────────────────────────────────────────────────

    def compute_returns_and_advantages(self, last_value: float, last_done: bool) -> None:
        """Compute GAE advantages and discounted returns in-place.

        Must be called once after filling the buffer (before sampling).

        Args:
            last_value: Critic's value estimate for the state *after* the
                        last stored transition (bootstrap value).
            last_done:  Whether the last step ended an episode.
        """
        last_gae = 0.0
        next_value = last_value
        next_done  = float(last_done)

        # Iterate backwards through the buffer
        for t in reversed(range(self._ptr)):
            # δ_t = r_t + γ V(s_{t+1}) (1 - done_t) - V(s_t)
            delta = (
                self.rewards[t]
                + self.gamma * next_value * (1.0 - next_done)
                - self.values[t]
            )
            # A_t = δ_t + γλ A_{t+1} (1 - done_t)
            last_gae = delta + self.gamma * self.gae_lambda * (1.0 - next_done) * last_gae
            self.advantages[t] = last_gae

            next_value = self.values[t]
            next_done  = self.dones[t]

        # Returns = advantages + values (used as critic regression target)
        self.returns[: self._ptr] = self.advantages[: self._ptr] + self.values[: self._ptr]

    # ── Sampling ──────────────────────────────────────────────────────────────

    def get_mini_batches(
        self, num_mini_batches: int
    ) -> list[dict[str, torch.Tensor]]:
        """Shuffle and split the buffer into mini-batches.

        Advantages are normalised (zero mean, unit variance) per call
        for training stability.

        Args:
            num_mini_batches: Number of mini-batches to split into.

        Returns:
            List of dicts, each with keys:
            ``obs``, ``actions``, ``old_log_probs``, ``advantages``, ``returns``.
        """
        assert self._full or self._ptr == self.buffer_size, (
            "Buffer not full. Call after a complete rollout."
        )
        n = self._ptr
        indices = np.random.permutation(n)
        batch_size = n // num_mini_batches

        # Normalise advantages over the entire rollout
        adv = self.advantages[:n]
        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)

        mini_batches = []
        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            mini_batches.append(
                {
                    "obs":           torch.tensor(self.obs[idx],        device=self.device),
                    "actions":       torch.tensor(self.actions[idx],    device=self.device),
                    "old_log_probs": torch.tensor(self.log_probs[idx],  device=self.device),
                    "old_values":    torch.tensor(self.values[idx],     device=self.device),
                    "advantages":    torch.tensor(adv_norm[idx],        device=self.device),
                    "returns":       torch.tensor(self.returns[idx],    device=self.device),
                }
            )
        return mini_batches

    # ── Bookkeeping ───────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear the buffer for the next rollout collection."""
        self._ptr = 0
        self._full = False

    @property
    def is_full(self) -> bool:
        return self._full

    @property
    def size(self) -> int:
        return self._ptr
