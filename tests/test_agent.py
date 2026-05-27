"""Unit tests for the PPO agent, network, and rollout buffer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ppo.agent import PPOAgent, PPOConfig
from ppo.buffer import RolloutBuffer
from ppo.network import ActorCritic


OBS_DIM    = 20
ACTION_DIM = 5


# ── ActorCritic network ───────────────────────────────────────────────────────

class TestActorCritic:
    @pytest.fixture
    def net(self):
        return ActorCritic(obs_dim=OBS_DIM, action_dim=ACTION_DIM)

    def test_output_shapes(self, net):
        obs = torch.zeros(4, OBS_DIM)
        action, log_prob, entropy, value = net.get_action_and_value(obs)
        assert action.shape   == (4, ACTION_DIM)
        assert log_prob.shape == (4,)
        assert entropy.shape  == (4,)
        assert value.shape    == (4, 1)

    def test_value_shape(self, net):
        obs = torch.zeros(8, OBS_DIM)
        v = net.get_value(obs)
        assert v.shape == (8, 1)

    def test_deterministic_action_clamped(self, net):
        obs = torch.randn(1, OBS_DIM) * 100   # extreme obs
        action = net.get_deterministic_action(obs)
        assert torch.all(action >= -1.0)
        assert torch.all(action <=  1.0)

    def test_action_given_returned_unchanged(self, net):
        obs = torch.zeros(3, OBS_DIM)
        given_action = torch.ones(3, ACTION_DIM) * 0.5
        returned_action, _, _, _ = net.get_action_and_value(obs, given_action)
        torch.testing.assert_close(returned_action, given_action)

    def test_log_prob_finite(self, net):
        obs = torch.randn(16, OBS_DIM)
        _, log_prob, _, _ = net.get_action_and_value(obs)
        assert torch.all(torch.isfinite(log_prob))

    def test_entropy_positive(self, net):
        obs = torch.randn(8, OBS_DIM)
        _, _, entropy, _ = net.get_action_and_value(obs)
        assert torch.all(entropy > 0)

    def test_gradient_flows(self, net):
        obs = torch.randn(4, OBS_DIM)
        action, log_prob, entropy, value = net.get_action_and_value(obs)
        loss = -log_prob.mean() + value.mean() - entropy.mean()
        loss.backward()
        for name, param in net.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_single_obs_works(self, net):
        obs = torch.zeros(OBS_DIM)
        action = net.get_deterministic_action(obs.unsqueeze(0))
        assert action.shape == (1, ACTION_DIM)


# ── RolloutBuffer ─────────────────────────────────────────────────────────────

class TestRolloutBuffer:
    @pytest.fixture
    def buf(self):
        return RolloutBuffer(
            buffer_size=64,
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            gamma=0.99,
            gae_lambda=0.95,
        )

    def _fill(self, buf, n=64):
        rng = np.random.default_rng(0)
        for i in range(n):
            buf.add(
                obs=rng.standard_normal(OBS_DIM).astype(np.float32),
                action=rng.uniform(-1, 1, ACTION_DIM).astype(np.float32),
                log_prob=float(rng.standard_normal()),
                reward=float(rng.standard_normal()),
                value=float(rng.standard_normal()),
                done=(i == n - 1),
            )

    def test_add_increments_ptr(self, buf):
        buf.add(
            np.zeros(OBS_DIM, np.float32),
            np.zeros(ACTION_DIM, np.float32),
            0.0, 1.0, 0.5, False,
        )
        assert buf.size == 1

    def test_full_after_buffer_size_adds(self, buf):
        self._fill(buf, 64)
        assert buf.is_full

    def test_gae_advantages_shape(self, buf):
        self._fill(buf, 64)
        buf.compute_returns_and_advantages(last_value=0.0, last_done=True)
        assert buf.advantages.shape == (64,)

    def test_returns_finite(self, buf):
        self._fill(buf, 64)
        buf.compute_returns_and_advantages(0.0, True)
        assert np.all(np.isfinite(buf.returns))

    def test_advantages_finite(self, buf):
        self._fill(buf, 64)
        buf.compute_returns_and_advantages(0.0, True)
        assert np.all(np.isfinite(buf.advantages))

    def test_mini_batch_count(self, buf):
        self._fill(buf, 64)
        buf.compute_returns_and_advantages(0.0, True)
        mbs = buf.get_mini_batches(num_mini_batches=4)
        assert len(mbs) == 4

    def test_mini_batch_tensors(self, buf):
        self._fill(buf, 64)
        buf.compute_returns_and_advantages(0.0, True)
        mb = buf.get_mini_batches(8)[0]
        for key in ("obs", "actions", "old_log_probs", "old_values", "advantages", "returns"):
            assert key in mb
            assert isinstance(mb[key], torch.Tensor)

    def test_advantages_normalised_approx(self, buf):
        self._fill(buf, 64)
        buf.compute_returns_and_advantages(0.0, True)
        mbs = buf.get_mini_batches(1)
        adv = mbs[0]["advantages"].numpy()
        # Normalised advantages should have approximately zero mean and unit std
        assert abs(adv.mean()) < 0.1
        assert abs(adv.std() - 1.0) < 0.2

    def test_reset_clears_ptr(self, buf):
        self._fill(buf, 64)
        buf.reset()
        assert buf.size == 0
        assert not buf.is_full

    def test_gae_terminal_state_zero_bootstrap(self, buf):
        """Advantages at terminal state should not include future value."""
        self._fill(buf, 64)
        buf.compute_returns_and_advantages(last_value=0.0, last_done=True)
        # The advantage at the last step should not depend on last_value
        adv_done = buf.advantages[63]
        buf2 = RolloutBuffer(64, OBS_DIM, ACTION_DIM)
        rng = np.random.default_rng(0)
        for i in range(64):
            buf2.add(
                rng.standard_normal(OBS_DIM).astype(np.float32),
                rng.uniform(-1, 1, ACTION_DIM).astype(np.float32),
                float(rng.standard_normal()),
                float(rng.standard_normal()),
                float(rng.standard_normal()),
                (i == 63),
            )
        buf2.compute_returns_and_advantages(last_value=999.0, last_done=True)
        # last_value should not affect advantage when last_done=True
        assert abs(adv_done - buf2.advantages[63]) < 1e-4


# ── PPOAgent ──────────────────────────────────────────────────────────────────

class TestPPOAgent:
    @pytest.fixture
    def agent(self):
        cfg = PPOConfig(
            rollout_steps=64,
            num_epochs=2,
            num_mini_batches=4,
            learning_rate=1e-3,
            anneal_lr=False,
        )
        return PPOAgent(
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            config=cfg,
            total_steps=10_000,
        )

    def _fill_buffer(self, agent):
        rng = np.random.default_rng(1)
        for i in range(agent.config.rollout_steps):
            obs = rng.standard_normal(OBS_DIM).astype(np.float32)
            action, log_prob, value = agent.get_action(obs)
            agent.buffer.add(
                obs=obs,
                action=action,
                log_prob=log_prob,
                reward=float(rng.standard_normal()),
                value=value,
                done=(i == agent.config.rollout_steps - 1),
            )

    def test_get_action_shape(self, agent):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        action, log_prob, value = agent.get_action(obs)
        assert action.shape == (ACTION_DIM,)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_action_in_range(self, agent):
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action, _, _ = agent.get_action(obs)
        assert np.all(action >= -1.0)
        assert np.all(action <=  1.0)

    def test_deterministic_action_stable(self, agent):
        obs = np.ones(OBS_DIM, dtype=np.float32)
        a1 = agent.get_deterministic_action(obs)
        a2 = agent.get_deterministic_action(obs)
        np.testing.assert_array_equal(a1, a2)

    def test_update_returns_stats(self, agent):
        self._fill_buffer(agent)
        last_obs = np.zeros(OBS_DIM, dtype=np.float32)
        stats = agent.update(last_obs, last_done=True)
        assert hasattr(stats, "policy_loss")
        assert hasattr(stats, "value_loss")
        assert hasattr(stats, "entropy")
        assert hasattr(stats, "approx_kl")

    def test_update_stats_finite(self, agent):
        self._fill_buffer(agent)
        stats = agent.update(np.zeros(OBS_DIM, np.float32), True)
        for field in ("policy_loss", "value_loss", "entropy", "approx_kl"):
            val = getattr(stats, field)
            assert np.isfinite(val), f"{field} is not finite: {val}"

    def test_update_changes_weights(self, agent):
        self._fill_buffer(agent)
        params_before = [p.clone() for p in agent.network.parameters()]
        agent.update(np.zeros(OBS_DIM, np.float32), True)
        params_after = list(agent.network.parameters())
        changed = any(
            not torch.equal(b, a)
            for b, a in zip(params_before, params_after)
        )
        assert changed, "Weights did not change after update"

    def test_save_and_load(self, agent, tmp_path):
        path = str(tmp_path / "model.pt")
        agent.save(path)
        obs = np.ones(OBS_DIM, dtype=np.float32)
        action_before = agent.get_deterministic_action(obs).copy()

        # New agent with different random init
        new_agent = PPOAgent(OBS_DIM, ACTION_DIM, config=agent.config)
        new_agent.load(path)
        action_after = new_agent.get_deterministic_action(obs)
        np.testing.assert_array_almost_equal(action_before, action_after)

    def test_buffer_reset_after_update(self, agent):
        self._fill_buffer(agent)
        agent.update(np.zeros(OBS_DIM, np.float32), True)
        assert agent.buffer.size == 0
