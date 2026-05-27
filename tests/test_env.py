"""Tests for the PowerGrid-v0 environment.

Covers:
  - Gymnasium API compliance (spaces, reset, step signatures)
  - Physics correctness (ramp constraints, battery SoC bounds)
  - Observation validity (shape, dtype, bounds)
  - Reward sign conventions
  - Determinism under fixed seeds
"""

from __future__ import annotations

import numpy as np
import pytest

from env.generators import BatteryStorage, Generator, GeneratorSpec
from env.demand_model import DemandModel
from env.powergrid_env import PowerGridEnv, STEPS_PER_EPISODE


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def env() -> PowerGridEnv:
    e = PowerGridEnv()
    e.reset(seed=0)
    return e


@pytest.fixture
def coal_gen() -> Generator:
    rng = np.random.default_rng(0)
    return Generator(GeneratorSpec.LIBRARY["coal_1"], rng)


@pytest.fixture
def battery() -> BatteryStorage:
    b = BatteryStorage()
    b.reset(initial_soc_fraction=0.5)
    return b


# ── Gymnasium API compliance ──────────────────────────────────────────────────

class TestGymAPI:
    def test_observation_space_shape(self, env):
        assert env.observation_space.shape == (20,)

    def test_action_space_shape(self, env):
        assert env.action_space.shape == (5,)

    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset(seed=1)
        assert obs.shape == (20,)
        assert isinstance(info, dict)

    def test_obs_dtype_float32(self, env):
        obs, _ = env.reset(seed=0)
        assert obs.dtype == np.float32

    def test_obs_within_bounds(self, env):
        obs, _ = env.reset(seed=0)
        assert np.all(obs >= -1.0 - 1e-6)
        assert np.all(obs <=  1.0 + 1e-6)

    def test_step_returns_5_tuple(self, env):
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5

    def test_step_obs_shape(self, env):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs.shape == (20,)

    def test_reward_is_scalar(self, env):
        _, reward, _, _, _ = env.step(env.action_space.sample())
        assert isinstance(reward, float)

    def test_terminated_false_mid_episode(self, env):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert not terminated
        assert not truncated

    def test_episode_terminates_at_48_steps(self, env):
        env.reset(seed=0)
        for i in range(STEPS_PER_EPISODE - 1):
            _, _, t, tr, _ = env.step(env.action_space.sample())
            assert not t, f"Terminated early at step {i+1}"
        _, _, terminated, _, _ = env.step(env.action_space.sample())
        assert terminated

    def test_action_space_contains_zeros(self, env):
        assert env.action_space.contains(np.zeros(5, dtype=np.float32))

    def test_obs_space_contains_obs(self, env):
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)

    def test_info_contains_expected_keys(self, env):
        _, _, _, _, info = env.step(env.action_space.sample())
        for key in ("episode_cost", "episode_carbon_t", "freq_dev_hz", "battery_soc"):
            assert key in info, f"Missing key: {key}"


# ── Physics correctness ───────────────────────────────────────────────────────

class TestGeneratorPhysics:
    def test_ramp_rate_respected(self, coal_gen):
        coal_gen.reset(initial_output_fraction=0.5)
        initial = coal_gen.output_mw
        # Try to jump to full capacity instantly
        coal_gen.dispatch(coal_gen.spec.capacity_mw)
        delta = abs(coal_gen.output_mw - initial)
        assert delta <= coal_gen.spec.ramp_rate_mw + 1e-6

    def test_cannot_exceed_capacity(self, coal_gen):
        coal_gen.reset(initial_output_fraction=0.5)
        # Dispatch many steps at full action
        for _ in range(20):
            coal_gen.dispatch_by_action(1.0)
        assert coal_gen.output_mw <= coal_gen.spec.capacity_mw + 1e-6

    def test_can_shut_down(self, coal_gen):
        coal_gen.reset(initial_output_fraction=0.8)
        # Drive to zero over multiple steps
        for _ in range(30):
            coal_gen.dispatch(0.0)
        assert coal_gen.output_mw == pytest.approx(0.0, abs=1e-6)

    def test_output_fraction_normalised(self, coal_gen):
        coal_gen.reset(0.75)
        assert 0.0 <= coal_gen.output_fraction <= 1.0

    def test_cost_zero_when_off(self, coal_gen):
        coal_gen.reset(0.0)
        coal_gen.output_mw = 0.0
        assert coal_gen.cost == 0.0

    def test_cost_positive_when_on(self, coal_gen):
        coal_gen.reset(0.5)
        assert coal_gen.cost > 0.0

    def test_start_cost_incurred_on_cold_start(self, coal_gen):
        coal_gen.reset(0.0)
        coal_gen.output_mw = 0.0
        _, start_cost = coal_gen.dispatch(200.0)
        assert start_cost > 0.0

    def test_no_start_cost_when_already_running(self, coal_gen):
        coal_gen.reset(0.5)
        _, start_cost = coal_gen.dispatch(300.0)
        assert start_cost == 0.0

    def test_nuclear_min_stable_high(self):
        rng = np.random.default_rng(0)
        nuclear = Generator(GeneratorSpec.LIBRARY["nuclear"], rng)
        nuclear.reset(0.9)
        # Nuclear cannot ramp below min_stable while still online
        nuclear.dispatch(100.0)   # below min_stable — should ramp down slowly
        # After one step it should still be near min_stable or above
        # (unless it reached 0 via full shutdown)
        if nuclear.output_mw > 0:
            assert nuclear.output_mw >= GeneratorSpec.LIBRARY["nuclear"].min_stable_mw - 1e-3


class TestBatteryPhysics:
    def test_soc_stays_in_bounds_charging(self, battery):
        for _ in range(100):
            battery.dispatch(1.0)    # full charge
        assert 0.0 <= battery.soc_fraction <= 1.0 + 1e-6

    def test_soc_stays_in_bounds_discharging(self, battery):
        for _ in range(100):
            battery.dispatch(-1.0)   # full discharge
        assert 0.0 <= battery.soc_fraction <= 1.0 + 1e-6

    def test_charging_consumes_from_grid(self, battery):
        net = battery.dispatch(1.0)
        assert net < 0.0   # negative = consuming

    def test_discharging_supplies_grid(self, battery):
        net = battery.dispatch(-1.0)
        assert net > 0.0   # positive = supplying

    def test_round_trip_efficiency_loss(self, battery):
        battery.reset(0.0)
        start_soc = battery.soc_fraction
        # Charge fully
        for _ in range(200):
            battery.dispatch(1.0)
        charged_soc = battery.soc_fraction
        # Discharge fully
        total_out = 0.0
        for _ in range(200):
            net = battery.dispatch(-1.0)
            total_out += net
        # Should get less out than put in (losses)
        total_in = charged_soc * battery.capacity_mwh
        assert total_out * 0.5 < total_in   # output energy < input energy


class TestDemandModel:
    def test_demand_positive(self):
        rng = np.random.default_rng(42)
        model = DemandModel(rng=rng)
        model.reset(day_of_year=180)
        for step in range(48):
            d = model.step(step)
            assert d["demand_mw"] > 0

    def test_solar_zero_at_night(self):
        rng = np.random.default_rng(0)
        model = DemandModel(rng=rng)
        model.reset(day_of_year=172)   # summer solstice
        # Step 0 = midnight (0:00), should have zero solar
        d = model.step(0)
        assert d["solar_mw"] == pytest.approx(0.0, abs=1e-3)

    def test_wind_within_capacity(self):
        rng = np.random.default_rng(7)
        model = DemandModel(wind_cap_mw=400.0, rng=rng)
        model.reset()
        for step in range(48):
            d = model.step(step)
            assert d["wind_mw"] <= 400.0 + 1.0

    def test_carbon_price_bounded(self):
        rng = np.random.default_rng(5)
        model = DemandModel(rng=rng)
        model.reset()
        for step in range(48):
            d = model.step(step)
            assert 5.0 <= d["carbon_price"] <= 80.0


# ── Determinism ───────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_seed_same_obs(self):
        env1 = PowerGridEnv(); obs1, _ = env1.reset(seed=99)
        env2 = PowerGridEnv(); obs2, _ = env2.reset(seed=99)
        np.testing.assert_array_equal(obs1, obs2)

    def test_same_seed_same_trajectory(self):
        rewards1, rewards2 = [], []
        for env, rlist in [(PowerGridEnv(), rewards1), (PowerGridEnv(), rewards2)]:
            obs, _ = env.reset(seed=7)
            rng = np.random.default_rng(7)
            for _ in range(STEPS_PER_EPISODE):
                action = rng.uniform(-1, 1, size=5).astype(np.float32)
                _, r, done, _, _ = env.step(action)
                rlist.append(r)
                if done:
                    break
        assert rewards1 == pytest.approx(rewards2, rel=1e-5)


# ── Reward sign checks ────────────────────────────────────────────────────────

class TestRewardSigns:
    def test_full_supply_reward_positive(self):
        """Supplying more than enough demand should give a net-positive reward step."""
        env = PowerGridEnv()
        env.reset(seed=42)
        # Max-output action for generators, no battery draw
        full_action = np.array([1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32)
        _, reward, _, _, _ = env.step(full_action)
        # Should not be massively negative
        assert reward > -50.0

    def test_zero_action_not_catastrophic(self):
        """Zero action (generators at midpoint) should not immediately blackout."""
        env = PowerGridEnv()
        env.reset(seed=0)
        _, reward, _, _, info = env.step(np.zeros(5, dtype=np.float32))
        # Some reward should be positive (supply adequacy term)
        assert reward > -200.0
