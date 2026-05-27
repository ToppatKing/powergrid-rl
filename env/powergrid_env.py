"""PowerGrid-v0: Custom Gymnasium environment for power grid dispatch.

Observation space: Box(20,) — see README for full feature table.
Action space:      Box(5,)  — continuous setpoints in [-1, 1].
Reward:            Composite — supply adequacy, cost, carbon, frequency, blackouts.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.demand_model import DemandModel
from env.generators import BatteryStorage, Generator, GeneratorSpec

# ── Constants ──────────────────────────────────────────────────────────────────

STEPS_PER_EPISODE = 48          # 48 × 30 min = 24 h
NOMINAL_FREQ_HZ = 50.0          # European grid nominal frequency
MAX_FREQ_DEV_HZ = 1.0           # ±1 Hz before tripping
FREQUENCY_INERTIA = 8.0         # MW·s²  (synthetic inertia constant)

# Reward weights
W_SUPPLY   = 5.0    # reward for meeting demand
W_COST     = 0.002  # penalty per $ of fuel cost
W_CARBON   = 0.05   # penalty per tCO₂ × carbon_price
W_FREQ     = 3.0    # penalty per (Hz deviation)²
W_RAMP     = 0.5    # penalty per MW of ramp violation
W_BLACKOUT = 50.0   # penalty per MWh of unserved energy


class PowerGridEnv(gym.Env):
    """Physics-based power grid economic dispatch environment.

    The agent controls 4 thermal generators and a battery storage unit
    to meet stochastic electricity demand across a simulated 24-hour day.

    Observation (20-dim Box):
        [0–3]   generator output fractions (normalised)
        [4]     battery SoC fraction
        [5]     solar output fraction
        [6]     wind output fraction
        [7]     net demand fraction (after renewables, normalised)
        [8]     grid frequency deviation (normalised to [-1, 1])
        [9–10]  sin/cos(hour_of_day / 24 × 2π)
        [11–12] sin/cos(day_of_year / 365 × 2π)
        [13]    solar irradiance [0, 1]
        [14]    wind speed normalised [0, 1]
        [15]    carbon price normalised [0, 1]
        [16]    demand forecast error [-1, 1]
        [17]    total supply / max supply capacity
        [18]    demand / max capacity
        [19]    battery power headroom fraction

    Action (5-dim Box[-1, 1]):
        [0]  Coal-1 setpoint    (−1=off, +1=full capacity)
        [1]  Coal-2 setpoint
        [2]  Gas CCGT setpoint
        [3]  Nuclear setpoint
        [4]  Battery            (−1=discharge, +1=charge)

    Reward:
        Composite scalar described in README reward function.
    """

    metadata = {"render_modes": ["ansi", "rgb_array"]}

    # Maximum system capacity (MW) — used for normalisation
    MAX_CAPACITY_MW = (
        GeneratorSpec.LIBRARY["coal_1"].capacity_mw
        + GeneratorSpec.LIBRARY["coal_2"].capacity_mw
        + GeneratorSpec.LIBRARY["gas"].capacity_mw
        + GeneratorSpec.LIBRARY["nuclear"].capacity_mw
        + 200.0   # battery max discharge
        + 300.0   # solar cap
        + 400.0   # wind cap
    )  # ≈ 2450 MW

    def __init__(
        self,
        render_mode: str | None = None,
        base_load_mw: float = 1200.0,
        max_carbon_price: float = 80.0,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self._max_carbon_price = max_carbon_price

        # ── Action space: 5 continuous controls in [-1, 1] ───────────────────
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # ── Observation space: 20-dim normalised features ─────────────────────
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(20,), dtype=np.float32
        )

        # ── Sub-components ────────────────────────────────────────────────────
        self._rng = np.random.default_rng()
        self._demand_model = DemandModel(
            base_load_mw=base_load_mw,
            solar_cap_mw=300.0,
            wind_cap_mw=400.0,
            rng=self._rng,
        )
        self._generators: list[Generator] = [
            Generator(GeneratorSpec.LIBRARY["coal_1"], self._rng),
            Generator(GeneratorSpec.LIBRARY["coal_2"], self._rng),
            Generator(GeneratorSpec.LIBRARY["gas"],    self._rng),
            Generator(GeneratorSpec.LIBRARY["nuclear"],self._rng),
        ]
        self._battery = BatteryStorage(
            capacity_mwh=1000.0, max_power_mw=200.0
        )

        # ── Episode state ─────────────────────────────────────────────────────
        self._step_idx: int = 0
        self._freq_dev_hz: float = 0.0
        self._last_demand: dict[str, float] = {}

        # ── Logging for render / info ─────────────────────────────────────────
        self._episode_cost: float = 0.0
        self._episode_carbon: float = 0.0
        self._episode_unserved_mwh: float = 0.0
        self._episode_reward: float = 0.0
        self._history: list[dict[str, Any]] = []

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._demand_model._rng = self._rng
            for g in self._generators:
                g._rng = self._rng

        self._demand_model.reset()
        self._step_idx = 0
        self._freq_dev_hz = 0.0

        # Reset generators to sensible initial dispatch
        for i, gen in enumerate(self._generators):
            # Nuclear always at high output; others at moderate levels
            fracs = [0.6, 0.5, 0.3, 0.85]
            gen.reset(initial_output_fraction=fracs[i])

        self._battery.reset(initial_soc_fraction=0.5)

        # Get first demand reading
        self._last_demand = self._demand_model.step(0)

        # Reset episode accumulators
        self._episode_cost = 0.0
        self._episode_carbon = 0.0
        self._episode_unserved_mwh = 0.0
        self._episode_reward = 0.0
        self._history = []

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # ── 1. Dispatch generators ────────────────────────────────────────────
        total_cost = 0.0
        total_carbon = 0.0
        total_ramp_violation = 0.0

        gen_outputs: list[float] = []
        for i, gen in enumerate(self._generators):
            prev_output = gen.output_mw
            new_output, start_cost = gen.dispatch_by_action(float(action[i]))
            total_cost += gen.cost + start_cost
            total_carbon += gen.carbon
            gen_outputs.append(new_output)

            # Ramp violation: how much did we ask beyond ramp rate?
            desired_mw = (action[i] + 1.0) / 2.0 * gen.spec.capacity_mw
            actual_delta = abs(new_output - prev_output)
            desired_delta = abs(desired_mw - prev_output)
            ramp_excess = max(0.0, desired_delta - gen.spec.ramp_rate_mw)
            total_ramp_violation += ramp_excess

        # ── 2. Dispatch battery ────────────────────────────────────────────────
        battery_net_mw = self._battery.dispatch(float(action[4]))

        # ── 3. Get demand + renewables ─────────────────────────────────────────
        d = self._demand_model.step(self._step_idx)
        self._last_demand = d

        total_supply_mw = (
            sum(gen_outputs)
            + d["solar_mw"]
            + d["wind_mw"]
            + battery_net_mw
        )
        demand_mw = d["demand_mw"]

        # ── 4. Frequency deviation dynamics ───────────────────────────────────
        # Simplified swing equation: Δf ∝ (supply − demand) / inertia
        imbalance_mw = total_supply_mw - demand_mw
        # Frequency recovers via droop control with time constant
        df = imbalance_mw / (FREQUENCY_INERTIA * 1000.0)   # normalised
        self._freq_dev_hz = float(
            np.clip(self._freq_dev_hz * 0.7 + df, -MAX_FREQ_DEV_HZ, MAX_FREQ_DEV_HZ)
        )

        # ── 5. Unserved energy (blackout) ──────────────────────────────────────
        unserved_mw = max(0.0, demand_mw - total_supply_mw)
        unserved_mwh = unserved_mw * 0.5   # 30-min step

        # ── 6. Reward ──────────────────────────────────────────────────────────
        reward = self._compute_reward(
            supply_mw=total_supply_mw,
            demand_mw=demand_mw,
            cost=total_cost,
            carbon_tonne=total_carbon,
            carbon_price=d["carbon_price"],
            freq_dev_hz=self._freq_dev_hz,
            ramp_violation_mw=total_ramp_violation,
            unserved_mwh=unserved_mwh,
        )

        # ── 7. Accumulate episode stats ────────────────────────────────────────
        self._episode_cost += total_cost
        self._episode_carbon += total_carbon
        self._episode_unserved_mwh += unserved_mwh
        self._episode_reward += reward

        self._history.append({
            "step": self._step_idx,
            "demand_mw": demand_mw,
            "supply_mw": total_supply_mw,
            "solar_mw": d["solar_mw"],
            "wind_mw": d["wind_mw"],
            "battery_mw": battery_net_mw,
            "battery_soc": self._battery.soc_fraction,
            "freq_dev_hz": self._freq_dev_hz,
            "cost": total_cost,
            "carbon": total_carbon,
            "reward": reward,
            "gen_outputs": list(gen_outputs),
        })

        self._step_idx += 1
        terminated = self._step_idx >= STEPS_PER_EPISODE
        truncated = False

        obs = self._get_obs()
        info = self._get_info()
        return obs, float(reward), terminated, truncated, info

    # ── Observation builder ───────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        d = self._last_demand
        hour = d.get("hour_of_day", 0.0)
        day  = d.get("day_of_year", 0.0)
        demand_mw = d.get("demand_mw", self._demand_model.base_load_mw)
        solar_mw  = d.get("solar_mw", 0.0)
        wind_mw   = d.get("wind_mw", 0.0)

        # Net demand after free renewables
        net_demand = max(0.0, demand_mw - solar_mw - wind_mw)

        obs = np.zeros(20, dtype=np.float32)

        # [0–3] generator output fractions
        for i, gen in enumerate(self._generators):
            obs[i] = gen.output_fraction

        # [4] battery SoC
        obs[4] = self._battery.soc_fraction

        # [5] solar fraction of capacity
        obs[5] = solar_mw / (self._demand_model.solar_cap_mw + 1e-6)

        # [6] wind fraction of capacity
        obs[6] = wind_mw / (self._demand_model.wind_cap_mw + 1e-6)

        # [7] net demand fraction of max capacity
        obs[7] = net_demand / (self.MAX_CAPACITY_MW + 1e-6)

        # [8] frequency deviation normalised to [-1, 1]
        obs[8] = self._freq_dev_hz / MAX_FREQ_DEV_HZ

        # [9–10] time-of-day encoding
        angle_h = 2 * math.pi * hour / 24.0
        obs[9]  = math.sin(angle_h)
        obs[10] = math.cos(angle_h)

        # [11–12] day-of-year encoding
        angle_d = 2 * math.pi * day / 365.0
        obs[11] = math.sin(angle_d)
        obs[12] = math.cos(angle_d)

        # [13] solar irradiance
        obs[13] = float(d.get("solar_irr", 0.0))

        # [14] wind speed normalised
        obs[14] = float(d.get("wind_speed_n", 0.0))

        # [15] carbon price normalised
        obs[15] = float(d.get("carbon_price", 25.0)) / self._max_carbon_price

        # [16] demand forecast error
        obs[16] = float(d.get("forecast_error", 0.0))

        # [17] total supply / max supply
        total_gen = sum(g.output_mw for g in self._generators)
        obs[17] = total_gen / (self.MAX_CAPACITY_MW + 1e-6)

        # [18] demand / max capacity
        obs[18] = demand_mw / (self.MAX_CAPACITY_MW + 1e-6)

        # [19] battery discharge headroom (SoC available for discharge)
        obs[19] = self._battery.soc_fraction

        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    # ── Reward function ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_reward(
        supply_mw: float,
        demand_mw: float,
        cost: float,
        carbon_tonne: float,
        carbon_price: float,
        freq_dev_hz: float,
        ramp_violation_mw: float,
        unserved_mwh: float,
    ) -> float:
        # Supply adequacy: +1 per MWh served (normalised)
        served_mwh = min(supply_mw, demand_mw) * 0.5
        r_supply = W_SUPPLY * served_mwh / 600.0   # normalise by ~max served

        # Fuel cost penalty
        r_cost = -W_COST * cost

        # Carbon emission penalty (cost × carbon intensity)
        r_carbon = -W_CARBON * carbon_tonne * carbon_price / 1000.0

        # Frequency deviation penalty (quadratic)
        r_freq = -W_FREQ * (freq_dev_hz ** 2)

        # Ramp violation penalty
        r_ramp = -W_RAMP * ramp_violation_mw / 1000.0

        # Blackout penalty
        r_blackout = -W_BLACKOUT * unserved_mwh / 100.0

        return r_supply + r_cost + r_carbon + r_freq + r_ramp + r_blackout

    # ── Info dict ─────────────────────────────────────────────────────────────

    def _get_info(self) -> dict[str, Any]:
        return {
            "step": self._step_idx,
            "episode_cost": self._episode_cost,
            "episode_carbon_t": self._episode_carbon,
            "episode_unserved_mwh": self._episode_unserved_mwh,
            "episode_reward": self._episode_reward,
            "freq_dev_hz": self._freq_dev_hz,
            "battery_soc": self._battery.soc_fraction,
            "gen_outputs_mw": [g.output_mw for g in self._generators],
            "weather_state": self._demand_model.weather_state,
        }

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self) -> str | None:
        if self.render_mode == "ansi":
            return self._render_ansi()
        return None

    def _render_ansi(self) -> str:
        if not self._history:
            return "No steps taken yet."
        h = self._history[-1]
        gens = h["gen_outputs"]
        names = ["Coal-1", "Coal-2", "Gas   ", "Nucl. "]
        lines = [
            f"Step {h['step']:02d}/48 | "
            f"Demand: {h['demand_mw']:6.0f} MW | "
            f"Supply: {h['supply_mw']:6.0f} MW | "
            f"Freq Δ: {h['freq_dev_hz']:+.3f} Hz"
        ]
        for name, mw, gen in zip(names, gens, self._generators):
            bar_len = int(20 * gen.output_fraction)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"  {name} [{bar}] {mw:5.0f}/{gen.spec.capacity_mw:.0f} MW")
        soc_bar = int(20 * self._battery.soc_fraction)
        lines.append(
            f"  Battery [{('█'*soc_bar + '░'*(20-soc_bar))}] "
            f"SoC {self._battery.soc_fraction*100:.0f}%  "
            f"Net: {h['battery_mw']:+.0f} MW"
        )
        lines.append(
            f"  Solar: {h['solar_mw']:5.0f} MW | "
            f"Wind: {h['wind_mw']:5.0f} MW | "
            f"Cost: ${h['cost']:7.0f} | "
            f"r={h['reward']:+.3f}"
        )
        return "\n".join(lines)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def episode_history(self) -> list[dict[str, Any]]:
        """Full per-step history for the current episode."""
        return self._history


# ── Registration helper ───────────────────────────────────────────────────────

def make_env(seed: int | None = None, render_mode: str | None = None) -> PowerGridEnv:
    """Create and seed a :class:`PowerGridEnv` instance."""
    env = PowerGridEnv(render_mode=render_mode)
    env.reset(seed=seed)
    return env
