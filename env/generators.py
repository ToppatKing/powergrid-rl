"""Generator physics models for the PowerGrid-v0 environment.

Each generator has:
  - Capacity limits (min/max output in MW)
  - Ramp-rate constraints (max MW change per 30-min interval)
  - Fuel cost curve (quadratic heat-rate model → $/MWh)
  - Carbon intensity (tCO₂/MWh)
  - Start-up inertia (minimum output when online)

All values are grounded in real-world engineering data from
NREL, EIA, and the UK National Grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np


@dataclass
class GeneratorSpec:
    """Static specification for one generator unit.

    Attributes:
        name:          Human-readable label.
        capacity_mw:   Maximum nameplate output (MW).
        min_stable_mw: Minimum stable output when online (MW).
        ramp_rate_mw:  Maximum output change per 30-min step (MW).
        cost_a:        Quadratic fuel-cost coefficient ($/MWh²).
        cost_b:        Linear fuel-cost coefficient ($/MWh).
        cost_c:        Fixed operating cost per step ($).
        carbon_t_mwh:  CO₂ intensity (tCO₂ per MWh generated).
        start_cost:    One-time cost if unit transitions from 0 → online ($).
    """

    name: str
    capacity_mw: float
    min_stable_mw: float
    ramp_rate_mw: float
    cost_a: float          # $/MWh² (quadratic)
    cost_b: float          # $/MWh  (linear)
    cost_c: float          # $/step (fixed)
    carbon_t_mwh: float    # tCO₂/MWh
    start_cost: float      # $ per cold start

    # ── Library of real-world generator specs ────────────────────────────────
    LIBRARY: ClassVar[dict[str, "GeneratorSpec"]]

    def cost_per_step(self, output_mw: float) -> float:
        """Compute fuel + variable operating cost for one 30-min dispatch step.

        Uses a quadratic heat-rate model:
            cost = (a·P² + b·P + c) × 0.5   [half-hour step]

        Args:
            output_mw: Actual output dispatched this step (MW).

        Returns:
            Cost in USD for this half-hour interval.
        """
        if output_mw <= 0.0:
            return 0.0
        return (self.cost_a * output_mw ** 2 + self.cost_b * output_mw + self.cost_c) * 0.5

    def carbon_per_step(self, output_mw: float) -> float:
        """CO₂ emitted (tonnes) in one 30-min dispatch step."""
        return self.carbon_t_mwh * output_mw * 0.5  # MWh = MW × 0.5 h


# Build the class-level library after the dataclass is defined
GeneratorSpec.LIBRARY = {
    "coal_1": GeneratorSpec(
        name="Coal-1",
        capacity_mw=500.0,
        min_stable_mw=150.0,
        ramp_rate_mw=50.0,      # slow: coal takes ~10 min/% to ramp
        cost_a=0.0012,
        cost_b=30.0,
        cost_c=800.0,
        carbon_t_mwh=0.82,
        start_cost=15_000.0,
    ),
    "coal_2": GeneratorSpec(
        name="Coal-2",
        capacity_mw=400.0,
        min_stable_mw=120.0,
        ramp_rate_mw=40.0,
        cost_a=0.0014,
        cost_b=32.0,
        cost_c=700.0,
        carbon_t_mwh=0.82,
        start_cost=12_000.0,
    ),
    "gas": GeneratorSpec(
        name="Gas CCGT",
        capacity_mw=250.0,
        min_stable_mw=50.0,
        ramp_rate_mw=125.0,     # very flexible: CCGT ramps ~50%/min
        cost_a=0.0020,
        cost_b=55.0,
        cost_c=400.0,
        carbon_t_mwh=0.45,
        start_cost=3_000.0,
    ),
    "nuclear": GeneratorSpec(
        name="Nuclear",
        capacity_mw=900.0,
        min_stable_mw=700.0,    # nuclear is nearly inflexible
        ramp_rate_mw=18.0,      # ~2%/min of 900 MW
        cost_a=0.0003,
        cost_b=10.0,
        cost_c=2_500.0,
        carbon_t_mwh=0.012,     # lifecycle emissions only
        start_cost=0.0,         # never started/stopped in dispatch
    ),
}


class Generator:
    """Stateful generator unit that tracks its current output and applies
    ramp-rate constraints each step.

    Args:
        spec: The static :class:`GeneratorSpec` for this unit.
        rng:  NumPy random generator (for potential outage modelling).
    """

    def __init__(self, spec: GeneratorSpec, rng: np.random.Generator) -> None:
        self.spec = spec
        self._rng = rng
        # Current output [MW] — initialised at reset()
        self.output_mw: float = 0.0
        self._was_online: bool = False

    # ── State management ──────────────────────────────────────────────────────

    def reset(self, initial_output_fraction: float = 0.5) -> None:
        """Reset to a fraction of capacity.

        Args:
            initial_output_fraction: Fraction of capacity to start at [0, 1].
        """
        cap = self.spec.capacity_mw
        self.output_mw = float(
            np.clip(
                initial_output_fraction * cap,
                self.spec.min_stable_mw,
                cap,
            )
        )
        self._was_online = self.output_mw > 0.0

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def dispatch(self, target_mw: float) -> tuple[float, float]:
        """Attempt to move toward *target_mw* subject to ramp-rate limits.

        The target is first clamped to [min_stable, capacity] (or 0 if
        shutting down).  Then the change is bounded by the ramp rate.

        Args:
            target_mw: Desired output in MW.

        Returns:
            Tuple of (actual_output_mw, start_cost_incurred).
        """
        spec = self.spec

        # Allow complete shutdown (target <= 0 interpreted as shut-off)
        if target_mw <= 0.0:
            # Ramp down to zero (can always de-commit)
            max_decrease = spec.ramp_rate_mw
            new_output = max(0.0, self.output_mw - max_decrease)
        else:
            # Clamp to feasible operating range
            target_clamped = float(np.clip(target_mw, spec.min_stable_mw, spec.capacity_mw))
            # Apply ramp constraint
            delta = target_clamped - self.output_mw
            delta_clipped = float(np.clip(delta, -spec.ramp_rate_mw, spec.ramp_rate_mw))
            new_output = self.output_mw + delta_clipped

        # Start-up cost
        was_off = self.output_mw <= 0.0
        now_on = new_output > 0.0
        start_cost = spec.start_cost if (was_off and now_on) else 0.0

        self.output_mw = new_output
        return new_output, start_cost

    def dispatch_by_action(self, action: float) -> tuple[float, float]:
        """Dispatch using a normalized action in [-1, 1].

        The action is interpreted as:
            action = +1 → full capacity
            action =  0 → midpoint (min_stable + capacity) / 2
            action = -1 → shut down

        This mapping makes the action space symmetric and gives the agent
        an intuitive "increase / decrease" control.

        Args:
            action: Normalized action in [-1, 1].

        Returns:
            Tuple of (actual_output_mw, start_cost_incurred).
        """
        spec = self.spec
        # Map [-1, 1] → [0, capacity]
        target = (action + 1.0) / 2.0 * spec.capacity_mw
        return self.dispatch(target)

    # ── Cost / carbon accessors ───────────────────────────────────────────────

    @property
    def cost(self) -> float:
        """Fuel + variable cost for current output over a 30-min step ($)."""
        return self.spec.cost_per_step(self.output_mw)

    @property
    def carbon(self) -> float:
        """CO₂ emitted (tonnes) for current output over a 30-min step."""
        return self.spec.carbon_per_step(self.output_mw)

    @property
    def output_fraction(self) -> float:
        """Normalised output: 0.0 = off, 1.0 = full capacity."""
        return self.output_mw / self.spec.capacity_mw if self.spec.capacity_mw > 0 else 0.0


class BatteryStorage:
    """Grid-scale lithium-ion battery energy storage system.

    Models:
      - Round-trip efficiency losses (charge + discharge)
      - State-of-charge (SoC) constraints
      - Power limits (max charge/discharge rate)

    Args:
        capacity_mwh: Usable energy capacity (MWh).
        max_power_mw: Max charge or discharge rate (MW).
        charge_efficiency: Fraction of energy retained when charging [0, 1].
        discharge_efficiency: Fraction of stored energy recovered [0, 1].
    """

    def __init__(
        self,
        capacity_mwh: float = 1000.0,
        max_power_mw: float = 200.0,
        charge_efficiency: float = 0.92,
        discharge_efficiency: float = 0.92,
    ) -> None:
        self.capacity_mwh = capacity_mwh
        self.max_power_mw = max_power_mw
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.soc_mwh: float = capacity_mwh * 0.5  # start at 50% SoC

    def reset(self, initial_soc_fraction: float = 0.5) -> None:
        self.soc_mwh = float(
            np.clip(initial_soc_fraction * self.capacity_mwh, 0.0, self.capacity_mwh)
        )

    @property
    def soc_fraction(self) -> float:
        """State of charge as a fraction [0, 1]."""
        return self.soc_mwh / self.capacity_mwh

    def dispatch(self, action: float) -> float:
        """Charge (+) or discharge (−) the battery.

        Args:
            action: Normalised in [-1, 1].
                    +1 = charge at max rate, -1 = discharge at max rate.

        Returns:
            Net power flow into the grid (MW).
            Positive = discharging (supplying grid).
            Negative = charging (consuming from grid).
        """
        # Requested power (MW) — positive = charge, negative = discharge
        requested_mw = action * self.max_power_mw

        if requested_mw >= 0.0:  # charging
            # Can't charge more than headroom allows
            headroom_mwh = self.capacity_mwh - self.soc_mwh
            max_charge_mw = headroom_mwh / (0.5 * self.charge_efficiency)
            actual_mw = min(requested_mw, self.max_power_mw, max_charge_mw)
            # Energy stored accounts for round-trip losses
            self.soc_mwh += actual_mw * 0.5 * self.charge_efficiency
            net_grid_power = -actual_mw   # consuming from grid
        else:  # discharging
            # Can't discharge more than SoC allows
            available_mwh = self.soc_mwh
            max_discharge_mw = available_mwh / (0.5 / self.discharge_efficiency)
            actual_mw = max(requested_mw, -self.max_power_mw, -max_discharge_mw)
            self.soc_mwh += actual_mw * 0.5 / self.discharge_efficiency  # actual_mw < 0
            net_grid_power = -actual_mw   # supplying to grid (positive)

        self.soc_mwh = float(np.clip(self.soc_mwh, 0.0, self.capacity_mwh))
        return net_grid_power
