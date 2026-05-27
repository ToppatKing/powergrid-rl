"""Stochastic electricity demand and renewable generation models.

Demand is modelled as:
    D(t) = base_load × profile(t) × season(t) × (1 + ε_t)

where:
    - profile(t)  is a smooth daily shape (morning/evening peaks)
    - season(t)   is annual amplitude modulation (higher in winter)
    - ε_t         is IID Gaussian noise (σ = 5%)

Solar and wind are correlated with weather state (Markov chain with 3
weather regimes: sunny, cloudy, stormy).

All outputs are in MW.  The caller provides the current simulation time
(step index within episode and episode seed).
"""

from __future__ import annotations

import numpy as np


# ── Weather state machine ─────────────────────────────────────────────────────

# 3 regimes: 0=sunny, 1=cloudy, 2=stormy
_WEATHER_TRANSITION = np.array(
    [
        [0.70, 0.25, 0.05],   # sunny → {sunny, cloudy, stormy}
        [0.30, 0.50, 0.20],   # cloudy → ...
        [0.10, 0.40, 0.50],   # stormy → ...
    ]
)

# (solar_fraction_mean, solar_fraction_std, wind_fraction_mean, wind_fraction_std)
_WEATHER_PARAMS: dict[int, tuple[float, float, float, float]] = {
    0: (0.80, 0.08, 0.30, 0.10),   # sunny: lots of solar, moderate wind
    1: (0.35, 0.10, 0.55, 0.12),   # cloudy: low solar, good wind
    2: (0.05, 0.03, 0.70, 0.15),   # stormy: almost no solar, high wind
}


class DemandModel:
    """Simulates electricity demand and renewable generation for one episode.

    Args:
        base_load_mw:   Mean system demand (MW).
        solar_cap_mw:   Installed solar capacity (MW).
        wind_cap_mw:    Installed wind capacity (MW).
        noise_sigma:    Fractional demand noise (default 5%).
        rng:            NumPy random generator.
    """

    def __init__(
        self,
        base_load_mw: float = 1200.0,
        solar_cap_mw: float = 300.0,
        wind_cap_mw: float = 400.0,
        noise_sigma: float = 0.05,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.base_load_mw = base_load_mw
        self.solar_cap_mw = solar_cap_mw
        self.wind_cap_mw = wind_cap_mw
        self.noise_sigma = noise_sigma
        self._rng = rng or np.random.default_rng()

        # State
        self.weather_state: int = 0
        self._day_of_year: int = 0
        self._carbon_price: float = 25.0    # $/tCO₂

    # ── Episode initialisation ────────────────────────────────────────────────

    def reset(self, day_of_year: int | None = None) -> None:
        """Randomise weather state and carbon price for a new episode.

        Args:
            day_of_year: 0–364; randomised if None.
        """
        self._day_of_year = (
            int(self._rng.integers(0, 365))
            if day_of_year is None
            else int(day_of_year % 365)
        )
        # Start in a random weather regime
        self.weather_state = int(self._rng.integers(0, 3))
        # Carbon price: geometric Brownian motion, mean-reverting around $25
        self._carbon_price = float(self._rng.uniform(15.0, 40.0))

    # ── Step update ───────────────────────────────────────────────────────────

    def step(self, step_index: int) -> dict[str, float]:
        """Advance weather, return demand and renewable output for this step.

        Args:
            step_index: Current step within the episode (0–47 for a 24 h day).

        Returns:
            Dictionary with keys:
              - ``demand_mw``:      Gross electricity demand (MW).
              - ``solar_mw``:       Solar generation available (MW).
              - ``wind_mw``:        Wind generation available (MW).
              - ``solar_irr``:      Normalised solar irradiance [0, 1].
              - ``wind_speed_n``:   Normalised wind speed [0, 1].
              - ``carbon_price``:   CO₂ price ($/tCO₂).
              - ``hour_of_day``:    Float hour (0–24).
              - ``day_of_year``:    Integer (0–364).
              - ``forecast_error``: Relative surprise vs. day-ahead (fraction).
        """
        # ── Update weather (Markov transition) ───────────────────────────────
        self.weather_state = int(
            self._rng.choice(3, p=_WEATHER_TRANSITION[self.weather_state])
        )

        # ── Carbon price: mean-reverting GBM ─────────────────────────────────
        mu = 25.0          # $/tCO₂ long-run mean
        theta = 0.05       # reversion speed
        sigma_c = 0.8      # volatility per step
        self._carbon_price += theta * (mu - self._carbon_price) + sigma_c * self._rng.standard_normal()
        self._carbon_price = float(np.clip(self._carbon_price, 5.0, 80.0))

        # ── Time features ─────────────────────────────────────────────────────
        hour_of_day = (step_index % 48) * 0.5          # 30 min steps → 0–23.5
        day = (self._day_of_year + step_index // 48) % 365

        # ── Demand shape: double-peak (morning + evening) ────────────────────
        profile = self._demand_profile(hour_of_day)
        season = self._seasonality(day)
        noise = float(1.0 + self._rng.normal(0.0, self.noise_sigma))
        demand_mw = float(np.clip(self.base_load_mw * profile * season * noise, 200.0, 2500.0))

        # Day-ahead forecast (computed at start of step with noise)
        forecast_mw = self.base_load_mw * profile * season
        forecast_error = (demand_mw - forecast_mw) / (forecast_mw + 1e-6)

        # ── Renewable output ─────────────────────────────────────────────────
        solar_irr, solar_noise, wind_frac, wind_noise = _WEATHER_PARAMS[self.weather_state]

        # Solar: zero at night, peaks at midday
        daylight = self._solar_daylight(hour_of_day, day)
        solar_irr_actual = float(np.clip(
            solar_irr * daylight + self._rng.normal(0.0, solar_noise) * daylight,
            0.0, 1.0,
        ))
        solar_mw = solar_irr_actual * self.solar_cap_mw

        # Wind: stochastic around weather-regime mean
        wind_frac_actual = float(np.clip(
            wind_frac + self._rng.normal(0.0, wind_noise),
            0.0, 1.0,
        ))
        wind_mw = wind_frac_actual * self.wind_cap_mw

        return {
            "demand_mw": demand_mw,
            "solar_mw": solar_mw,
            "wind_mw": wind_mw,
            "solar_irr": solar_irr_actual,
            "wind_speed_n": wind_frac_actual,
            "carbon_price": self._carbon_price,
            "hour_of_day": hour_of_day,
            "day_of_year": float(day),
            "forecast_error": float(np.clip(forecast_error, -1.0, 1.0)),
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _demand_profile(hour: float) -> float:
        """Smooth double-peak demand shape.

        Calibrated to resemble UK National Grid half-hourly profiles:
        - Morning peak ~8:00 (1.15× base)
        - Evening peak ~18:30 (1.25× base)
        - Night trough ~3:00 (0.65× base)
        """
        # Sum of two Gaussians on [0, 24), plus a flat base
        morning = 0.25 * np.exp(-0.5 * ((hour - 8.0) / 2.0) ** 2)
        evening = 0.35 * np.exp(-0.5 * ((hour - 18.5) / 1.8) ** 2)
        return float(0.65 + morning + evening)

    @staticmethod
    def _seasonality(day: int) -> float:
        """Annual seasonality factor (higher demand in winter).

        Uses a cosine with peak on Jan 1 (day 0) and trough on Jul 2 (day 182).
        Amplitude ±15% around 1.0.
        """
        return float(1.0 + 0.15 * np.cos(2 * np.pi * day / 365))

    @staticmethod
    def _solar_daylight(hour: float, day: int) -> float:
        """Fraction of solar irradiance available (accounts for day length).

        Simplified: sunrise/sunset varies ±3 h around 6:00/18:00 with season.
        """
        # Day length: 9 h in winter, 15 h in summer
        day_length = 12.0 + 3.0 * np.cos(np.pi + 2 * np.pi * day / 365)
        sunrise = 12.0 - day_length / 2.0
        sunset = 12.0 + day_length / 2.0

        if hour < sunrise or hour > sunset:
            return 0.0
        # Sinusoidal within daylight window
        t = (hour - sunrise) / (sunset - sunrise)   # 0 → 1
        return float(np.sin(np.pi * t))
