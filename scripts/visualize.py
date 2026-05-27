#!/usr/bin/env python
"""Episode visualisation — renders a 6-panel matplotlib dashboard.

Panels:
  1. Supply vs Demand (MW) — stacked area chart
  2. Grid frequency deviation (Hz)
  3. Battery state of charge (%)
  4. Generator dispatch over the day
  5. Cumulative reward
  6. Per-step cost and carbon

Usage::

    python scripts/visualize.py --checkpoint results/best_model.pt
    python scripts/visualize.py --checkpoint results/best_model.pt --save episode.png
    python scripts/visualize.py --random          # random policy baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.generators import GeneratorSpec
from env.powergrid_env import make_env
from ppo.agent import PPOAgent, PPOConfig


def _run_episode(checkpoint: str | None, seed: int) -> list[dict]:
    env = make_env(seed=seed)
    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if checkpoint:
        agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=PPOConfig())
        agent.load(checkpoint)
        policy_label = f"PPO ({Path(checkpoint).stem})"

        def act(o):
            return agent.get_deterministic_action(o)
    else:
        policy_label = "Random policy"
        def act(o):
            return env.action_space.sample()

    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        action = act(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    print(f"Policy: {policy_label}")
    return env.episode_history


def _plot(history: list[dict], save_path: str | None) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("Install matplotlib: pip install matplotlib")
        return

    steps      = [h["step"] for h in history]
    hours      = [s * 0.5 for s in steps]          # 30-min steps → hours
    demand     = [h["demand_mw"]   for h in history]
    supply     = [h["supply_mw"]   for h in history]
    solar      = [h["solar_mw"]    for h in history]
    wind       = [h["wind_mw"]     for h in history]
    battery    = [h["battery_mw"]  for h in history]
    soc        = [h["battery_soc"] * 100 for h in history]
    freq       = [h["freq_dev_hz"] for h in history]
    cost       = [h["cost"]        for h in history]
    carbon     = [h["carbon"]      for h in history]
    reward     = np.cumsum([h["reward"] for h in history])
    gen_outs   = np.array([h["gen_outputs"] for h in history])  # (48, 4)

    gen_names  = ["Coal-1", "Coal-2", "Gas CCGT", "Nuclear"]
    gen_colors = ["#4a4a4a", "#666666", "#e67e22", "#2ecc71"]
    ren_colors = {"Solar": "#f39c12", "Wind": "#3498db", "Battery": "#9b59b6"}

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("PowerGrid-v0 — Episode Dispatch Dashboard", fontsize=14, fontweight="bold")

    # ── Panel 1: Supply stack vs demand ──────────────────────────────────────
    ax = axes[0, 0]
    bottom = np.zeros(len(steps))
    for i, (name, color) in enumerate(zip(gen_names, gen_colors)):
        vals = gen_outs[:, i]
        ax.fill_between(hours, bottom, bottom + vals, alpha=0.85,
                        color=color, label=name, step="mid")
        bottom += vals

    # Renewables on top
    ax.fill_between(hours, bottom, bottom + solar, alpha=0.7,
                    color=ren_colors["Solar"], label="Solar", step="mid")
    bottom_s = bottom + np.array(solar)
    ax.fill_between(hours, bottom_s, bottom_s + wind, alpha=0.7,
                    color=ren_colors["Wind"], label="Wind", step="mid")

    ax.plot(hours, demand, "r--", lw=2, label="Demand", zorder=5)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("MW")
    ax.set_title("Supply Stack vs Demand")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.set_xlim(0, 24); ax.grid(alpha=0.3)

    # ── Panel 2: Frequency deviation ─────────────────────────────────────────
    ax = axes[0, 1]
    ax.fill_between(hours, freq, 0, where=np.array(freq) >= 0,
                    alpha=0.6, color="#2ecc71", label="Over-freq")
    ax.fill_between(hours, freq, 0, where=np.array(freq) < 0,
                    alpha=0.6, color="#e74c3c", label="Under-freq")
    ax.axhline(0, color="black", lw=1)
    ax.axhline( 0.5, color="orange", lw=1, ls="--", label="±0.5 Hz warn")
    ax.axhline(-0.5, color="orange", lw=1, ls="--")
    ax.set_xlabel("Hour"); ax.set_ylabel("Δf (Hz)")
    ax.set_title("Grid Frequency Deviation")
    ax.legend(fontsize=8); ax.set_xlim(0, 24); ax.grid(alpha=0.3)

    # ── Panel 3: Battery SoC ──────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(hours, soc, color="#9b59b6", lw=2)
    ax.fill_between(hours, soc, alpha=0.3, color="#9b59b6")
    ax.axhline(20, color="red",   ls="--", lw=1, label="20% floor")
    ax.axhline(80, color="green", ls="--", lw=1, label="80% ceiling")
    ax.set_xlabel("Hour"); ax.set_ylabel("State of Charge (%)")
    ax.set_title("Battery State of Charge")
    ax.set_ylim(0, 100); ax.legend(fontsize=8)
    ax.set_xlim(0, 24); ax.grid(alpha=0.3)

    # ── Panel 4: Individual generator dispatch ────────────────────────────────
    ax = axes[1, 1]
    for i, (name, color) in enumerate(zip(gen_names, gen_colors)):
        ax.plot(hours, gen_outs[:, i], color=color, lw=1.8, label=name)
    ax.set_xlabel("Hour"); ax.set_ylabel("Output (MW)")
    ax.set_title("Generator Dispatch")
    ax.legend(fontsize=8); ax.set_xlim(0, 24); ax.grid(alpha=0.3)

    # ── Panel 5: Cumulative reward ────────────────────────────────────────────
    ax = axes[2, 0]
    ax.plot(hours, reward, color="#1abc9c", lw=2)
    ax.fill_between(hours, reward, alpha=0.3, color="#1abc9c")
    ax.set_xlabel("Hour"); ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Episode Reward")
    ax.set_xlim(0, 24); ax.grid(alpha=0.3)

    # ── Panel 6: Cost + carbon ────────────────────────────────────────────────
    ax = axes[2, 1]
    ax2 = ax.twinx()
    ax.bar(hours, cost,   width=0.4, alpha=0.7, color="#e74c3c", label="Fuel cost ($)")
    ax2.plot(hours, carbon, color="#f39c12", lw=1.5, label="Carbon (tCO₂)")
    ax.set_xlabel("Hour"); ax.set_ylabel("Cost ($)", color="#e74c3c")
    ax2.set_ylabel("Carbon (tCO₂)", color="#f39c12")
    ax.set_title("Per-Step Fuel Cost & Emissions")
    ax.set_xlim(0, 24); ax.grid(alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize PPO episode")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pt checkpoint (omit for random policy)")
    parser.add_argument("--random", action="store_true",
                        help="Force random policy baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default=None,
                        help="Save figure to this path instead of showing")
    args = parser.parse_args()

    ckpt = None if args.random else args.checkpoint
    history = _run_episode(ckpt, seed=args.seed)
    _plot(history, save_path=args.save)


if __name__ == "__main__":
    main()
