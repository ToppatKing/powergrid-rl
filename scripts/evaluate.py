#!/usr/bin/env python
"""Evaluate a trained PPO checkpoint on the PowerGrid environment.

Runs N episodes with the deterministic policy (mean action, no sampling),
then prints a full metrics report: returns, fuel cost, carbon, blackouts,
and per-generator dispatch statistics.

Usage::

    python scripts/evaluate.py --checkpoint results/best_model.pt --episodes 50
    python scripts/evaluate.py --checkpoint results/best_model.pt --render
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.generators import GeneratorSpec
from env.powergrid_env import make_env
from ppo.agent import PPOAgent, PPOConfig

_console = Console()


def _run_episodes(
    checkpoint: str,
    n_episodes: int,
    render: bool,
    seed: int,
) -> list[dict]:
    env = make_env(seed=seed)
    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=PPOConfig())
    agent.load(checkpoint)
    _console.print(f"[green]Loaded checkpoint:[/green] {checkpoint}")

    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0

        while not done:
            action = agent.get_deterministic_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            done = terminated or truncated
            if render:
                _console.print(env.render())

        # Collect per-generator dispatch stats from history
        history = env.episode_history
        gen_outputs = np.array([h["gen_outputs"] for h in history])  # (48, 4)

        results.append({
            "episode": ep + 1,
            "return": ep_reward,
            "cost": info["episode_cost"],
            "carbon_t": info["episode_carbon_t"],
            "unserved_mwh": info["episode_unserved_mwh"],
            "blackout": info["episode_unserved_mwh"] > 0.0,
            "mean_freq_dev": float(np.mean([abs(h["freq_dev_hz"]) for h in history])),
            "gen_mean_mw": gen_outputs.mean(axis=0).tolist(),
            "gen_max_mw":  gen_outputs.max(axis=0).tolist(),
            "mean_solar_mw": float(np.mean([h["solar_mw"]   for h in history])),
            "mean_wind_mw":  float(np.mean([h["wind_mw"]    for h in history])),
            "mean_batt_soc": float(np.mean([h["battery_soc"] for h in history])),
        })

    return results


def _print_report(results: list[dict]) -> None:
    n = len(results)
    returns     = [r["return"]       for r in results]
    costs       = [r["cost"]         for r in results]
    carbons     = [r["carbon_t"]     for r in results]
    unserved    = [r["unserved_mwh"] for r in results]
    freq_devs   = [r["mean_freq_dev"]for r in results]
    blackout_rate = np.mean([r["blackout"] for r in results]) * 100

    # ── Summary panel ─────────────────────────────────────────────────────────
    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_column("Metric", style="dim cyan", width=28)
    summary.add_column("Mean ± Std", style="bold white", width=20)
    summary.add_column("Min", style="yellow", width=12)
    summary.add_column("Max", style="green",  width=12)

    def _row(label, vals, fmt=".2f"):
        summary.add_row(
            label,
            f"{np.mean(vals):{fmt}} ± {np.std(vals):{fmt}}",
            f"{np.min(vals):{fmt}}",
            f"{np.max(vals):{fmt}}",
        )

    _row("Episode Return",         returns)
    _row("Fuel Cost ($)",          costs,    ".0f")
    _row("Carbon (tCO₂)",          carbons,  ".1f")
    _row("Unserved Energy (MWh)",  unserved, ".2f")
    _row("Mean |Δf| (Hz)",         freq_devs,".4f")
    summary.add_row("Blackout Rate",
                    f"{blackout_rate:.1f}%", "—", "—")

    _console.print(Panel(
        summary,
        title=f"[bold]Evaluation Report — {n} episodes[/bold]",
        border_style="green",
    ))

    # ── Generator dispatch table ──────────────────────────────────────────────
    gen_names = [
        GeneratorSpec.LIBRARY["coal_1"].name,
        GeneratorSpec.LIBRARY["coal_2"].name,
        GeneratorSpec.LIBRARY["gas"].name,
        GeneratorSpec.LIBRARY["nuclear"].name,
    ]
    gen_caps = [
        GeneratorSpec.LIBRARY["coal_1"].capacity_mw,
        GeneratorSpec.LIBRARY["coal_2"].capacity_mw,
        GeneratorSpec.LIBRARY["gas"].capacity_mw,
        GeneratorSpec.LIBRARY["nuclear"].capacity_mw,
    ]
    mean_gen = np.mean([r["gen_mean_mw"] for r in results], axis=0)

    gen_table = Table(title="Mean Generator Dispatch", header_style="bold cyan")
    gen_table.add_column("Generator")
    gen_table.add_column("Capacity (MW)", justify="right")
    gen_table.add_column("Mean Output (MW)", justify="right")
    gen_table.add_column("Utilisation %", justify="right")
    gen_table.add_column("Bar", width=20)

    for name, cap, mean_out in zip(gen_names, gen_caps, mean_gen):
        util = mean_out / cap
        bar = "█" * int(util * 20) + "░" * (20 - int(util * 20))
        gen_table.add_row(
            name,
            f"{cap:.0f}",
            f"{mean_out:.1f}",
            f"{util*100:.1f}%",
            bar,
        )

    mean_solar = np.mean([r["mean_solar_mw"] for r in results])
    mean_wind  = np.mean([r["mean_wind_mw"]  for r in results])
    mean_soc   = np.mean([r["mean_batt_soc"] for r in results])
    _console.print(gen_table)
    _console.print(
        f"  Solar: {mean_solar:.1f} MW avg  |  "
        f"Wind: {mean_wind:.1f} MW avg  |  "
        f"Battery SoC: {mean_soc*100:.1f}% avg"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO checkpoint")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                        help="Render each step to console")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed")
    args = parser.parse_args()

    results = _run_episodes(args.checkpoint, args.episodes, args.render, args.seed)
    _print_report(results)


if __name__ == "__main__":
    main()
