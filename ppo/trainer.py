"""Training loop for PPO on the PowerGrid-v0 environment.

Handles:
  - Rollout collection with a single environment
  - PPO updates after each full rollout
  - Running average tracking for smooth logging
  - Periodic evaluation on held-out episodes
  - Best-model checkpointing
  - Rich console progress display
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from env.powergrid_env import PowerGridEnv, make_env
from ppo.agent import PPOAgent, PPOConfig

_console = Console()


def _make_stats_table(
    global_step: int,
    total_steps: int,
    ep_rewards: deque,
    ep_costs: deque,
    ep_carbons: deque,
    ep_unserved: deque,
    update_stats: Any | None,
    sps: float,
) -> Table:
    """Render a Rich table of current training statistics."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Key", style="dim cyan", width=22)
    table.add_column("Value", style="bold white")

    def _mean(d: deque) -> str:
        return f"{np.mean(d):.2f}" if d else "—"

    table.add_row("Global step",         f"{global_step:,} / {total_steps:,}")
    table.add_row("Steps/sec",           f"{sps:.0f}")
    table.add_row("─ Episode metrics ─", "")
    table.add_row("Mean episode return", _mean(ep_rewards))
    table.add_row("Mean fuel cost ($)",  _mean(ep_costs))
    table.add_row("Mean carbon (tCO₂)",  _mean(ep_carbons))
    table.add_row("Unserved MWh",        _mean(ep_unserved))

    if update_stats is not None:
        table.add_row("─ PPO update ─",   "")
        table.add_row("Policy loss",       f"{update_stats.policy_loss:.4f}")
        table.add_row("Value loss",        f"{update_stats.value_loss:.4f}")
        table.add_row("Entropy",           f"{update_stats.entropy:.4f}")
        table.add_row("Approx KL",         f"{update_stats.approx_kl:.5f}")
        table.add_row("Clip fraction",     f"{update_stats.clip_fraction:.3f}")
        table.add_row("Explained var",     f"{update_stats.explained_var:.3f}")
    return table


class Trainer:
    """Orchestrates rollout collection and PPO updates.

    Args:
        config:          :class:`PPOConfig` hyperparameters.
        total_steps:     Total environment steps to train for.
        output_dir:      Directory for checkpoints and logs.
        eval_interval:   Evaluate every N global steps.
        log_interval:    Print stats every N updates.
        seed:            Random seed for reproducibility.
    """

    def __init__(
        self,
        config: PPOConfig | None = None,
        total_steps: int = 1_000_000,
        output_dir: str = "results",
        eval_interval: int = 50_000,
        log_interval: int = 10,
        seed: int = 42,
    ) -> None:
        self.config = config or PPOConfig()
        self.total_steps = total_steps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.seed = seed

        # Seeding
        np.random.seed(seed)

        # Environment
        self.env = make_env(seed=seed)
        obs_dim    = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        # PPO agent
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        self.agent = PPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=self.config,
            device=device,
            total_steps=total_steps,
        )

        # Metrics
        self._ep_rewards:  deque = deque(maxlen=20)
        self._ep_costs:    deque = deque(maxlen=20)
        self._ep_carbons:  deque = deque(maxlen=20)
        self._ep_unserved: deque = deque(maxlen=20)
        self._best_eval_return = -float("inf")
        self._log_rows: list[dict[str, float]] = []

        _console.print(
            Panel.fit(
                f"[bold green]PowerGrid-RL Training[/bold green]\n"
                f"Device: [cyan]{device}[/cyan]  |  "
                f"Obs: [cyan]{obs_dim}[/cyan]  |  "
                f"Actions: [cyan]{action_dim}[/cyan]  |  "
                f"Steps: [cyan]{total_steps:,}[/cyan]",
                title="PPO from scratch",
            )
        )

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self) -> None:
        """Run the full training loop."""
        obs, _ = self.env.reset(seed=self.seed)
        done = False
        ep_reward = 0.0

        global_step = 0
        update_count = 0
        last_stats = None
        last_eval_step = 0
        t_start = time.perf_counter()

        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=_console,
            transient=False,
        )
        task = progress.add_task("Training", total=self.total_steps)

        with Live(progress, console=_console, refresh_per_second=4):
            while global_step < self.total_steps:

                # ── Collect one rollout ───────────────────────────────────
                for _ in range(self.config.rollout_steps):
                    action, log_prob, value = self.agent.get_action(obs)
                    next_obs, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated

                    self.agent.buffer.add(
                        obs=obs,
                        action=action,
                        log_prob=log_prob,
                        reward=float(reward),
                        value=value,
                        done=done,
                    )

                    obs = next_obs
                    ep_reward += float(reward)
                    global_step += 1
                    progress.update(task, advance=1)

                    if done:
                        self._ep_rewards.append(ep_reward)
                        self._ep_costs.append(info["episode_cost"])
                        self._ep_carbons.append(info["episode_carbon_t"])
                        self._ep_unserved.append(info["episode_unserved_mwh"])
                        ep_reward = 0.0
                        obs, _ = self.env.reset()
                        done = False

                    if global_step >= self.total_steps:
                        break

                # ── PPO update ────────────────────────────────────────────
                last_stats = self.agent.update(obs, done)
                update_count += 1

                # ── Logging ───────────────────────────────────────────────
                if update_count % self.log_interval == 0:
                    elapsed = time.perf_counter() - t_start
                    sps = global_step / (elapsed + 1e-8)
                    tbl = _make_stats_table(
                        global_step, self.total_steps,
                        self._ep_rewards, self._ep_costs,
                        self._ep_carbons, self._ep_unserved,
                        last_stats, sps,
                    )
                    _console.print(
                        Panel(tbl, title=f"[bold]Update {update_count}[/bold]",
                              border_style="dim blue")
                    )

                    if self._ep_rewards:
                        self._log_rows.append({
                            "step": global_step,
                            "mean_return": float(np.mean(self._ep_rewards)),
                            "mean_cost": float(np.mean(self._ep_costs)),
                            "mean_carbon": float(np.mean(self._ep_carbons)),
                            "policy_loss": last_stats.policy_loss,
                            "value_loss": last_stats.value_loss,
                            "entropy": last_stats.entropy,
                            "approx_kl": last_stats.approx_kl,
                            "explained_var": last_stats.explained_var,
                        })

                # ── Periodic evaluation ───────────────────────────────────
                if global_step - last_eval_step >= self.eval_interval:
                    eval_return = self._evaluate(n_episodes=5)
                    last_eval_step = global_step
                    _console.print(
                        f"[bold yellow]Eval @ {global_step:,}[/bold yellow]  "
                        f"mean return = [bold]{eval_return:.2f}[/bold]"
                    )
                    if eval_return > self._best_eval_return:
                        self._best_eval_return = eval_return
                        self.agent.save(str(self.output_dir / "best_model.pt"))
                        _console.print(
                            f"  [green]✓ New best model saved "
                            f"(return={eval_return:.2f})[/green]"
                        )

        # Final checkpoint and CSV log
        self.agent.save(str(self.output_dir / "final_model.pt"))
        self._save_log()
        elapsed = time.perf_counter() - t_start
        _console.print(
            f"\n[bold green]Training complete![/bold green]  "
            f"Elapsed: {elapsed/60:.1f} min  |  "
            f"Best eval return: {self._best_eval_return:.2f}"
        )

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _evaluate(self, n_episodes: int = 5) -> float:
        """Run N deterministic episodes and return mean episode return."""
        eval_env = make_env(seed=self.seed + 9999)
        returns = []
        for ep in range(n_episodes):
            obs, _ = eval_env.reset(seed=self.seed + 9999 + ep)
            ep_ret = 0.0
            done = False
            while not done:
                action = self.agent.get_deterministic_action(obs)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                ep_ret += float(reward)
                done = terminated or truncated
            returns.append(ep_ret)
        return float(np.mean(returns))

    # ── Log saving ────────────────────────────────────────────────────────────

    def _save_log(self) -> None:
        """Save training log as CSV."""
        if not self._log_rows:
            return
        import csv
        path = self.output_dir / "training_log.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._log_rows[0].keys())
            writer.writeheader()
            writer.writerows(self._log_rows)
        _console.print(f"[dim]Training log saved to {path}[/dim]")
