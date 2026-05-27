#!/usr/bin/env python
"""Training entry point for PowerGrid-RL PPO.

Usage::

    python scripts/train.py
    python scripts/train.py --config configs/default.yaml --total-steps 500000
    python scripts/train.py --seed 123 --output-dir results/run_2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Make project root importable when running from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppo.agent import PPOConfig
from ppo.trainer import Trainer


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on PowerGrid-v0")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="Override total training steps")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    # ── Apply CLI overrides ───────────────────────────────────────────────────
    train_cfg  = cfg.get("training", {})
    ppo_cfg    = cfg.get("ppo", {})

    if args.total_steps is not None:
        train_cfg["total_steps"] = args.total_steps
    if args.seed is not None:
        train_cfg["seed"] = args.seed
    if args.output_dir is not None:
        train_cfg["output_dir"] = args.output_dir
    if args.lr is not None:
        ppo_cfg["learning_rate"] = args.lr

    # ── Build PPOConfig ───────────────────────────────────────────────────────
    ppo_config = PPOConfig(
        rollout_steps    = ppo_cfg.get("rollout_steps",    2048),
        num_envs         = ppo_cfg.get("num_envs",         1),
        num_epochs       = ppo_cfg.get("num_epochs",       4),
        num_mini_batches = ppo_cfg.get("num_mini_batches", 8),
        learning_rate    = ppo_cfg.get("learning_rate",    3e-4),
        anneal_lr        = ppo_cfg.get("anneal_lr",        True),
        gamma            = ppo_cfg.get("gamma",            0.99),
        gae_lambda       = ppo_cfg.get("gae_lambda",       0.95),
        clip_coef        = ppo_cfg.get("clip_coef",        0.2),
        value_coef       = ppo_cfg.get("value_coef",       0.5),
        entropy_coef     = ppo_cfg.get("entropy_coef",     0.01),
        max_grad_norm    = ppo_cfg.get("max_grad_norm",    0.5),
        clip_value_loss  = ppo_cfg.get("clip_value_loss",  True),
        hidden_dim       = ppo_cfg.get("hidden_dim",       256),
    )

    trainer = Trainer(
        config       = ppo_config,
        total_steps  = int(train_cfg.get("total_steps",   1_000_000)),
        output_dir   = str(train_cfg.get("output_dir",    "results")),
        eval_interval= int(train_cfg.get("eval_interval", 50_000)),
        log_interval = int(train_cfg.get("log_interval",  10)),
        seed         = int(train_cfg.get("seed",          42)),
    )

    trainer.train()


if __name__ == "__main__":
    main()
