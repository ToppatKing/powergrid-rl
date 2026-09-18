"""Plot training metrics emitted by main.py.

Usage:
    python main.py 2>&1 | tee training.log
    python plot_results.py training.log

The figure is written to training_results.png by default.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


EPISODE_PATTERN = re.compile(
    r"Episode:\s*(?P<episode>\d+)\s*\|\s*"
    r"Return:\s*(?P<return>[-+]?\d*\.?\d+)\s*\|\s*"
    r"Final Hz:\s*(?P<frequency>[-+]?\d*\.?\d+)\s*\|\s*"
    r"Time:\s*(?P<time>[-+]?\d*\.?\d+)h"
)


def read_metrics(log_path):
    """Extract episode metrics from the standard main.py log format."""
    metrics = []
    for line in Path(log_path).read_text(encoding="utf-8").splitlines():
        match = EPISODE_PATTERN.search(line)
        if match:
            metrics.append(
                {
                    "episode": int(match.group("episode")),
                    "return": float(match.group("return")),
                    "frequency": float(match.group("frequency")),
                    "time": float(match.group("time")),
                }
            )

    if not metrics:
        raise ValueError(
            f"No episode metrics found in {log_path}. "
            "Expected lines like: Episode: 10 | Return: -12.3 | "
            "Final Hz: 60.00 | Time: 24.0h"
        )
    return metrics


def rolling_mean(values, window):
    """Return a trailing mean without requiring pandas."""
    means = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        sample = values[start : index + 1]
        means.append(sum(sample) / len(sample))
    return means


def plot_metrics(metrics, output_path, window):
    episodes = [item["episode"] for item in metrics]
    returns = [item["return"] for item in metrics]
    frequencies = [item["frequency"] for item in metrics]

    figure, (return_axis, frequency_axis) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, constrained_layout=True
    )

    return_axis.plot(episodes, returns, color="#2563eb", alpha=0.35, label="Episode return")
    return_axis.plot(
        episodes,
        rolling_mean(returns, window),
        color="#1d4ed8",
        linewidth=2,
        label=f"{window}-episode rolling mean",
    )
    return_axis.set_ylabel("Return")
    return_axis.set_title("Power Grid PPO Training Results")
    return_axis.grid(alpha=0.25)
    return_axis.legend()

    frequency_axis.plot(episodes, frequencies, color="#0f766e", label="Final frequency")
    frequency_axis.axhline(60.0, color="#111827", linestyle="--", linewidth=1, label="Nominal 60 Hz")
    frequency_axis.axhspan(59.3, 60.5, color="#10b981", alpha=0.12, label="Stable operating band")
    frequency_axis.set_xlabel("Episode")
    frequency_axis.set_ylabel("Frequency (Hz)")
    frequency_axis.grid(alpha=0.25)
    frequency_axis.legend()

    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Training log captured from main.py")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("training_results.png"),
        help="Output image path (default: training_results.png)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=25,
        help="Rolling-mean window in episodes (default: 25)",
    )
    args = parser.parse_args()

    if args.window < 1:
        parser.error("--window must be at least 1")

    metrics = read_metrics(args.log)
    plot_metrics(metrics, args.output, args.window)
    print(f"Plotted {len(metrics)} episodes to {args.output}")


if __name__ == "__main__":
    main()
