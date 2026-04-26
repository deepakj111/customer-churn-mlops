"""
Entry point for running the training pipeline manually.

Usage:
    poetry run python main.py
    # or via Makefile:
    make train
"""

import sys

from src.models.train import run_training_experiment


def main() -> None:
    """Run a manual training experiment and print all metrics."""
    metrics = run_training_experiment(run_name="manual-training-run")
    print("Training finished")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        sys.exit(1)
