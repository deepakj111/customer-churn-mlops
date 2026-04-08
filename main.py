"""
Entry point for running the training pipeline manually.

Usage:
    poetry run python main.py
    # or via Makefile:
    make train
"""

from src.models.train import run_training_experiment


def main():
    metrics = run_training_experiment(run_name="manual-training-run")
    print("Training finished")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
