"""
Unified Command-Line Interface (CLI) for Customer Churn MLOps.

This module provides a command-line interface entrypoint for all
lifecycle tasks across the platform:
    1. train      — Execute full production Prefect training & calibration pipeline
    2. retrain    — Evaluate drift triggers & run champion-challenger pipeline
    3. monitor    — Generate comprehensive population stability and drift reports
    4. uplift     — Train causal inference CATE models for prescriptive retention
    5. benchmark  — Run statistical benchmarks (LightGBM, TabPFN, FT-Transformer)
    6. serve      — Launch production FastAPI serving instance with Prometheus metrics

Usage:
    poetry run python main.py train
    poetry run python main.py retrain
    poetry run python main.py monitor
    poetry run python main.py uplift
    poetry run python main.py benchmark
    poetry run python main.py serve --port 8000
"""

from __future__ import annotations

import argparse
import sys


def cmd_train(args: argparse.Namespace) -> None:
    """Execute the full production training pipeline."""
    from pipelines.training_pipeline import run_training_pipeline

    print("🚀 Initiating Production Training Pipeline...")
    result = run_training_pipeline()
    print("\n✅ Training Pipeline Completed Successfully!")
    print(f"Optimal Threshold: {result.get('optimal_threshold')}")
    print(f"Brier Score:       {result.get('brier_score')}")
    print(f"Fairness Passed:   {result.get('fairness_passed')}")
    print(f"Pipeline Artifact: {result.get('pipeline_path')}")


def cmd_retrain(args: argparse.Namespace) -> None:
    """Execute the drift-triggered champion-challenger retraining pipeline."""
    from pipelines.retraining_pipeline import run_retraining_pipeline

    print("🔄 Evaluating Production Drift & Retraining Pipeline...")
    result = run_retraining_pipeline(force_retrain=args.force)
    print("\n✅ Retraining Pipeline Execution Complete!")
    print(f"Retrain Triggered: {result.get('retrain_triggered')}")
    print(f"Trigger Reasons:   {result.get('reasons')}")
    print(f"Challenger Promoted: {result.get('promoted')}")


def cmd_monitor(args: argparse.Namespace) -> None:
    """Generate production drift and population stability monitoring reports."""
    from scripts.generate_drift_report import main as generate_drift

    print("📊 Generating Data & Prediction Drift Report...")
    cli_args: list[str] = []
    if args.current:
        cli_args.extend(["--current", args.current])
    if args.output:
        cli_args.extend(["--output", args.output])
    generate_drift(cli_args=cli_args if cli_args else None)
    print("✅ Drift Report Generated.")


def cmd_uplift(args: argparse.Namespace) -> None:
    """Run causal inference / prescriptive uplift modeling."""
    from scripts.train_uplift import run_uplift_experiment

    print("🎯 Executing Causal Uplift (CATE) Experiment...")
    run_uplift_experiment()
    print("✅ Causal Uplift Experiment Completed.")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run cross-validation benchmark and Nadeau-Bengio statistical tests."""
    from scripts.run_benchmarks import run_all_benchmarks

    print("🔬 Executing Tabular ML Benchmark Suite...")
    run_all_benchmarks()
    print("✅ Benchmark Suite Completed.")


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI serving server."""
    import uvicorn

    print(f"⚡ Launching FastAPI Server on {args.host}:{args.port}...")
    uvicorn.run(
        "src.serving.api:app", host=args.host, port=args.port, reload=args.reload
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="customer-churn-mlops",
        description="Command-Line Interface for Customer Churn MLOps Pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # train
    subparsers.add_parser("train", help="Run automated production training pipeline")

    # retrain
    parser_retrain = subparsers.add_parser(
        "retrain", help="Run drift-triggered retraining pipeline"
    )
    parser_retrain.add_argument(
        "--force",
        action="store_true",
        help="Force challenger retraining regardless of drift thresholds",
    )

    # monitor
    parser_monitor = subparsers.add_parser(
        "monitor", help="Generate drift monitoring report"
    )
    parser_monitor.add_argument(
        "--current", type=str, default=None, help="Path to production CSV"
    )
    parser_monitor.add_argument(
        "--output",
        type=str,
        default="results/metrics/drift_results.json",
        help="Output path",
    )

    # uplift
    subparsers.add_parser("uplift", help="Run causal uplift modeling experiment")

    # benchmark
    subparsers.add_parser(
        "benchmark", help="Run statistical benchmark comparing Tabular models"
    )

    # serve
    parser_serve = subparsers.add_parser("serve", help="Launch FastAPI prediction API")
    parser_serve.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host address"
    )
    parser_serve.add_argument("--port", type=int, default=8000, help="Port")
    parser_serve.add_argument(
        "--reload", action="store_true", help="Enable auto-reload"
    )

    return parser


def main() -> None:
    """Main CLI entrypoint."""
    parser = build_parser()

    # If no subcommand passed, default to train for backwards compatibility
    if len(sys.argv) == 1:
        print(
            "ℹ️  No subcommand specified. Defaulting to 'train'. "
            "(Run with --help for all commands)"
        )
        cmd_train(argparse.Namespace())
        return

    args = parser.parse_args()

    commands = {
        "train": cmd_train,
        "retrain": cmd_retrain,
        "monitor": cmd_monitor,
        "uplift": cmd_uplift,
        "benchmark": cmd_benchmark,
        "serve": cmd_serve,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n❌ Execution failed: {exc}", file=sys.stderr)
        sys.exit(1)
