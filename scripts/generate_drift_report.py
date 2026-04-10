#!/usr/bin/env python
"""
Generate a data drift monitoring report.

Compares the current production data (or the training data itself as a
self-comparison baseline) against the saved reference snapshot.

Usage:
    poetry run python scripts/generate_drift_report.py
    poetry run python scripts/generate_drift_report.py --current data/raw/new_data.csv

Or via Makefile:
    make drift-report
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path for `src.*` imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ingest import load_raw_data  # noqa: E402
from src.data.preprocess import preprocess  # noqa: E402
from src.data.validate import validate_raw_data  # noqa: E402
from src.features.feature_store import engineer_features  # noqa: E402
from src.monitoring.drift_detector import DriftDetector  # noqa: E402
from src.monitoring.reference_builder import load_reference  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    """Generate a drift monitoring report."""
    parser = argparse.ArgumentParser(description="Generate a drift monitoring report.")
    parser.add_argument(
        "--current",
        type=str,
        default=None,
        help=(
            "Path to current production data CSV. "
            "If not provided, uses the training dataset for a self-comparison."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/drift_report.json",
        help="Path to save the JSON drift report.",
    )
    args = parser.parse_args()

    # Load reference data
    logger.info("Loading reference dataset...")
    reference_df, reference_proba = load_reference()

    # Load current data
    if args.current is not None:
        logger.info("Loading current data from: %s", args.current)
        current_raw = load_raw_data(args.current)
    else:
        # Self-comparison: use the training data as "current" data.
        # PSI and chi-squared should report no drifting.
        logger.info(
            "No --current path provided. Using training data for self-comparison."
        )
        from src.data.ingest import load_for_training

        current_raw = load_for_training()

    # Apply same pipeline as reference
    current_validated = validate_raw_data(current_raw)
    current_clean = preprocess(current_validated)

    # Drop Churn column if present (inference data won't have it)
    if "Churn" in current_clean.columns:
        current_clean = current_clean.drop(columns=["Churn"])

    current_engineered = engineer_features(current_clean)

    # Run drift detection
    detector = DriftDetector(reference_df)
    report = detector.generate_report(
        current_df=current_engineered,
        reference_proba=reference_proba,
        current_proba=None,  # No model predictions on current data yet
    )

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_dict = report.to_dict()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, default=str)

    print(f"\n{'=' * 55}")
    print("  DRIFT MONITORING REPORT")
    print(f"{'=' * 55}")
    print(f"  Reference shape     : {report.reference_shape}")
    print(f"  Current shape       : {report.current_shape}")
    drift_flag = "YES ⚠️" if report.overall_drift_detected else "NO ✅"
    print(f"  Overall drift       : {drift_flag}")
    print(f"  Drifted features    : {len(report.drifted_features)}")
    if report.drifted_features:
        for feat in report.drifted_features:
            print(f"    - {feat}")
    print(f"  Prediction drift    : {report.prediction_drift_score or 'N/A'}")
    print(f"\n  Report saved to: {output_path.resolve()}")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
