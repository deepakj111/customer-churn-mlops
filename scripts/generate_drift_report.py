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


def main(cli_args: list[str] | None = None) -> None:
    """Generate a drift monitoring report.

    Args:
        cli_args: Optional list of CLI arguments. If None, reads from sys.argv.
                  This allows callers (e.g., main.py) to pass arguments directly
                  without the brittle sys.argv manipulation anti-pattern.
    """
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
        default="results/metrics/drift_results.json",
        help="Path to save the JSON drift report.",
    )
    args = parser.parse_args(cli_args)

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

    # Compute model predictions on current data if champion pipeline exists
    current_proba = None
    champion_path = Path("models/champion_pipeline.joblib")
    if champion_path.exists():
        try:
            import joblib

            pipeline = joblib.load(champion_path)
            current_proba = pipeline.predict_proba(current_clean)[:, 1]
            logger.info(
                "Computed current prediction probabilities on %d samples",
                len(current_proba),
            )
        except Exception as e:
            logger.warning("Could not compute current model probabilities: %s", e)

    # Run drift detection
    detector = DriftDetector(reference_df)
    report = detector.generate_report(
        current_df=current_engineered,
        reference_proba=reference_proba,
        current_proba=current_proba,
    )

    # Save machine-readable JSON metrics to results/metrics/
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_dict = report.to_dict()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, default=str)

    # Generate Markdown monitoring report in results/
    md_report_path = Path("results/DRIFT_MONITORING_REPORT.md")
    drift_status_badge = (
        "🚨 **DRIFT DETECTED**"
        if report.overall_drift_detected
        else "✅ **NO DRIFT DETECTED (STABLE)**"
    )
    drifted_list = (
        "\n".join([f"- `{feat}`" for feat in report.drifted_features])
        if report.drifted_features
        else "_None. All feature distributions align with baseline._"
    )

    # Top features by PSI table
    psi_rows = []
    if report.numerical_drift:
        sorted_feats = sorted(
            report.numerical_drift, key=lambda x: x.drift_score, reverse=True
        )[:10]
        for item in sorted_feats:
            status = "🔴 Drifted" if item.is_drifted else "🟢 Stable"
            psi_rows.append(
                f"| `{item.feature_name}` | {item.drift_score:.4f} | {status} |"
            )
    psi_table = (
        "\n".join(psi_rows)
        if psi_rows
        else "| _No individual metrics recorded_ | - | - |"
    )

    pred_drift_display = (
        f"`{report.prediction_drift_score:.4f}`"
        if report.prediction_drift_score is not None
        else "`0.0000`"
    )

    md_content = f"""# Production Drift & Population Stability Monitoring Report

**Generated:** {report.timestamp}
**Evaluation Scope:** Continuous Distribution Drift Detection against Baseline Snapshot

---

## 1. Executive Summary

| Monitoring Dimension | Value | Operational Status |
|---|---|---|
| **Overall Drift Status** | {report.overall_drift_detected} | {drift_status_badge} |
| **Reference Snapshot Shape** | `{report.reference_shape}` | Baseline reference |
| **Current Production Shape** | `{report.current_shape}` | Production stream |
| **Drifted Features** | `{len(report.drifted_features)}` | Alert threshold $\\ge 3$ |
| **Prediction Drift Score (Wasserstein)** | {pred_drift_display} | Threshold: 0.10 |

---

## 2. Drift Detection Methodology & Thresholds

- **Numerical Features (Population Stability Index - PSI)**:
  - $\\text{{PSI}} < 0.10$: Minimal shift, distribution stable.
  - $0.10 \\le \\text{{PSI}} < 0.20$: Moderate shift, warning trigger.
  - $\\text{{PSI}} \\ge 0.20$: Significant shift, auto-retrain trigger.
- **Categorical Features**:
  - Two-sample Chi-Square goodness-of-fit test (Significance level $\\alpha = 0.05$).
- **Prediction Drift**:
  - Wasserstein distance metric on calibrated model output probabilities.

---

## 3. Top Feature Stability Analysis

| Feature Name | PSI Score | Status |
|---|---|---|
{psi_table}

---

## 4. Drifted Features List

{drifted_list}

---

## 5. Automated Operational Recommendations

1. **Retraining Action**: If overall drift is flagged, execute
   `poetry run python main.py retrain` to initiate champion-challenger tournament.
2. **Monitoring Dashboards**: Real-time feature PSI decile distribution is available
   in the Streamlit monitoring tab (`make dashboard`).
"""

    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write(md_content)

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
    print(f"\n  JSON Results saved to: {output_path.resolve()}")
    print(f"  Markdown Report at:   {md_report_path.resolve()}")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
