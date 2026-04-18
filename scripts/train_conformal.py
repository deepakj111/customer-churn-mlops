"""
Train Conformal Prediction Model.

This script demonstrates uncertainty quantification via Conformal
Prediction. It trains the existing LightGBM pipeline, then fits a
MAPIE conformal wrapper on a held-out calibration set to produce
prediction sets with mathematically guaranteed coverage.

The key insight: the calibration set MUST be unseen by the base model.
Using training data would invalidate the coverage guarantee.

Output:
    - models/conformal/conformal_model.joblib
    - models/conformal/conformal_meta.json
    - reports/conformal_report.json (coverage & efficiency metrics)
"""

import json
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from src.data.ingest import load_for_training
from src.data.preprocess import run_preprocessing
from src.data.validate import validate_raw_data
from src.models.conformal import ConformalChurnPredictor
from src.models.pipeline import build_pipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_conformal_experiment() -> None:
    """Train conformal prediction model and evaluate coverage."""
    logger.info("Starting Conformal Prediction experiment.")

    # 1. Load and prepare data
    raw_df = load_for_training()
    validated_df = validate_raw_data(raw_df)
    X, y = run_preprocessing(validated_df)

    # 2. Three-way split: train (60%) / calibration (20%) / test (20%)
    # The calibration set is CRITICAL — it must be completely unseen
    # by the base model. This provides the coverage guarantee.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=42,
        stratify=y,
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    logger.info(
        "Split — train: %d, calibration: %d, test: %d",
        len(X_train),
        len(X_cal),
        len(X_test),
    )

    # 3. Train the base pipeline (same as production)
    logger.info("Training base LightGBM pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # 3.5 Calibrate probabilities on half the calibration data
    logger.info("Calibrating probabilities (isotonic)...")
    calibrated = CalibratedClassifierCV(
        estimator=pipeline, method="isotonic", cv="prefit"
    )
    cal_half = len(X_cal) // 2
    X_cal_iso = X_cal.iloc[:cal_half]
    y_cal_iso = y_cal.iloc[:cal_half]
    X_cal_conf = X_cal.iloc[cal_half:]
    y_cal_conf = y_cal.iloc[cal_half:]
    calibrated.fit(X_cal_iso, y_cal_iso)

    # 4. Fit conformal predictor on remaining calibration data
    conformal = ConformalChurnPredictor(
        confidence_levels=[0.90, 0.95],
    )
    conformal.calibrate(calibrated, X_cal_conf, y_cal_conf)

    # 5. Evaluate on held-out test set
    logger.info(
        "Evaluating coverage on test set (%d samples)...",
        len(X_test),
    )
    results = conformal.predict(X_test)

    report: dict = {"test_set_size": len(X_test)}

    for confidence in [0.90, 0.95]:
        data = results[str(confidence)]
        pred_sets = np.array(data["prediction_sets"])
        set_sizes = np.array(data["set_sizes"])

        # Coverage: fraction where true label is in the set
        y_test_arr = y_test.values
        covered = np.array(
            [pred_sets[i, int(y_test_arr[i])] for i in range(len(y_test_arr))]
        )
        empirical_coverage = float(covered.mean())

        # Efficiency: average prediction set size (smaller = better)
        avg_set_size = float(set_sizes.mean())

        # Breakdown
        empty_sets = int((set_sizes == 0).sum())
        singleton_sets = int((set_sizes == 1).sum())
        ambiguous_sets = int((set_sizes == 2).sum())

        key = f"confidence_{int(confidence * 100)}"
        report[key] = {
            "target_coverage": confidence,
            "empirical_coverage": round(empirical_coverage, 4),
            "coverage_gap": round(empirical_coverage - confidence, 4),
            "avg_set_size": round(avg_set_size, 4),
            "empty_sets": empty_sets,
            "singleton_sets": singleton_sets,
            "ambiguous_sets": ambiguous_sets,
            "singleton_rate": round(singleton_sets / len(X_test), 4),
        }

        logger.info(
            "[%s] Coverage: %.2f%% (target: %.0f%%), "
            "Avg set size: %.2f, Singletons: %.1f%%",
            key,
            empirical_coverage * 100,
            confidence * 100,
            avg_set_size,
            singleton_sets / len(X_test) * 100,
        )

    # 6. Save artifacts
    conformal.save()

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    with open(reports_dir / "conformal_report.json", "w") as f:
        json.dump(report, f, indent=4)

    logger.info(
        "Conformal prediction experiment complete. "
        "Artifacts saved to models/conformal/"
    )


if __name__ == "__main__":
    run_conformal_experiment()
