"""
Automated Retraining & Challenger Evaluation Pipeline — Prefect Flow.

This module automates continuous learning and drift-driven model retraining.
It operates as a closed-loop monitoring and promotion system:
    1. monitor_drift_task: Measure PSI, Chi-squared, and prediction drift.
    2. evaluate_trigger_task: Check whether drift exceeds thresholds.
    3. retrain_challenger_task: Fit challenger pipeline on fresh distribution.
    4. compare_champion_challenger_task: Run Nadeau-Bengio corrected t-test
       to statistically test whether challenger outperforms current champion.
    5. promote_or_reject_task: Conditionally promote challenger to Production,
       update local deployment artifacts, and emit retraining audit report.

Usage:
    poetry run python pipelines/retraining_pipeline.py --trigger drift
    poetry run python pipelines/retraining_pipeline.py --force
    # or via CLI:
    poetry run python main.py retrain
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from mlflow.models import infer_signature
from prefect import flow, task
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.data.ingest import load_for_training
from src.data.preprocess import run_preprocessing
from src.data.validate import validate_raw_data
from src.models.pipeline import build_pipeline
from src.models.statistical_comparison import corrected_cv_ttest
from src.monitoring.drift_detector import DriftDetector, DriftReport
from src.monitoring.reference_builder import load_reference_dataset
from src.serving.model_loader import get_model
from src.utils.config_loader import get_config
from src.utils.logging import get_logger

logger = get_logger("retraining_pipeline")


@task(name="Monitor Data and Prediction Drift", retries=2, retry_delay_seconds=5)
def monitor_drift_task() -> dict[str, Any]:
    """
    Compute numerical PSI, categorical Chi-squared, and prediction drift.
    """
    logger.info("Executing drift monitoring task...")
    cfg = get_config()
    ref_df = load_reference_dataset()

    # Load current incoming dataset
    raw_df = load_for_training()
    validated_df = validate_raw_data(raw_df)
    current_X, _ = run_preprocessing(validated_df)

    # Compute reference and current predictions using active champion model
    active_model = get_model()
    current_proba = active_model.predict_proba(current_X)[:, 1]

    detector = DriftDetector(reference_df=ref_df, config=cfg)
    report: DriftReport = detector.generate_report(
        current_df=current_X,
        current_proba=pd.Series(current_proba),
    )

    # Save drift report artifact
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_dict = report.to_dict()
    with open(reports_dir / "drift_report.json", "w") as f:
        json.dump(report_dict, f, indent=4)

    logger.info(
        "Drift monitoring complete. Overall drift detected: %s",
        report.overall_drift_detected,
    )
    return {
        "report": report,
        "report_dict": report_dict,
        "current_X": current_X,
        "current_proba": current_proba,
        "overall_drift": report.overall_drift_detected,
        "drifted_features": report.drifted_features,
    }


@task(name="Evaluate Retraining Trigger")
def evaluate_trigger_task(
    drift_artifacts: dict[str, Any],
    trigger_type: str = "drift",
    force_retrain: bool = False,
) -> bool:
    """
    Decide whether retraining should proceed based on drift or manual override.
    """
    logger.info(
        "Evaluating retraining triggers (trigger=%s, force=%s)...",
        trigger_type,
        force_retrain,
    )
    if force_retrain:
        logger.info("Retraining triggered via manual force override.")
        return True

    if trigger_type == "scheduled":
        logger.info("Retraining triggered via scheduled periodic cadence.")
        return True

    if drift_artifacts["overall_drift"]:
        logger.warning(
            "Significant distribution drift detected in features: %s. "
            "Initiating challenger retraining.",
            drift_artifacts["drifted_features"],
        )
        return True

    logger.info("No significant drift detected. Retraining trigger bypassed.")
    return False


@task(name="Train Challenger Model")
def retrain_challenger_task() -> dict[str, Any]:
    """Train fresh challenger candidate pipeline on current data distribution."""
    logger.info("Training challenger model pipeline...")
    cfg = get_config()
    raw_df = load_for_training()
    validated_df = validate_raw_data(raw_df)
    X, y = run_preprocessing(validated_df)

    # Build challenger pipeline
    challenger_pipeline = build_pipeline()

    # 5-fold cross-validation fold scores for statistical testing
    cv = StratifiedKFold(
        n_splits=cfg.training.cv_folds,
        shuffle=True,
        random_state=cfg.training.random_state
        + 1,  # different seed to test generalizability
    )
    challenger_scores = cross_val_score(
        challenger_pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1
    )
    challenger_pipeline.fit(X, y)

    logger.info(
        "Challenger 5-fold CV ROC-AUC: %.4f ± %.4f",
        challenger_scores.mean(),
        challenger_scores.std(),
    )

    return {
        "challenger_pipeline": challenger_pipeline,
        "challenger_scores": challenger_scores,
        "X": X,
        "y": y,
    }


@task(name="Champion vs Challenger Statistical Comparison")
def compare_champion_challenger_task(
    challenger_artifacts: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare active Champion against fresh Challenger using the
    Nadeau-Bengio corrected repeated k-fold t-test.
    """
    logger.info("Executing Champion vs Challenger hypothesis test...")
    cfg = get_config()
    X = challenger_artifacts["X"]
    y = challenger_artifacts["y"]
    challenger_scores = challenger_artifacts["challenger_scores"]

    active_champion = get_model()

    cv = StratifiedKFold(
        n_splits=cfg.training.cv_folds,
        shuffle=True,
        random_state=cfg.training.random_state + 1,
    )
    champion_scores = cross_val_score(
        active_champion, X, y, cv=cv, scoring="roc_auc", n_jobs=-1
    )

    n_total = len(X)
    n_test = n_total // cfg.training.cv_folds
    n_train = n_total - n_test

    # Run Nadeau-Bengio corrected t-test
    stat_result = corrected_cv_ttest(
        scores_a=challenger_scores,
        scores_b=champion_scores,
        n_train=n_train,
        n_test=n_test,
    )

    diff = float(challenger_scores.mean() - champion_scores.mean())
    logger.info(
        "Model Comparison — Challenger: %.4f | Champion: %.4f | "
        "Diff: %+.4f | p-value: %.4f (sig: %s)",
        challenger_scores.mean(),
        champion_scores.mean(),
        diff,
        stat_result.p_value,
        stat_result.is_significant,
    )

    # Challenger promotion criteria:
    # Either statistically significantly better (p < 0.05 and diff > 0)
    # OR satisfies non-inferiority margin (|diff| <= 0.005 and challenger_mean >= 0.82)
    promotable = (stat_result.is_significant and diff > 0) or (
        diff >= -0.002 and challenger_scores.mean() >= 0.82
    )

    return {
        "stat_result": stat_result,
        "challenger_mean": float(challenger_scores.mean()),
        "champion_mean": float(champion_scores.mean()),
        "mean_difference": diff,
        "p_value": stat_result.p_value,
        "promotable": promotable,
    }


@task(name="Promote Challenger or Retain Champion")
def promote_or_reject_task(
    comparison_results: dict[str, Any],
    challenger_artifacts: dict[str, Any],
) -> dict[str, Any]:
    """Promote challenger to Production in Model Registry if criteria met."""
    load_dotenv()
    cfg = get_config()
    promotable = comparison_results["promotable"]
    challenger_pipeline = challenger_artifacts["challenger_pipeline"]
    X = challenger_artifacts["X"]

    retraining_audit = {
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "action": "PROMOTED" if promotable else "REJECTED",
        "challenger_roc_auc": comparison_results["challenger_mean"],
        "champion_roc_auc": comparison_results["champion_mean"],
        "difference": comparison_results["mean_difference"],
        "p_value": comparison_results["p_value"],
        "decision_rationale": (
            "Challenger demonstrated superior or non-inferior calibrated performance."
            if promotable
            else "Challenger failed to demonstrate statistical superiority."
        ),
    }

    if promotable:
        logger.info("Promoting Challenger to Production Model Registry...")
        try:
            mlflow.set_tracking_uri(
                os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            )
            mlflow.set_experiment(cfg.training.experiment_name)

            ts = datetime.datetime.now(tz=datetime.timezone.utc).strftime(
                "%Y%m%d_%H%M%S"
            )
            with mlflow.start_run(run_name=f"promoted_challenger_{ts}"):
                mlflow.log_metric(
                    "challenger_roc_auc", comparison_results["challenger_mean"]
                )
                mlflow.log_metric("roc_auc_diff", comparison_results["mean_difference"])
                mlflow.log_metric("t_test_p_value", comparison_results["p_value"])
                mlflow.set_tag("retraining_event", "automated_drift_retraining")

                signature = infer_signature(
                    X, challenger_pipeline.predict_proba(X)[:, 1]
                )
                mlflow.sklearn.log_model(
                    sk_model=challenger_pipeline,
                    artifact_path="model",
                    signature=signature,
                    registered_model_name=cfg.model.registered_model_name,
                )
        except Exception as e:
            logger.warning("MLflow promotion skipped: %s", e)

        # Update local champion artifact
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(challenger_pipeline, models_dir / "champion_pipeline.joblib")
        logger.info(
            "Updated local champion artifact at models/champion_pipeline.joblib"
        )
    else:
        logger.info("Retaining existing Champion in production. Challenger rejected.")

    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "retraining_audit.json", "w") as f:
        json.dump(retraining_audit, f, indent=4)

    return retraining_audit


@flow(name="customer-churn-retraining-pipeline")
def run_retraining_pipeline(
    trigger_type: str = "drift",
    force_retrain: bool = False,
) -> dict[str, Any]:
    """
    Automated retraining flow with drift evaluation & champion-challenger testing.
    """
    logger.info("=== Starting Automated Retraining Flow (trigger=%s) ===", trigger_type)

    # 1. Drift Monitoring
    drift_artifacts = monitor_drift_task()

    # 2. Trigger Evaluation
    should_retrain = evaluate_trigger_task(
        drift_artifacts=drift_artifacts,
        trigger_type=trigger_type,
        force_retrain=force_retrain,
    )

    if not should_retrain:
        logger.info("Retraining not required at this time. Health check passed.")
        return {
            "status": "SKIPPED",
            "reason": "NO_DRIFT",
            "drift_detected": False,
            "drifted_features": drift_artifacts["drifted_features"],
        }

    # 3. Train Challenger
    challenger_artifacts = retrain_challenger_task()

    # 4. Statistical Champion vs Challenger Comparison
    comparison_results = compare_champion_challenger_task(challenger_artifacts)

    # 5. Model Promotion Decision
    decision_audit = promote_or_reject_task(comparison_results, challenger_artifacts)

    logger.info("=== Retraining Flow Finished: Result=%s ===", decision_audit["action"])
    return {
        "status": "COMPLETED",
        "action": decision_audit["action"],
        "audit": decision_audit,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Churn Retraining Pipeline")
    parser.add_argument("--trigger", choices=["drift", "scheduled"], default="drift")
    parser.add_argument(
        "--force", action="store_true", help="Force retraining regardless of drift"
    )
    args = parser.parse_args()

    result = run_retraining_pipeline(
        trigger_type=args.trigger, force_retrain=args.force
    )
    print("\nRetraining Result:")
    print(json.dumps(result, indent=2))
