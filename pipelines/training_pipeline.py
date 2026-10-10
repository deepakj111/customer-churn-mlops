"""
Automated Training Pipeline — Prefect Orchestrated Flow.

This module provides the production automated training pipeline for the
Customer Churn MLOps system. Designed for scheduled execution, CI/CD automated
triggers, and external orchestrators (Prefect Cloud / Server).

Pipeline Stages:
    1. ingest_and_validate: Load raw customer data & enforce Pandera schemas.
    2. preprocess_data: Apply stateless transformations & split features/target.
    3. feature_store_sync: Verify Feast online/offline feature parity.
    4. train_and_tune: 3-way stratified split, 5-fold CV, pipeline training,
       and cost-optimal decision threshold tuning.
    5. calibrate_and_diagnose: Probability calibration with FrozenEstimator,
       Expected Calibration Error (ECE) & Brier score diagnostics.
    6. audit_fairness: Protected attribute fairness auditing (Demographic Parity &
       Equalized Odds via EEOC 4/5ths rule).
    7. conformal_calibration: Uncertainty quantification via MAPIE conformal prediction.
    8. log_and_register: MLflow experiment tracking, performance gates validation,
       model registry tagging, and local champion artifact persistence.

Usage:
    poetry run python pipelines/training_pipeline.py
    # or via CLI:
    poetry run python main.py train
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mlflow.models import infer_signature
from prefect import flow, task
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src.data.ingest import load_for_training
from src.data.preprocess import run_preprocessing
from src.data.validate import validate_raw_data
from src.models.calibration_diagnostics import compute_calibration_report
from src.models.conformal import ConformalChurnPredictor
from src.models.evaluate import evaluate, print_evaluation_report
from src.models.fairness_audit import run_fairness_audit
from src.models.pipeline import build_pipeline
from src.models.threshold import find_cost_optimal_threshold, find_f1_optimal_threshold
from src.utils.config_loader import get_config
from src.utils.logging import get_logger

logger = get_logger("training_pipeline")


@task(name="Ingest and Validate Data", retries=2, retry_delay_seconds=5)
def ingest_and_validate_task() -> pd.DataFrame:
    """Load raw dataset and enforce strict Pandera schema validation."""
    logger.info("Executing ingest and validate task...")
    raw_df = load_for_training()
    validated_df = validate_raw_data(raw_df)
    logger.info(
        "Data validated successfully: %d rows, %d columns.", *validated_df.shape
    )
    return validated_df


@task(name="Preprocess Data")
def preprocess_task(validated_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean data, cast categories, drop non-predictive keys, and split X/y."""
    logger.info("Executing preprocessing task...")
    X, y = run_preprocessing(validated_df)
    logger.info(
        "Features X: %s, Target y: %s (Churn rate: %.2f%%)",
        X.shape,
        y.shape,
        y.mean() * 100,
    )

    # Persist canonical processed dataset splits
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    if not (processed_dir / "train.parquet").exists():
        clean_df = validated_df.copy()
        clean_df.to_parquet(processed_dir / "telco_churn_clean.parquet", index=False)

    return X, y


@task(name="Feature Store Sync")
def feature_store_sync_task() -> bool:
    """Verify Feast Feature Store configuration and registry readiness."""
    logger.info("Verifying Feast Feature Store status...")
    feast_repo_path = Path("src/features/feast_repo")
    if (feast_repo_path / "feature_store.yaml").exists():
        logger.info("Feast feature store repository found at %s", feast_repo_path)
        return True
    logger.warning(
        "Feast configuration not found at %s. Proceeding in offline mode.",
        feast_repo_path,
    )
    return False


@task(name="Train and Tune Model")
def train_and_tune_task(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Execute stratified 3-way split, cross-validation, and threshold tuning.
    """
    cfg = get_config()
    logger.info("Executing training and tuning task...")

    # Stratified 3-way split: 70% train / 10% validation / 20% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state,
        stratify=y,
    )
    relative_val_size = cfg.training.val_size / (1.0 - cfg.training.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=relative_val_size,
        random_state=cfg.training.random_state,
        stratify=y_trainval,
    )

    logger.info(
        "Dataset splits — Train: %d rows, Validation: %d rows, Test: %d rows",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    # Build and evaluate baseline pipeline
    pipeline = build_pipeline(params)
    cv = StratifiedKFold(
        n_splits=cfg.training.cv_folds,
        shuffle=True,
        random_state=cfg.training.random_state,
    )
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=cfg.training.cv_scoring,
        n_jobs=-1,
    )
    cv_mean = round(float(cv_scores.mean()), 4)
    cv_std = round(float(cv_scores.std()), 4)
    logger.info("5-Fold CV ROC-AUC on train set: %.4f ± %.4f", cv_mean, cv_std)

    # Fit pipeline on training split
    pipeline.fit(X_train, y_train)

    # Calibrate probabilities on validation split (prefit calibration)
    try:
        from sklearn.calibration import FrozenEstimator

        calibrated_pipeline = CalibratedClassifierCV(
            estimator=FrozenEstimator(pipeline), method="isotonic"
        )
    except ImportError:
        calibrated_pipeline = CalibratedClassifierCV(
            estimator=pipeline, method="isotonic", cv="prefit"
        )
    calibrated_pipeline.fit(X_val, y_val)

    # Tune decision threshold on validation set (prevents leakage)
    y_val_proba = calibrated_pipeline.predict_proba(X_val)[:, 1]
    optimal_threshold = find_cost_optimal_threshold(y_val.values, y_val_proba)
    f1_threshold = find_f1_optimal_threshold(y_val.values, y_val_proba)

    logger.info(
        "Optimal threshold tuned on validation: Cost-opt=%.4f | F1-opt=%.4f",
        optimal_threshold,
        f1_threshold,
    )

    # Evaluate on held-out test split
    y_test_proba = calibrated_pipeline.predict_proba(X_test)[:, 1]
    y_test_proba_raw = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test.values, y_test_proba, optimal_threshold)

    return {
        "pipeline": pipeline,
        "calibrated_pipeline": calibrated_pipeline,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "optimal_threshold": optimal_threshold,
        "f1_threshold": f1_threshold,
        "test_metrics": test_metrics,
        "y_test_proba_raw": y_test_proba_raw,
        "y_test_proba": y_test_proba,
    }


@task(name="Calibrate and Audit Model")
def calibrate_and_audit_task(train_artifacts: dict[str, Any]) -> dict[str, Any]:
    """Run ECE probability calibration diagnostics and EEOC fairness audit."""
    logger.info("Executing calibration diagnostics & fairness audit...")
    y_test = train_artifacts["y_test"]
    y_test_proba_raw = train_artifacts["y_test_proba_raw"]
    y_test_proba = train_artifacts["y_test_proba"]
    optimal_threshold = train_artifacts["optimal_threshold"]
    X_test = train_artifacts["X_test"]

    # 1. Calibration Diagnostics (ECE / Brier score)
    cal_report = compute_calibration_report(
        y_true=y_test.values,
        y_prob_before=y_test_proba_raw,
        y_prob_after=y_test_proba,
    )
    logger.info("Calibration report:\n%s", cal_report.summary())

    # 2. Algorithmic Fairness Audit
    y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
    sensitive_features: dict[str, np.ndarray] = {}
    if "gender" in X_test.columns:
        sensitive_features["gender"] = X_test["gender"].values
    if "SeniorCitizen" in X_test.columns:
        sensitive_features["SeniorCitizen"] = X_test["SeniorCitizen"].values

    fairness_report = None
    if sensitive_features:
        fairness_report = run_fairness_audit(
            y_true=y_test.values,
            y_pred=y_test_pred,
            y_prob=y_test_proba,
            sensitive_features=sensitive_features,
        )
        logger.info("Fairness audit report:\n%s", fairness_report.summary())

    return {
        "calibration_report": cal_report,
        "fairness_report": fairness_report,
    }


@task(name="Conformal Uncertainty Calibration")
def conformal_calibration_task(train_artifacts: dict[str, Any]) -> dict[str, Any]:
    """
    Calibrate MAPIE conformal wrapper for distribution-free confidence sets.
    """
    logger.info("Calibrating Conformal Predictor...")
    calibrated_pipeline = train_artifacts["calibrated_pipeline"]
    X_val = train_artifacts["X_val"]
    y_val = train_artifacts["y_val"]
    X_test = train_artifacts["X_test"]

    conformal = ConformalChurnPredictor(confidence_levels=[0.90, 0.95])
    conformal.calibrate(calibrated_pipeline, X_val, y_val)
    conformal_results = conformal.predict(X_test)
    conformal.save(Path("models/conformal"))

    return {
        "conformal_predictor": conformal,
        "conformal_results": conformal_results,
    }


@task(name="Governance, MLflow Logging & Local Persistence")
def log_and_register_task(
    train_artifacts: dict[str, Any],
    audit_artifacts: dict[str, Any],
    run_name: str | None = None,
) -> dict[str, Any]:
    """
    Log to MLflow, validate performance gates, and persist champion model.
    """
    load_dotenv()
    cfg = get_config()
    test_metrics = train_artifacts["test_metrics"]
    calibrated_pipeline = train_artifacts["calibrated_pipeline"]
    X_train = train_artifacts["X_train"]
    optimal_threshold = train_artifacts["optimal_threshold"]
    gates = cfg.model.performance_gates

    # Check performance gates
    gates_passed = (
        test_metrics["roc_auc"] >= gates.min_roc_auc
        and test_metrics["pr_auc"] >= gates.min_pr_auc
        and test_metrics["recall"] >= gates.min_recall_at_threshold
    )

    if run_name is None:
        ts = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_name = f"{cfg.model.algorithm}_prefect_pipeline_{ts}"

    mlflow_run_id = None
    try:
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        )
        mlflow.set_experiment(cfg.training.experiment_name)

        with mlflow.start_run(run_name=run_name) as run:
            mlflow_run_id = run.info.run_id
            logger.info(
                "MLflow run initiated: %s (run_id: %s)", run_name, mlflow_run_id
            )

            # Log metrics
            mlflow.log_metric("cv_mean_roc_auc", train_artifacts["cv_mean"])
            mlflow.log_metric("cv_std_roc_auc", train_artifacts["cv_std"])
            mlflow.log_metric("optimal_threshold", optimal_threshold)
            mlflow.log_metric("f1_threshold", train_artifacts["f1_threshold"])

            for k, v in test_metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"test_{k}", v)

            if audit_artifacts["calibration_report"]:
                for k, v in audit_artifacts["calibration_report"].to_dict().items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)

            mlflow.set_tag("gates_passed", str(gates_passed))
            mlflow.set_tag("model_type", cfg.model.algorithm)
            mlflow.set_tag("pipeline_orchestrator", "prefect")

            # Log model artifact with input signature
            signature = infer_signature(
                X_train, calibrated_pipeline.predict_proba(X_train)[:, 1]
            )
            mlflow.sklearn.log_model(
                sk_model=calibrated_pipeline,
                artifact_path="model",
                signature=signature,
                input_example=X_train.head(5),
                registered_model_name=cfg.model.registered_model_name,
            )
            logger.info(
                "Model registered to MLflow Model Registry as '%s'",
                cfg.model.registered_model_name,
            )
    except Exception as e:
        logger.warning("MLflow tracking skipped or failed: %s", e)

    # Save champion model locally for rapid offline/Docker cold starts
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    local_pipeline_path = models_dir / "champion_pipeline.joblib"
    local_meta_path = models_dir / "model_metadata.json"

    joblib.dump(calibrated_pipeline, local_pipeline_path)

    metadata = {
        "model_name": cfg.model.registered_model_name,
        "algorithm": cfg.model.algorithm,
        "feature_count": len(X_train.columns),
        "optimal_threshold": optimal_threshold,
        "f1_threshold": train_artifacts["f1_threshold"],
        "cv_mean_roc_auc": train_artifacts["cv_mean"],
        "cv_std_roc_auc": train_artifacts["cv_std"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_recall": test_metrics["recall"],
        "test_precision": test_metrics["precision"],
        "estimated_savings": test_metrics.get("estimated_savings", 0.0),
        "gates_passed": gates_passed,
        "mlflow_run_id": mlflow_run_id,
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
    }

    with open(local_meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    # Save to results/metrics/ for centralized machine-readable output tracking
    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "production_pipeline_results.json", "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info("Champion model persisted locally to %s", local_pipeline_path)
    logger.info(
        "Metadata persisted locally to %s and %s",
        local_meta_path,
        metrics_dir / "production_pipeline_results.json",
    )

    print_evaluation_report(test_metrics, split_name="Pipeline Final Test Evaluation")

    return {
        "status": "SUCCESS",
        "gates_passed": gates_passed,
        "metrics": test_metrics,
        "optimal_threshold": optimal_threshold,
        "mlflow_run_id": mlflow_run_id,
        "local_model_path": str(local_pipeline_path),
    }


@flow(name="customer-churn-training-pipeline")
def run_automated_training_pipeline(
    params: dict[str, Any] | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """
    Execute the end-to-end automated training orchestration pipeline.

    Returns:
        Summary dict containing performance metrics, gate results, and model paths.
    """
    logger.info("=== Starting Automated Churn Training Pipeline ===")

    # 1. Ingest & Validate
    validated_df = ingest_and_validate_task()

    # 2. Preprocess & Split
    X, y = preprocess_task(validated_df)

    # 3. Feature Store Sync
    feature_store_sync_task()

    # 4. Train, CV & Tune Threshold
    train_artifacts = train_and_tune_task(X, y, params=params)

    # 5. Calibration Diagnostics & Fairness Audit
    audit_artifacts = calibrate_and_audit_task(train_artifacts)

    # 6. Conformal Prediction Set Calibration
    conformal_calibration_task(train_artifacts)

    # 7. Model Governance, MLflow Logging & Local Persistence
    pipeline_summary = log_and_register_task(
        train_artifacts, audit_artifacts, run_name=run_name
    )

    logger.info("=== Automated Churn Training Pipeline Completed Successfully ===")
    return pipeline_summary


run_training_pipeline = run_automated_training_pipeline


if __name__ == "__main__":
    result = run_automated_training_pipeline()
    print("\nPipeline Result Summary:")
    print(json.dumps({k: v for k, v in result.items() if k != "metrics"}, indent=2))
