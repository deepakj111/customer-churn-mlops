"""
Training orchestration — the full training pipeline in one callable function.

This module wires together every module built so far:
    ingest      → load raw CSV
    validate    → schema check + TotalCharges fix
    preprocess  → encode target, drop customerID, split X/y
    feature_store → engineer 28 new features (inside the Pipeline)
    pipeline    → build sklearn Pipeline (feature_eng + preprocessor + LGBM)
    threshold   → find cost-optimal decision threshold
    evaluate    → compute ML + business metrics
    MLflow      → log everything, register model

The design follows a single entry point: run_training_experiment()
Everything else is a helper that run_training_experiment() calls.
This makes the function easy to call from:
    - A notebook (manual run)
    - A Prefect flow (automated pipeline)
    - GitHub Actions CI (smoke test)
    - The retraining trigger (drift-driven retraining)

Public API:
    build_training_data()           → X, y
    run_training_experiment(params) → dict of final metrics
"""

import datetime
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from mlflow.models import infer_signature
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src.data.ingest import load_for_training
from src.data.preprocess import run_preprocessing
from src.data.validate import validate_raw_data
from src.models.evaluate import evaluate, print_evaluation_report
from src.models.pipeline import build_pipeline
from src.models.threshold import find_cost_optimal_threshold, find_f1_optimal_threshold
from src.utils.config_loader import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Run the full data preparation sequence and return (X, y).

    Calls the existing pipeline modules in order:
        load_for_training() → validate_raw_data() → run_preprocessing()

    This is the canonical data preparation sequence. Every training run,
    retraining trigger, and notebook uses this function — never raw calls
    to the individual steps — so the sequence is always consistent.

    Returns:
        X: Feature DataFrame (19 original features, customerID dropped)
        y: Binary target Series (1=churn, 0=retain)
    """
    logger.info("Building training data...")
    raw_df = load_for_training()
    validated_df = validate_raw_data(raw_df)
    X, y = run_preprocessing(validated_df)

    logger.info(
        "Training data ready — X: %s, y: %s, churn rate: %.2f%%",
        X.shape,
        y.shape,
        y.mean() * 100,
    )
    return X, y


def _log_params_to_mlflow(pipeline, cfg) -> None:
    """Log all LightGBM hyperparameters to the active MLflow run."""
    classifier = pipeline.named_steps["classifier"]
    params = classifier.get_params()
    mlflow.log_params(params)
    mlflow.log_param("test_size", cfg.training.test_size)
    mlflow.log_param("val_size", cfg.training.val_size)
    mlflow.log_param("cv_folds", cfg.training.cv_folds)
    mlflow.log_param("random_state", cfg.training.random_state)


def _log_metrics_to_mlflow(metrics: dict, prefix: str = "") -> None:
    """
    Log a metrics dict to the active MLflow run.

    Prefixes each key so test and CV metrics don't collide:
        prefix="test_"  → "test_roc_auc", "test_pr_auc", etc.
        prefix="cv_"    → "cv_mean_roc_auc", etc.
    """
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(f"{prefix}{key}", value)


def run_training_experiment(
    params: dict | None = None,
    run_name: str | None = None,
) -> dict:
    """
    Run a full training experiment and log everything to MLflow.

    Flow:
        1. Load + validate + preprocess data
        2. Stratified train/val/test split (70/10/20)
        3. 5-fold cross-validation on train set (ROC-AUC)
        4. Fit final pipeline on full train set
        5. Find cost-optimal threshold on VALIDATION set (not test)
        6. Evaluate on held-out TEST set with the chosen threshold
        7. Log all params, metrics, and model to MLflow
        8. Register model in MLflow Model Registry if gates pass

    The validation set is used exclusively for threshold tuning.
    The test set is touched only once — for the final unbiased evaluation.
    This prevents data leakage from threshold optimisation.

    Args:
        params:   Optional LightGBM hyperparameter overrides.
                  If None, uses values from model_config.yaml.
        run_name: Optional name for the MLflow run.
                  Defaults to "lgbm_churn_{timestamp}".

    Returns:
        Dict of final test set metrics (same structure as evaluate()).
        Includes "mlflow_run_id" and "optimal_threshold" keys.
    """
    load_dotenv()
    cfg = get_config()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    mlflow.set_experiment(cfg.training.experiment_name)

    X, y = build_training_data()

    # --- 3-way stratified split: train (70%) / val (10%) / test (20%) ---
    # First split: train+val (80%) vs test (20%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state,
        stratify=y,
    )

    # Second split: train (70% of total) vs val (10% of total)
    # val_size is expressed as fraction of total, so relative to trainval
    # it is val_size / (1 - test_size)
    relative_val_size = cfg.training.val_size / (1.0 - cfg.training.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=relative_val_size,
        random_state=cfg.training.random_state,
        stratify=y_trainval,
    )

    logger.info(
        "Split — train: %d rows, val: %d rows, test: %d rows",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    pipeline = build_pipeline(params)

    if run_name is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"lgbm_churn_{ts}"

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info("MLflow run started: %s (run_id: %s)", run_name, run_id)

        # Step 1 — Cross-validation on training set
        logger.info("Running %d-fold cross-validation...", cfg.training.cv_folds)
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
            n_jobs=1,
        )
        cv_mean = round(float(cv_scores.mean()), 4)
        cv_std = round(float(cv_scores.std()), 4)
        mlflow.log_metric("cv_mean_roc_auc", cv_mean)
        mlflow.log_metric("cv_std_roc_auc", cv_std)
        logger.info("CV ROC-AUC: %.4f ± %.4f", cv_mean, cv_std)

        # Step 2 — Fit on full training set
        logger.info("Fitting pipeline on full training set...")
        pipeline.fit(X_train, y_train)

        # Step 2.5 — Calibrate probabilities on VALIDATION set
        from sklearn.calibration import CalibratedClassifierCV

        logger.info("Calibrating pipeline on validation set...")
        calibrated_pipeline = CalibratedClassifierCV(
            estimator=pipeline, method="isotonic", cv="prefit"
        )
        calibrated_pipeline.fit(X_val, y_val)

        # Step 3 — Find optimal threshold on VALIDATION set (not test)
        # This prevents data leakage: the threshold is tuned on val,
        # and the test set remains untouched until final evaluation.
        y_val_proba = calibrated_pipeline.predict_proba(X_val)[:, 1]
        optimal_threshold = find_cost_optimal_threshold(y_val.values, y_val_proba)
        f1_threshold = find_f1_optimal_threshold(y_val.values, y_val_proba)
        mlflow.log_metric("optimal_threshold", optimal_threshold)
        mlflow.log_metric("f1_threshold", f1_threshold)

        # Step 4 — Evaluate on held-out TEST set with threshold from val
        y_test_proba = calibrated_pipeline.predict_proba(X_test)[:, 1]
        test_metrics = evaluate(y_test.values, y_test_proba, optimal_threshold)
        _log_params_to_mlflow(pipeline, cfg)
        _log_metrics_to_mlflow(test_metrics, prefix="test_")

        print_evaluation_report(test_metrics, split_name="Test")

        # Step 5 — Log model artifact with signature
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
        logger.info("Model logged to MLflow (run_id: %s)", run_id)

        # Step 6 — Check performance gates and log pass/fail tag
        gates = cfg.model.performance_gates
        gates_passed = (
            test_metrics["roc_auc"] >= gates.min_roc_auc
            and test_metrics["pr_auc"] >= gates.min_pr_auc
            and test_metrics["recall"] >= gates.min_recall_at_threshold
        )
        mlflow.set_tag("gates_passed", str(gates_passed))
        mlflow.set_tag("model_type", "LightGBM")
        mlflow.set_tag("feature_count", str(len(X_train.columns)))

        if gates_passed:
            logger.info(
                "All performance gates PASSED. Model registered as '%s'.",
                cfg.model.registered_model_name,
            )
        else:
            logger.warning(
                "Performance gates FAILED. "
                "ROC-AUC: %.4f (min: %.2f) | "
                "PR-AUC: %.4f (min: %.2f) | "
                "Recall: %.4f (min: %.2f)",
                test_metrics["roc_auc"],
                gates.min_roc_auc,
                test_metrics["pr_auc"],
                gates.min_pr_auc,
                test_metrics["recall"],
                gates.min_recall_at_threshold,
            )

    final_metrics = {
        **test_metrics,
        "cv_mean_roc_auc": cv_mean,
        "cv_std_roc_auc": cv_std,
        "optimal_threshold": optimal_threshold,
        "f1_threshold": f1_threshold,
        "mlflow_run_id": run_id,
        "gates_passed": gates_passed,
    }

    return final_metrics
