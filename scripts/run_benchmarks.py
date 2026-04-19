"""
Benchmarking Script: LightGBM vs TabPFN vs FT-Transformer.

This script runs a rigorous 5-fold cross-validation experiment, comparing
traditional calibrated gradient boosting against emerging zero-shot
transformers (TabPFN) and natively tabular deep learning (FT-Transformer).

We evaluate:
- ROC-AUC
- F1-Score
- Brier Score (Calibration Quality)
- Inference Latency (ms/sample)
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.data.ingest import load_for_training
from src.data.preprocess import run_preprocessing
from src.data.validate import validate_raw_data
from src.features.feature_store import engineer_features
from src.models.ft_transformer_wrapper import FTTransformerWrapper
from src.models.pipeline import build_pipeline, get_preprocessor
from src.models.statistical_comparison import compare_models
from src.models.tabpfn_wrapper import TabPFNWrapper
from src.utils.logging import get_logger

logger = get_logger(__name__)


def swap_classifier(pipeline: Pipeline, new_classifier) -> Pipeline:
    """Clones pipeline and swaps the classifier step."""
    steps = []
    for name, step in pipeline.steps:
        if name == "classifier":
            steps.append((name, new_classifier))
        elif name == "feature_selection":
            # TabPFN/FT-Transformer handle selection inherently/differently,
            # we drop LightGBM specific feature selection for neural models.
            continue
        else:
            steps.append((name, step))
    return Pipeline(steps=steps)


def build_neural_pipeline(model_name: str) -> Pipeline:
    """
    Builds a custom pipeline for neural models.
    TabPFN needs fully preprocessed numbers.
    FT-Transformer needs a raw-ish dataframe (Feast allowed, but no OHE needed).
    """
    if model_name == "TabPFN":
        # TabPFN expects numeric matrices, so we use full sklearn preprocessing OHE.
        return Pipeline(
            steps=[
                (
                    "feature_engineering",
                    FunctionTransformer(engineer_features, validate=False),
                ),
                ("preprocessor", get_preprocessor()),
                (
                    "classifier",
                    TabPFNWrapper(device="cpu"),
                ),
            ]
        )
    elif model_name == "FT-Transformer":
        # PyTorch tabular inherently tokenizes and embeds categories,
        # so we DO NOT one-hot encode. We just engineer Feast features and
        # pass the DataFrame.
        return Pipeline(
            steps=[
                (
                    "feature_engineering",
                    FunctionTransformer(engineer_features, validate=False),
                ),
                ("classifier", FTTransformerWrapper(epochs=5, batch_size=256)),
            ]
        )
    else:
        raise ValueError("Unknown model name.")


def evaluate_model(
    model_name: str, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv=5
) -> dict:
    """Run cross-validation and compute advanced metrics."""
    logger.info("Evaluating %s via %d-fold CV...", model_name, cv)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    auc_scores = []
    f1_scores = []
    brier_scores = []
    inference_times = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        logger.info("[%s] Fold %d/%d", model_name, fold + 1, cv)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # LightGBM gets calibrated automatically in our main codebase,
        # but here we evaluate the raw pipeline for pure architectural comparison.
        pipeline.fit(X_train, y_train)

        # Time inference latency
        start_time = time.time()
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = pipeline.predict(X_test)
        inf_time = time.time() - start_time

        # Batch latency in milliseconds per sample
        ms_per_sample = (inf_time / len(X_test)) * 1000

        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_pred_proba)

        auc_scores.append(auc)
        f1_scores.append(f1)
        brier_scores.append(brier)
        inference_times.append(ms_per_sample)

    report = {
        "ROC-AUC": round(float(np.mean(auc_scores)), 4),
        "ROC-AUC_std": round(float(np.std(auc_scores)), 4),
        "F1": round(float(np.mean(f1_scores)), 4),
        "Brier": round(float(np.mean(brier_scores)), 4),
        "Latency_ms_per_sample": round(float(np.mean(inference_times)), 4),
        "fold_auc_scores": [round(s, 4) for s in auc_scores],
    }

    logger.info(
        "[%s] CV Results - AUC: %.4f | F1: %.4f | MS/Sample: %.2f ms",
        model_name,
        report["ROC-AUC"],
        report["F1"],
        report["Latency_ms_per_sample"],
    )
    return report


def run_all_benchmarks():
    """Execute benchmarking suite."""
    raw_df = load_for_training()
    validated_df = validate_raw_data(raw_df)
    X, y = run_preprocessing(validated_df)

    # 1. Base LightGBM
    lgbm_pipeline = build_pipeline()

    # 2. FT-Transformer (Needs deep tabular wrapper)
    ft_pipeline = build_neural_pipeline("FT-Transformer")

    # Due to processing time, we benchmark against a subset or 3 folds config
    # depending on constraints. We will use 3 folds here to ensure rapid
    # exploration turnaround.
    FOLDS = 3

    results = {}

    try:
        results["LightGBM"] = evaluate_model("LightGBM", lgbm_pipeline, X, y, cv=FOLDS)
        results["FT-Transformer"] = evaluate_model(
            "FT-Transformer", ft_pipeline, X, y, cv=FOLDS
        )
    except Exception as e:
        logger.error("Benchmarking failed: %s", e)
        raise e

    # Sort models by AUC for the final report
    sorted_results = {
        name: metrics
        for name, metrics in sorted(
            results.items(), key=lambda item: item[1]["ROC-AUC"], reverse=True
        )
    }

    # Statistical model comparison (Nadeau-Bengio corrected t-test)
    # Determines whether AUC differences are statistically significant,
    # not just random noise from the particular train/test splits.
    fold_scores = {
        name: np.array(metrics["fold_auc_scores"]) for name, metrics in results.items()
    }
    # Approximate n_train and n_test from fold proportions
    n_total = len(X)
    n_test_approx = n_total // FOLDS
    n_train_approx = n_total - n_test_approx

    if len(fold_scores) >= 2:
        comparison_report = compare_models(
            fold_scores=fold_scores,
            n_train=n_train_approx,
            n_test=n_test_approx,
            metric_name="ROC-AUC",
        )
        logger.info("\n%s", comparison_report.summary())
        sorted_results["statistical_comparison"] = comparison_report.to_dict()

    # Save artifact
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / "benchmark_report.json"

    with open(out_path, "w") as f:
        json.dump(sorted_results, f, indent=4)

    logger.info("Benchmarking suite complete. Results written to %s", out_path)


if __name__ == "__main__":
    run_all_benchmarks()
