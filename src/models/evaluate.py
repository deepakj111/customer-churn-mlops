"""
Model evaluation — ML metrics and business ROI metrics.

Separates evaluation logic from training orchestration so metrics
can be computed identically during training, cross-validation,
monitoring, and challenger comparison.

All functions accept numpy arrays (y_true, y_pred_proba) so they
work with any sklearn-compatible model, not just LightGBM.

Public API:
    compute_ml_metrics(y_true, y_pred_proba, threshold)  → dict
    compute_business_metrics(y_true, y_pred, cfg)        → dict
    evaluate(y_true, y_pred_proba, threshold)            → dict (combined)
    print_evaluation_report(metrics)                     → None
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.config_loader import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_ml_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute standard ML classification metrics.

    Uses the provided threshold to convert probabilities to binary
    predictions. The threshold is not optimised here — pass an
    already-optimised threshold from threshold.py.

    Metrics returned:
        roc_auc      — Area under ROC curve (good for overall discrimination)
        pr_auc       — Area under Precision-Recall curve (primary metric for
                       imbalanced data — not fooled by TN dominance)
        f1           — Harmonic mean of precision and recall at threshold
        precision    — Of all predicted churners, how many actually churned
        recall       — Of all actual churners, how many did we catch
        threshold    — The threshold used for binary conversion

    Args:
        y_true:        Ground truth binary labels (0 or 1).
        y_pred_proba:  Predicted churn probabilities from model.predict_proba.
        threshold:     Decision threshold for positive class. Default 0.5.

    Returns:
        Dict of metric name → float value, all rounded to 4 decimal places.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        "roc_auc": round(float(roc_auc_score(y_true, y_pred_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_pred_proba)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "threshold": round(float(threshold), 4),
    }

    logger.debug("ML metrics computed: %s", metrics)
    return metrics


def compute_business_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute business-oriented metrics using the cost matrix from config.

    Business cost model (from configs/model_config.yaml):
        False Negative (FN): Miss a churner → lose their full annual value.
                             Cost = false_negative_cost per customer.
        False Positive (FP): Flag a loyal customer → waste a retention offer.
                             Cost = false_positive_cost per customer.

    This is why we optimise for recall over precision — missing a churner
    is ~10x more expensive than a wasted retention offer.

    Metrics returned:
        true_positives       — Churners correctly identified
        false_positives      — Loyal customers incorrectly flagged
        false_negatives      — Churners missed by the model
        true_negatives       — Loyal customers correctly identified
        cost_of_fn           — Revenue lost from missed churners
        cost_of_fp           — Cost of wasted retention offers
        total_cost           — Combined cost at this threshold
        estimated_savings    — Revenue protected vs no model baseline

    Args:
        y_true:  Ground truth binary labels.
        y_pred:  Binary predictions at the chosen threshold.

    Returns:
        Dict of business metric name → value.
    """
    cfg = get_config()
    fn_cost = cfg.model.cost_matrix.false_negative_cost
    fp_cost = cfg.model.cost_matrix.false_positive_cost

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    cost_fn = fn * fn_cost
    cost_fp = fp * fp_cost
    total_cost = cost_fn + cost_fp

    # Baseline: no model, flag nobody → all churners are missed
    baseline_cost = int(y_true.sum()) * fn_cost
    estimated_savings = baseline_cost - total_cost

    metrics = {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "cost_of_fn": round(cost_fn, 2),
        "cost_of_fp": round(cost_fp, 2),
        "total_cost": round(total_cost, 2),
        "baseline_cost": round(baseline_cost, 2),
        "estimated_savings": round(estimated_savings, 2),
    }

    logger.debug("Business metrics computed: %s", metrics)
    return metrics


def evaluate(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Combine ML metrics and business metrics into one evaluation dict.

    This is the function called by the training pipeline and the
    monitoring pipeline. Returns a flat dict that maps directly to
    MLflow metric names — every key becomes an mlflow.log_metric call.

    Args:
        y_true:         Ground truth binary labels.
        y_pred_proba:   Predicted churn probabilities.
        threshold:      Decision threshold for binary predictions.

    Returns:
        Flat dict combining all ML and business metrics.
    """
    ml_metrics = compute_ml_metrics(y_true, y_pred_proba, threshold)

    y_pred = (y_pred_proba >= threshold).astype(int)
    business_metrics = compute_business_metrics(y_true, y_pred)

    combined = {**ml_metrics, **business_metrics}

    logger.info(
        "Evaluation complete — ROC-AUC: %.4f | PR-AUC: %.4f | "
        "Recall: %.4f | Estimated savings: $%.0f",
        combined["roc_auc"],
        combined["pr_auc"],
        combined["recall"],
        combined["estimated_savings"],
    )

    return combined


def print_evaluation_report(metrics: dict, split_name: str = "Test") -> None:
    """
    Print a formatted evaluation report to stdout.

    Used in notebooks and training scripts for human-readable output.
    Not used in automated pipelines — those read the metrics dict directly.

    Args:
        metrics:    Dict returned by evaluate().
        split_name: Label for the split (e.g. "Test", "Validation").
    """
    print(f"\n{'=' * 55}")
    print(f"  EVALUATION REPORT — {split_name.upper()} SET")
    print(f"{'=' * 55}")
    print("\n  ML METRICS")
    print(f"  {'ROC-AUC':<28}: {metrics['roc_auc']:.4f}")
    print(f"  {'PR-AUC (primary metric)':<28}: {metrics['pr_auc']:.4f}")
    print(f"  {'F1 Score':<28}: {metrics['f1']:.4f}")
    print(f"  {'Precision':<28}: {metrics['precision']:.4f}")
    print(f"  {'Recall':<28}: {metrics['recall']:.4f}")
    print(f"  {'Decision Threshold':<28}: {metrics['threshold']:.4f}")
    print("\n  CONFUSION MATRIX")
    print(f"  {'True Positives':<28}: {metrics['true_positives']}")
    print(f"  {'False Positives':<28}: {metrics['false_positives']}")
    print(f"  {'False Negatives':<28}: {metrics['false_negatives']}")
    print(f"  {'True Negatives':<28}: {metrics['true_negatives']}")
    print("\n  BUSINESS IMPACT")
    print(f"  {'FN Cost (missed churners)':<28}: ${metrics['cost_of_fn']:>10,.0f}")
    print(f"  {'FP Cost (wasted offers)':<28}: ${metrics['cost_of_fp']:>10,.0f}")
    print(f"  {'Total Cost':<28}: ${metrics['total_cost']:>10,.0f}")
    print(f"  {'Baseline Cost (no model)':<28}: ${metrics['baseline_cost']:>10,.0f}")
    print(f"  {'Estimated Savings':<28}: ${metrics['estimated_savings']:>10,.0f}")
    print(f"{'=' * 55}\n")
