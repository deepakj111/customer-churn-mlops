"""
Decision threshold optimisation.

The default threshold of 0.5 is almost never optimal for imbalanced
binary classification. This module finds the threshold that minimises
the business cost function defined in model_config.yaml.

Two strategies are provided:
    1. cost_optimal_threshold  — minimises (FN_cost × FN + FP_cost × FP)
                                  Primary strategy. Uses business cost matrix.
    2. f1_optimal_threshold    — maximises F1 score
                                  Secondary strategy. Used as a sanity check
                                  and for comparing against the cost-optimal result.

Why not just use 0.5?
    Our cost matrix is asymmetric: FN costs 10× more than FP.
    A threshold of 0.5 maximises accuracy, not business value.
    At 0.34 (typical result), recall improves from ~60% to ~78%
    while precision drops only marginally — a net positive on the
    cost function.

Public API:
    find_cost_optimal_threshold(y_true, y_pred_proba)  → float
    find_f1_optimal_threshold(y_true, y_pred_proba)    → float
    get_risk_tier(probability)                         → str
"""

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from src.utils.config_loader import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Search 99 candidate thresholds between 0.01 and 0.99
_THRESHOLD_CANDIDATES = np.linspace(0.01, 0.99, 99)


def find_cost_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> float:
    """
    Find the threshold that minimises total business cost.

    Total cost = (FN_cost × false_negatives) + (FP_cost × false_positives)

    Searches 99 candidate thresholds from 0.01 to 0.99 and returns
    the one with the lowest total cost. Ties are broken by choosing
    the lower threshold (maximises recall when cost is equal).

    Args:
        y_true:         Ground truth binary labels (0 or 1).
        y_pred_proba:   Predicted churn probabilities.

    Returns:
        Optimal threshold as a float, rounded to 2 decimal places.
    """
    cfg = get_config()
    fn_cost = cfg.model.cost_matrix.false_negative_cost
    fp_cost = cfg.model.cost_matrix.false_positive_cost

    best_threshold = 0.5
    best_cost = float("inf")

    for threshold in _THRESHOLD_CANDIDATES:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = (fn * fn_cost) + (fp * fp_cost)

        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = threshold

    optimal = round(float(best_threshold), 2)

    logger.info(
        "Cost-optimal threshold: %.2f (total cost: $%.0f | "
        "FN cost: $%.0f per miss, FP cost: $%.0f per false alarm)",
        optimal,
        best_cost,
        fn_cost,
        fp_cost,
    )

    return optimal


def find_f1_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> float:
    """
    Find the threshold that maximises the F1 score.

    Used as a secondary reference point alongside cost_optimal_threshold.
    When the two thresholds diverge significantly (> 0.15 apart), it
    signals that the cost matrix is driving the decision meaningfully —
    which is exactly what we want.

    Args:
        y_true:         Ground truth binary labels.
        y_pred_proba:   Predicted churn probabilities.

    Returns:
        F1-optimal threshold as a float, rounded to 2 decimal places.
    """
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in _THRESHOLD_CANDIDATES:
        y_pred = (y_pred_proba >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    optimal = round(float(best_threshold), 2)

    logger.info("F1-optimal threshold: %.2f (best F1: %.4f)", optimal, best_f1)
    return optimal


def get_risk_tier(probability: float) -> str:
    """
    Map a churn probability to a human-readable risk tier.

    Tier boundaries are read from model_config.yaml:
        HIGH_RISK   : probability >= high_risk_threshold (default 0.60)
        MEDIUM_RISK : probability >= medium_risk_threshold (default 0.35)
        LOW_RISK    : probability < medium_risk_threshold

    The tiered output is what the FastAPI response returns to the
    retention team. They prioritise the HIGH_RISK list for immediate
    outreach and the MEDIUM_RISK list for automated campaigns.

    Args:
        probability: Churn probability from model.predict_proba (0–1).

    Returns:
        One of: "HIGH_RISK", "MEDIUM_RISK", "LOW_RISK"
    """
    cfg = get_config()
    high = cfg.model.risk_tiers.high_risk_threshold
    medium = cfg.model.risk_tiers.medium_risk_threshold

    if probability >= high:
        return "HIGH_RISK"
    elif probability >= medium:
        return "MEDIUM_RISK"
    return "LOW_RISK"
