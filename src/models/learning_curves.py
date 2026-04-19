"""
Learning Curve Analysis — Data Efficiency and Bias-Variance Diagnostics.

Answers two critical questions that every ML researcher should address:
    1. "Would more data improve this model?" — If the validation curve
       is still rising steeply, collecting more data is worthwhile.
    2. "Is the model overfitting or underfitting?" — The gap between
       training and validation curves reveals the bias-variance regime.

Interpretation guide:
    HIGH BIAS (underfitting):
        - Both train and val curves plateau at low performance.
        - The gap between them is small.
        - Fix: More complex model, more features, less regularisation.

    HIGH VARIANCE (overfitting):
        - Training score is high, validation score is much lower.
        - The gap between them is large.
        - Fix: More data, more regularisation, simpler model.

    WELL-FITTED:
        - Both curves converge at high performance.
        - The gap is small and stable.
        - This is where you want to be.

References:
    - Ng, Andrew (2017), "Machine Learning Yearning", Chapters 5-6
    - Hastie, Tibshirani, Friedman (2009), "The Elements of Statistical
      Learning", Chapter 7 (Model Assessment and Selection)

Public API:
    compute_learning_curves(pipeline, X, y, ...)     → LearningCurveReport
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, learning_curve

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LearningCurveReport:
    """Results of a learning curve analysis."""

    train_sizes_abs: list[int]  # Actual number of training samples used
    train_sizes_rel: list[float]  # Fraction of total training data
    train_scores_mean: list[float]
    train_scores_std: list[float]
    val_scores_mean: list[float]
    val_scores_std: list[float]
    scoring_metric: str
    n_cv_folds: int
    total_samples: int

    # Derived diagnostics
    final_gap: float  # Train - Val at max data (bias-variance signal)
    val_slope: float  # Slope of val curve at tail (data hunger signal)
    diagnosis: str  # "high_bias", "high_variance", "well_fitted"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for JSON export and MLflow logging."""
        return {
            "train_sizes_abs": self.train_sizes_abs,
            "train_sizes_rel": self.train_sizes_rel,
            "train_scores_mean": self.train_scores_mean,
            "train_scores_std": self.train_scores_std,
            "val_scores_mean": self.val_scores_mean,
            "val_scores_std": self.val_scores_std,
            "scoring_metric": self.scoring_metric,
            "n_cv_folds": self.n_cv_folds,
            "total_samples": self.total_samples,
            "final_gap": self.final_gap,
            "val_slope": self.val_slope,
            "diagnosis": self.diagnosis,
        }

    def summary(self) -> str:
        """Human-readable learning curve analysis."""
        lines = [
            "=" * 60,
            "  LEARNING CURVE ANALYSIS",
            "=" * 60,
            f"  Scoring metric : {self.scoring_metric}",
            f"  CV folds       : {self.n_cv_folds}",
            f"  Total samples  : {self.total_samples}",
            "",
            "  Training Size    Train Score    Val Score      Gap",
            "  " + "-" * 54,
        ]

        for i in range(len(self.train_sizes_abs)):
            gap = self.train_scores_mean[i] - self.val_scores_mean[i]
            lines.append(
                f"  {self.train_sizes_abs[i]:>7d} ({self.train_sizes_rel[i]:.0%})   "
                f"{self.train_scores_mean[i]:.4f} ± {self.train_scores_std[i]:.3f}  "
                f"{self.val_scores_mean[i]:.4f} ± {self.val_scores_std[i]:.3f}  "
                f"{gap:+.4f}"
            )

        lines.append("")
        lines.append(f"  Final gap (train - val at 100%): {self.final_gap:+.4f}")
        lines.append(f"  Validation tail slope:           {self.val_slope:+.6f}")
        lines.append(f"  Diagnosis:                       {self.diagnosis.upper()}")

        if self.diagnosis == "high_variance":
            lines.append(
                "  → Model is overfitting. Consider: more data, "
                "regularisation, simpler model."
            )
        elif self.diagnosis == "high_bias":
            lines.append(
                "  → Model is underfitting. Consider: more features, "
                "less regularisation, more complex model."
            )
        else:
            lines.append(
                "  → Model is well-fitted. Training and validation have converged."
            )

        lines.append("=" * 60)
        return "\n".join(lines)


def compute_learning_curves(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    train_sizes: list[float] | None = None,
    cv_folds: int = 5,
    scoring: str = "average_precision",
    random_state: int = 42,
    n_jobs: int = 1,
) -> LearningCurveReport:
    """
    Compute stratified learning curves for the given pipeline.

    Trains the pipeline on increasing fractions of the data and evaluates
    on held-out folds at each size. Uses stratified k-fold CV to maintain
    class balance at every training size.

    Args:
        pipeline:     Unfitted sklearn Pipeline or estimator.
        X:            Full training feature DataFrame.
        y:            Full training label Series.
        train_sizes:  List of fractions of total data to train on.
                      Default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        cv_folds:     Number of stratified CV folds.
        scoring:      sklearn scoring metric name.
        random_state: Random seed.
        n_jobs:       Parallel jobs (-1 for all cores).

    Returns:
        LearningCurveReport with train/val curves and bias-variance diagnosis.
    """
    if train_sizes is None:
        train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    logger.info(
        "Computing learning curves — %d sizes, %d-fold CV, metric: %s...",
        len(train_sizes),
        cv_folds,
        scoring,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator=pipeline,
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        return_times=False,
    )

    # Compute statistics across folds
    train_mean = [round(float(s), 4) for s in train_scores.mean(axis=1)]
    train_std = [round(float(s), 4) for s in train_scores.std(axis=1)]
    val_mean = [round(float(s), 4) for s in val_scores.mean(axis=1)]
    val_std = [round(float(s), 4) for s in val_scores.std(axis=1)]

    # Diagnostics
    final_gap = round(float(train_mean[-1] - val_mean[-1]), 4)

    # Val slope at tail: average change per step over last 3 points.
    # Positive slope = validation is still improving with more data.
    if len(val_mean) >= 3:
        tail_diffs = np.diff(val_mean[-3:])
        val_slope = round(float(np.mean(tail_diffs)), 6)
    else:
        val_slope = 0.0

    # Diagnosis heuristics:
    # High variance: large gap between train and val (>0.05)
    # High bias: both train and val plateau low (val < 0.7 for PR-AUC)
    # Thresholds are calibrated for PR-AUC on churn data
    if final_gap > 0.05:
        diagnosis = "high_variance"
    elif val_mean[-1] < 0.60:
        diagnosis = "high_bias"
    else:
        diagnosis = "well_fitted"

    report = LearningCurveReport(
        train_sizes_abs=[int(s) for s in train_sizes_abs],
        train_sizes_rel=[round(float(s / len(X)), 2) for s in train_sizes_abs],
        train_scores_mean=train_mean,
        train_scores_std=train_std,
        val_scores_mean=val_mean,
        val_scores_std=val_std,
        scoring_metric=scoring,
        n_cv_folds=cv_folds,
        total_samples=len(X),
        final_gap=final_gap,
        val_slope=val_slope,
        diagnosis=diagnosis,
    )

    logger.info(
        "Learning curves computed — diagnosis: %s, final gap: %.4f, tail slope: %.6f",
        diagnosis,
        final_gap,
        val_slope,
    )

    return report
