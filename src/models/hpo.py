"""
Bayesian Hyperparameter Optimisation with Optuna.

Replaces ad-hoc hyperparameter selection with principled Bayesian search.
Uses the Tree-Parzen Estimator (TPE) sampler — a sequential model-based
optimisation algorithm that models P(x|y) and P(y) separately (unlike
Gaussian Processes which model P(y|x) directly), making it scale well
to high-dimensional, conditional search spaces.

Why PR-AUC as objective (not ROC-AUC):
    ROC-AUC is inflated by true negatives, which dominate in imbalanced
    data (73.5% non-churners). PR-AUC focuses exclusively on the
    positive class (churners) — the class we care about operationally.
    Optimising ROC-AUC may select hyperparameters that improve TN
    classification at the expense of churner detection.

Pruning strategy:
    MedianPruner stops unpromising trials early by comparing each trial's
    intermediate CV fold scores against the running median of all trials.
    This typically saves 30-50% of total compute budget.

Integration:
    The function returns a params dict that plugs directly into
    build_pipeline(params=best_params) — the pipeline factory already
    supports param overrides.

References:
    - Bergstra et al. (2011), "Algorithms for Hyper-Parameter
      Optimization", NeurIPS
    - Akiba et al. (2019), "Optuna: A Next-generation Hyperparameter
      Optimization Framework", KDD
    - Watanabe (2023), "Tree-Structured Parzen Estimator: Understanding
      Its Algorithm Components and Their Roles", arXiv:2304.11127

Public API:
    create_search_space(trial)                            → dict
    run_hpo(X_train, y_train, X_val, y_val, n_trials)    → HPOResult
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from src.models.pipeline import build_pipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Suppress Optuna's verbose trial-by-trial logging.
# Our own logger handles progress reporting.
optuna.logging.set_verbosity(optuna.logging.WARNING)


def create_search_space(trial: optuna.Trial) -> dict[str, Any]:
    """
    Define the LightGBM hyperparameter search space.

    Each parameter range is chosen based on:
        - LightGBM documentation recommendations
        - Empirical evidence from tabular benchmarking papers
          (Grinsztajn et al., NeurIPS 2022)
        - Domain knowledge about the Telco churn dataset size (~7K rows)

    Args:
        trial: Optuna trial object for parameter suggestion.

    Returns:
        Dict of hyperparameters compatible with LGBMClassifier.
    """
    return {
        # Number of boosting rounds. Log-uniform avoids spending too
        # many trials on high values.
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        # Learning rate. Lower = more boosting rounds needed but
        # potentially better generalisation.
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        # Tree complexity — num_leaves controls the main expressiveness.
        # Values above 63 tend to overfit on datasets < 10K rows.
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        # Max depth as a secondary complexity control.
        # -1 means unlimited (controlled by num_leaves instead).
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        # Minimum samples in a leaf. Higher = more regularised.
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        # L1 regularisation on leaf weights. Sparsity-inducing.
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        # L2 regularisation on leaf weights. Ridge-like smoothing.
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        # Row and column subsampling — stochastic gradient boosting.
        # Reduces variance and overfitting.
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        # Class weight to handle the 26.5% churn imbalance.
        # Searched around the theoretical inverse ratio (73.5/26.5 ≈ 2.77).
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
        # Fixed params (not searched — these are infrastructure choices)
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }


@dataclass
class HPOResult:
    """Result of a hyperparameter optimisation run."""

    best_params: dict[str, Any]
    best_score: float  # Best PR-AUC
    n_trials: int
    n_pruned: int
    n_completed: int
    all_trials_summary: list[dict[str, Any]]

    def summary(self) -> str:
        """Human-readable HPO summary."""
        lines = [
            "=" * 60,
            "  BAYESIAN HYPERPARAMETER OPTIMISATION REPORT",
            "=" * 60,
            f"  Total trials       : {self.n_trials}",
            f"  Completed trials   : {self.n_completed}",
            f"  Pruned trials      : {self.n_pruned}",
            f"  Best PR-AUC (CV)   : {self.best_score:.4f}",
            "",
            "  Best Hyperparameters:",
        ]
        for k, v in sorted(self.best_params.items()):
            if k in ("random_state", "verbose", "n_jobs"):
                continue
            lines.append(f"    {k:<25}: {v}")
        lines.append("=" * 60)
        return "\n".join(lines)


def run_hpo(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
    cv_folds: int = 3,
    random_state: int = 42,
) -> HPOResult:
    """
    Run Bayesian hyperparameter optimisation with Optuna.

    Uses 3-fold stratified CV with PR-AUC as the objective.
    MedianPruner prunes unpromising trials based on intermediate
    fold scores.

    Args:
        X_train:      Training features.
        y_train:      Training labels.
        n_trials:     Number of Optuna trials to run.
        cv_folds:     Number of CV folds per trial.
        random_state: Random seed for reproducibility.

    Returns:
        HPOResult with best parameters and trial summaries.
    """
    logger.info(
        "Starting Bayesian HPO — %d trials, %d-fold CV, objective: PR-AUC...",
        n_trials,
        cv_folds,
    )

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = create_search_space(trial)
        pipeline = build_pipeline(params=params)

        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr = X_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_va = X_train.iloc[val_idx]
            y_va = y_train.iloc[val_idx]

            pipeline.fit(X_tr, y_tr)
            y_prob = pipeline.predict_proba(X_va)[:, 1]
            pr_auc = average_precision_score(y_va, y_prob)
            fold_scores.append(pr_auc)

            # Report intermediate value for pruning.
            # After each fold, the pruner decides if this trial is
            # worth continuing based on how it compares to other trials.
            trial.report(pr_auc, fold_idx)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    # Create study with TPE sampler and MedianPruner
    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,  # Don't prune the first 5 trials
        n_warmup_steps=1,  # Start pruning after 1 fold
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="churn_lgbm_hpo",
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Compile results
    best_trial = study.best_trial
    trial_summaries = []
    for t in study.trials:
        trial_summaries.append(
            {
                "number": t.number,
                "value": round(t.value, 4) if t.value is not None else None,
                "state": str(t.state),
                "params": {
                    k: round(v, 6) if isinstance(v, float) else v
                    for k, v in t.params.items()
                },
            }
        )

    result = HPOResult(
        best_params=best_trial.params,
        best_score=round(best_trial.value, 4),
        n_trials=n_trials,
        n_pruned=len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        ),
        n_completed=len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        ),
        all_trials_summary=trial_summaries,
    )

    logger.info(
        "HPO complete — best PR-AUC: %.4f, %d/%d trials pruned.",
        result.best_score,
        result.n_pruned,
        result.n_trials,
    )

    return result
