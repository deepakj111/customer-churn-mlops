"""Feature selection module — multi-stage pipeline.

Implements a 5-stage selection pipeline to determine which of the 28
engineered features contribute unique and significant predictive signal.

Why multi-stage? No single method is perfect:
    - Variance filter removes near-constant features that carry no information
    - Correlation filter removes redundant features (e.g., is_isolated mirrors
      has_family perfectly — only one is needed)
    - Mutual Information scores non-linear associations but is noisy
    - RF Gini importance favors features used early in trees
    - Permutation importance is model-agnostic but evaluates actual prediction impact
    Consensus across all scoring methods gives robust, defensible selection.

This module is used in the feature engineering notebook (02_feature_engineering
_experiments.py) to justify which features are retained in the final model.

Public API:
    run_variance_filter(X, threshold)          → surviving_cols, dropped_cols
    run_correlation_filter(X, threshold)       → surviving_cols, dropped_cols,
        corr_matrix
    run_mutual_information(X, y)               → scored DataFrame
    run_model_based_selection(X, y)            → scored DataFrame
    run_permutation_importance(X, y)           → scored DataFrame
    build_rank_consensus(mi_df, rf_df, perm_df, features)  → ranked DataFrame
    run_full_selection(X_engineered, y)        → FeatureSelectionReport
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.logging import get_logger

warnings.filterwarnings("ignore")

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


@dataclass
class FeatureSelectionReport:
    """Container for all outputs of the full feature selection pipeline."""

    all_features: list[str]
    variance_dropped: list[str]
    correlation_dropped: list[str]
    mi_scores: pd.DataFrame
    rf_scores: pd.DataFrame
    perm_scores: pd.DataFrame
    consensus_df: pd.DataFrame
    consensus_selected: list[str]
    correlation_matrix: pd.DataFrame
    cv_pr_auc_all: float
    cv_pr_auc_selected: float
    n_features_before: int = field(init=False)
    n_features_after: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_features_before = len(self.all_features)
        self.n_features_after = len(self.consensus_selected)

    def summary(self) -> str:
        lines = [
            "=" * 65,
            "FEATURE SELECTION REPORT",
            "=" * 65,
            f"  Features before selection : {self.n_features_before}",
            f"  Dropped by variance filter : {len(self.variance_dropped)}",
            f"  Dropped by corr filter     : {len(self.correlation_dropped)}",
            f"  Features after selection   : {self.n_features_after}",
            f"  CV PR-AUC (all filtered)   : {self.cv_pr_auc_all:.4f}",
            f"  CV PR-AUC (selected)       : {self.cv_pr_auc_selected:.4f}",
            f"  PR-AUC delta: {self.cv_pr_auc_selected - self.cv_pr_auc_all:+.4f}",
            "-" * 65,
            "  VARIANCE DROPPED:",
        ]
        for f in self.variance_dropped:
            lines.append(f"    - {f}")
        lines.append("  CORRELATION DROPPED:")
        for f in self.correlation_dropped:
            lines.append(f"    - {f}")
        lines.append("  CONSENSUS SELECTED:")
        for f in self.consensus_selected:
            lines.append(f"    + {f}")
        lines.append("=" * 65)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _label_encode(X: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all object columns for scikit-learn selectors."""
    X_enc = X.copy()
    le = LabelEncoder()
    for col in X_enc.select_dtypes(include="object").columns:
        X_enc[col] = le.fit_transform(X_enc[col].astype(str))
    return X_enc.fillna(0)


# ---------------------------------------------------------------------------
# Stage 1 — Variance Filter
# ---------------------------------------------------------------------------


def run_variance_filter(
    X: pd.DataFrame,
    threshold: float = 0.005,
) -> tuple[list[str], list[str]]:
    """Remove near-zero-variance features.

    Features with variance < threshold are nearly constant across all samples
    — they cannot discriminate between churners and non-churners.

    Rule of thumb: for a binary feature, variance=p*(1-p). A feature that is
    positive in only 0.5% of samples has variance ≈ 0.005 and carries almost
    no signal. Default threshold=0.005 catches features positive in < ~0.7%
    of samples.

    Args:
        X:         Feature DataFrame (engineered, object cols allowed).
        threshold: Minimum variance to retain a feature.

    Returns:
        surviving_cols: Features that passed the filter.
        dropped_cols:   Features removed.
    """
    X_enc = _label_encode(X)
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_enc)
    mask = selector.get_support()
    surviving = [col for col, keep in zip(X_enc.columns, mask) if keep]
    dropped = [col for col, keep in zip(X_enc.columns, mask) if not keep]
    logger.info(
        "Variance filter (threshold=%.4f): kept %d, dropped %d → %s",
        threshold,
        len(surviving),
        len(dropped),
        dropped or "none",
    )
    return surviving, dropped


# ---------------------------------------------------------------------------
# Stage 2 — Correlation Filter
# ---------------------------------------------------------------------------


def run_correlation_filter(
    X: pd.DataFrame,
    threshold: float = 0.90,
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Drop one feature from each highly correlated pair.

    When |Pearson r| > threshold between two features they carry nearly
    identical information. Keeping both destabilises linear models (multicollinearity)
    and wastes model capacity in tree models. From each correlated pair, the
    feature with HIGHER mean absolute correlation to ALL other features is
    dropped — it is the more redundant one in the feature space.

    Expected catches at r > 0.90:
        is_isolated ↔ has_family          (perfect negatives, r = -1.0)
        is_month_to_month ↔ contract_numeric  (derived, r ≈ -0.95)
        is_month_to_month ↔ is_committed_customer  (r = -1.0)

    Args:
        X:         Feature DataFrame (label-encoded).
        threshold: |r| above which one feature is dropped.

    Returns:
        surviving_cols: Features that passed.
        dropped_cols:   Features removed (one from each pair).
        corr_matrix:    Full |r| matrix for visualisation.
    """
    X_enc = _label_encode(X)
    corr = X_enc.corr().abs()
    upper = corr.where(np.triu(np.ones_like(corr, dtype=bool), k=1))
    to_drop: set[str] = set()
    for col in upper.columns:
        high_corr_partners = upper.index[upper[col] > threshold].tolist()
        for partner in high_corr_partners:
            mean_col = corr[col].mean()
            mean_partner = corr[partner].mean()
            loser = col if mean_col > mean_partner else partner
            to_drop.add(loser)
            logger.debug(
                "Corr filter: %s ↔ %s  r=%.3f → dropping %s",
                col,
                partner,
                corr.loc[partner, col] if partner in corr.index else 0.0,
                loser,
            )
    dropped = sorted(to_drop)
    surviving = [c for c in X_enc.columns if c not in to_drop]
    logger.info(
        "Correlation filter (threshold=%.2f): kept %d, dropped %d → %s",
        threshold,
        len(surviving),
        len(dropped),
        dropped or "none",
    )
    return surviving, dropped, corr


# ---------------------------------------------------------------------------
# Stage 3 — Mutual Information Scoring
# ---------------------------------------------------------------------------


def run_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> pd.DataFrame:
    """Score all features by mutual information with the binary target.

    MI measures the reduction in uncertainty about the target given a
    feature — it captures non-linear associations unlike Pearson r.
    MI = 0 means feature and target are independent.

    Args:
        X:            Feature DataFrame (label-encoded).
        y:            Binary target (0/1).
        random_state: For reproducibility.

    Returns:
        DataFrame with columns [feature, mi_score, mi_rank].
    """
    X_enc = _label_encode(X)
    scores = mutual_info_classif(
        X_enc,
        y,
        discrete_features="auto",
        random_state=random_state,
    )
    df = (
        pd.DataFrame({"feature": X_enc.columns, "mi_score": scores})
        .sort_values("mi_score", ascending=False)
        .reset_index(drop=True)
    )
    df["mi_rank"] = range(1, len(df) + 1)
    logger.info(
        "MI scoring complete. Top feature: %s (score=%.4f).",
        df["feature"].iloc[0],
        df["mi_score"].iloc[0],
    )
    return df


# ---------------------------------------------------------------------------
# Stage 4 — Model-Based Importance (Random Forest Gini)
# ---------------------------------------------------------------------------


def run_model_based_selection(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 300,
    random_state: int = 42,
) -> pd.DataFrame:
    """Score features using Random Forest Gini impurity importance.

    RF importance is fast, handles mixed feature types, and captures
    non-linear interactions. Limitation: can overestimate high-cardinality
    or redundant features. Cross-validate with MI and permutation importance.

    Args:
        X:             Feature DataFrame (label-encoded).
        y:             Binary target.
        n_estimators:  Number of RF trees.
        random_state:  For reproducibility.

    Returns:
        DataFrame with columns [feature, rf_importance, rf_rank].
    """
    X_enc = _label_encode(X)
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=12,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_enc, y)
    df = (
        pd.DataFrame(
            {"feature": X_enc.columns, "rf_importance": rf.feature_importances_}
        )
        .sort_values("rf_importance", ascending=False)
        .reset_index(drop=True)
    )
    df["rf_rank"] = range(1, len(df) + 1)
    logger.info(
        "RF importance complete. Top feature: %s (importance=%.4f).",
        df["feature"].iloc[0],
        df["rf_importance"].iloc[0],
    )
    return df


# ---------------------------------------------------------------------------
# Stage 5 — Permutation Importance
# ---------------------------------------------------------------------------


def run_permutation_importance(
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation importance against a Logistic Regression baseline.

    Permutation importance shuffles one feature at a time and measures the
    drop in PR-AUC. Features that don't matter can be shuffled without
    hurting the model (importance ≈ 0). Negative importance = feature was
    adding noise (model performs better without it).

    Uses LogisticRegression deliberately — shows which features matter for
    linear models, where most feature engineering value is realised. Tree
    models handle raw features better; this measures the engineering lift.

    Args:
        X:            Feature DataFrame (will be label-encoded + scaled).
        y:            Binary target.
        n_repeats:    Permutation repeats per feature (higher = more stable).
        random_state: For reproducibility.

    Returns:
        DataFrame with columns [feature, perm_mean, perm_std, perm_rank].
    """
    X_enc = _label_encode(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_enc)

    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
    )
    lr.fit(X_scaled, y)

    result = permutation_importance(
        lr,
        X_scaled,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="average_precision",
        n_jobs=-1,
    )
    df = pd.DataFrame(
        {
            "feature": X_enc.columns,
            "perm_mean": result.importances_mean,
            "perm_std": result.importances_std,
        }
    )
    df = df.sort_values("perm_mean", ascending=False).reset_index(drop=True)
    df["perm_rank"] = range(1, len(df) + 1)
    logger.info(
        "Permutation importance complete. Top feature: %s (mean=%.4f).",
        df["feature"].iloc[0],
        df["perm_mean"].iloc[0],
    )
    return df


# ---------------------------------------------------------------------------
# Stage 6 — Consensus Selection
# ---------------------------------------------------------------------------


def build_rank_consensus(
    mi_df: pd.DataFrame,
    rf_df: pd.DataFrame,
    perm_df: pd.DataFrame,
    all_features: list[str],
    top_n: int = 20,
    min_votes: int = 2,
) -> pd.DataFrame:
    """Rank all features by consensus across three selection methods.

    A feature earns a vote for each method where it appears in the top_n.
    Features with votes >= min_votes are selected. Ties broken by average rank.

    Selection logic:
        vote = (in top_n MI) + (in top_n RF) + (in top_n perm)
        select if vote >= min_votes

    Args:
        mi_df, rf_df, perm_df: Scored DataFrames from stages 3-5.
        all_features:          Full list of candidate features.
        top_n:                 Top-N cutoff per method.
        min_votes:             Minimum method agreement to select.

    Returns:
        DataFrame with columns [feature, mi_rank, rf_rank, perm_rank,
                                 avg_rank, votes, selected].
    """
    top_mi = set(mi_df.head(top_n)["feature"])
    top_rf = set(rf_df.head(top_n)["feature"])
    top_perm = set(perm_df.head(top_n)["feature"])

    mi_ranks = dict(zip(mi_df["feature"], mi_df["mi_rank"]))
    rf_ranks = dict(zip(rf_df["feature"], rf_df["rf_rank"]))
    perm_ranks = dict(zip(perm_df["feature"], perm_df["perm_rank"]))

    n = len(all_features)
    rows = []
    for feat in all_features:
        mi_r = mi_ranks.get(feat, n)
        rf_r = rf_ranks.get(feat, n)
        pm_r = perm_ranks.get(feat, n)
        votes = int(feat in top_mi) + int(feat in top_rf) + int(feat in top_perm)
        rows.append(
            {
                "feature": feat,
                "mi_rank": mi_r,
                "rf_rank": rf_r,
                "perm_rank": pm_r,
                "avg_rank": (mi_r + rf_r + pm_r) / 3,
                "votes": votes,
                "selected": votes >= min_votes,
            }
        )

    consensus_df = pd.DataFrame(rows).sort_values("avg_rank").reset_index(drop=True)
    selected = consensus_df.loc[consensus_df["selected"], "feature"].tolist()
    logger.info(
        "Consensus selection (top_%d, min_%d votes): %d/%d features selected.",
        top_n,
        min_votes,
        len(selected),
        len(all_features),
    )
    return consensus_df


# ---------------------------------------------------------------------------
# Full Pipeline Entry Point
# ---------------------------------------------------------------------------


def run_full_selection(
    X_engineered: pd.DataFrame,
    y: pd.Series,
    variance_threshold: float = 0.005,
    correlation_threshold: float = 0.90,
    top_n: int = 20,
    min_votes: int = 2,
    random_state: int = 42,
) -> FeatureSelectionReport:
    """Run the complete 5-stage feature selection pipeline.

    Stages:
        1. Variance filter       → drop near-constant features
        2. Correlation filter    → drop redundant features
        3. MI scoring            → non-linear association with target
        4. RF importance         → Gini impurity-based ranking
        5. Permutation importance → actual prediction impact (linear model)
        6. Consensus             → features that rank in top_n in 2+ methods
        7. CV comparison         → PR-AUC with all vs. selected features

    Args:
        X_engineered:          DataFrame with all 28 engineered features.
        y:                     Binary target (0/1).
        variance_threshold:    Min variance to keep a feature (default 0.005).
        correlation_threshold: |r| above which one of a pair is dropped (0.90).
        top_n:                 Top-N cutoff for each scoring method.
        min_votes:             Min methods agreeing for consensus selection.
        random_state:          Global random seed.

    Returns:
        FeatureSelectionReport with all intermediate results and final list.
    """
    all_features = X_engineered.columns.tolist()
    logger.info("Full feature selection on %d features.", len(all_features))

    # Stage 1
    surviving_v, var_dropped = run_variance_filter(X_engineered, variance_threshold)

    # Stage 2
    X_v = X_engineered[surviving_v]
    surviving_c, corr_dropped, corr_matrix = run_correlation_filter(
        X_v, correlation_threshold
    )

    # Stages 3-5 on correlation-surviving features
    X_filtered = X_engineered[surviving_c]
    mi_df = run_mutual_information(X_filtered, y, random_state=random_state)
    rf_df = run_model_based_selection(X_filtered, y, random_state=random_state)
    perm_df = run_permutation_importance(X_filtered, y, random_state=random_state)

    # Stage 6
    consensus_df = build_rank_consensus(
        mi_df,
        rf_df,
        perm_df,
        all_features=surviving_c,
        top_n=top_n,
        min_votes=min_votes,
    )
    selected = consensus_df.loc[consensus_df["selected"], "feature"].tolist()

    # Stage 7: CV comparison (LogReg — where feature engineering helps most)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    def _cv_prauc(X_subset: pd.DataFrame) -> float:
        X_enc = _label_encode(X_subset)
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        return float(
            cross_val_score(
                pipe, X_enc, y, cv=cv, scoring="average_precision", n_jobs=1
            ).mean()
        )

    cv_all = _cv_prauc(X_filtered)
    cv_sel = _cv_prauc(X_engineered[selected]) if selected else 0.0

    logger.info(
        "CV PR-AUC: all_filtered=%.4f  consensus_selected=%.4f  delta=%+.4f",
        cv_all,
        cv_sel,
        cv_sel - cv_all,
    )

    return FeatureSelectionReport(
        all_features=all_features,
        variance_dropped=var_dropped,
        correlation_dropped=corr_dropped,
        mi_scores=mi_df,
        rf_scores=rf_df,
        perm_scores=perm_df,
        consensus_df=consensus_df,
        consensus_selected=selected,
        correlation_matrix=corr_matrix,
        cv_pr_auc_all=cv_all,
        cv_pr_auc_selected=cv_sel,
    )
