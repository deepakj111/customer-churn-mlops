"""
Data drift detection module for production monitoring.

Detects when incoming production data has shifted away from the training
distribution, which signals that the model may be making predictions on
data it was not trained for — a leading indicator of model degradation.

Three types of drift are monitored:
    1. Numerical feature drift (PSI — Population Stability Index)
    2. Categorical feature drift (Chi-squared test)
    3. Prediction drift (rolling churn probability vs. training baseline)

All thresholds are read from configs/monitoring_config.yaml so they can
be tuned without code changes.

PSI interpretation:
    PSI < 0.10  → No significant shift
    PSI 0.10–0.25 → Moderate shift (monitor closely)
    PSI > 0.25    → Significant shift (alert + potential retrain trigger)

Public API:
    DriftDetector(reference_df, config)
        .compute_psi(current_df)         → dict of PSI scores per feature
        .compute_categorical_drift(current_df)  → dict of chi2 p-values
        .compute_prediction_drift(ref_proba, current_proba) → float
        .generate_report(current_df, current_proba) → DriftReport
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.config_loader import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# PSI constants — bin configuration for numerical features
# ---------------------------------------------------------------------------

# Number of quantile-based bins for PSI calculation.
# 10 bins is a standard industry choice — enough granularity to detect
# distributional shifts without being so fine-grained that noise triggers
# false alarms.
_PSI_BINS = 10

# Small constant added to bin proportions to prevent division by zero
# and log(0) when a bin has zero observations in reference or current data.
_PSI_EPSILON = 1e-6


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


@dataclass
class FeatureDriftResult:
    """Drift result for a single feature."""

    feature_name: str
    drift_score: float
    is_drifted: bool
    method: str  # "psi" or "chi_squared"
    threshold: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftReport:
    """Complete drift monitoring report."""

    timestamp: str
    reference_shape: tuple[int, int]
    current_shape: tuple[int, int]
    numerical_drift: list[FeatureDriftResult]
    categorical_drift: list[FeatureDriftResult]
    prediction_drift_score: Optional[float]
    prediction_is_drifted: bool
    overall_drift_detected: bool
    drifted_features: list[str]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to a flat dict for MLflow logging or JSON export."""
        result: dict[str, Any] = {
            "timestamp": self.timestamp,
            "reference_rows": self.reference_shape[0],
            "current_rows": self.current_shape[0],
            "prediction_drift_score": self.prediction_drift_score,
            "prediction_is_drifted": self.prediction_is_drifted,
            "overall_drift_detected": self.overall_drift_detected,
            "n_drifted_features": len(self.drifted_features),
            "drifted_features": self.drifted_features,
            "summary": self.summary,
        }

        # Add per-feature scores
        for feat_result in self.numerical_drift + self.categorical_drift:
            result[f"drift_{feat_result.feature_name}"] = round(
                feat_result.drift_score, 4
            )

        return result


# ---------------------------------------------------------------------------
# Drift Detector class
# ---------------------------------------------------------------------------


class DriftDetector:
    """
    Production data drift detector.

    Compares a reference dataset (training data snapshot) against current
    production data to detect distributional shifts.

    Usage:
        detector = DriftDetector(reference_df)
        report = detector.generate_report(current_df, current_proba)
    """

    def __init__(
        self,
        reference_df: pd.DataFrame,
        numerical_features: Optional[list[str]] = None,
        categorical_features: Optional[list[str]] = None,
    ) -> None:
        """
        Initialize the drift detector with a reference dataset.

        Args:
            reference_df: Training data snapshot to compare against.
            numerical_features: List of numerical column names to monitor.
                If None, auto-detects from DataFrame dtypes.
            categorical_features: List of categorical column names to monitor.
                If None, auto-detects from DataFrame dtypes.
        """
        self._reference_df = reference_df.copy()
        self._cfg = get_config()

        # Auto-detect feature types if not provided
        if numerical_features is None:
            self._numerical_features = reference_df.select_dtypes(
                include=[np.number]
            ).columns.tolist()
        else:
            self._numerical_features = numerical_features

        if categorical_features is None:
            self._categorical_features = reference_df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
        else:
            self._categorical_features = categorical_features

        logger.info(
            "DriftDetector initialized — reference shape: %s, "
            "numerical: %d, categorical: %d features.",
            reference_df.shape,
            len(self._numerical_features),
            len(self._categorical_features),
        )

    # -----------------------------------------------------------------------
    # PSI — Population Stability Index for numerical features
    # -----------------------------------------------------------------------

    @staticmethod
    def _compute_single_psi(
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = _PSI_BINS,
    ) -> float:
        """
        Compute PSI between two 1-D arrays using quantile-based binning.

        PSI = Σ (P_current - P_reference) × ln(P_current / P_reference)

        Quantile-based binning ensures that each bin has roughly equal
        representation in the reference data, which makes PSI robust to
        skewed distributions (e.g., TotalCharges).

        Args:
            reference: Reference (training) feature values.
            current: Current (production) feature values.
            n_bins: Number of quantile bins.

        Returns:
            PSI score (float ≥ 0). Higher = more drift.
        """
        # Create quantile-based bin edges from reference data
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(reference, quantiles)

        # Ensure unique bin edges (handles constant features)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 3:
            # Feature is near-constant in reference — can't compute PSI
            return 0.0

        # Compute bin proportions
        ref_hist, _ = np.histogram(reference, bins=bin_edges)
        cur_hist, _ = np.histogram(current, bins=bin_edges)

        ref_proportions = ref_hist / len(reference) + _PSI_EPSILON
        cur_proportions = cur_hist / len(current) + _PSI_EPSILON

        # PSI formula
        psi = np.sum(
            (cur_proportions - ref_proportions)
            * np.log(cur_proportions / ref_proportions)
        )

        return float(psi)

    def compute_psi(self, current_df: pd.DataFrame) -> list[FeatureDriftResult]:
        """
        Compute PSI for all numerical features.

        Args:
            current_df: Current production data.

        Returns:
            List of FeatureDriftResult, one per numerical feature.
        """
        psi_threshold = self._cfg.monitoring.psi_threshold
        results = []

        for feature in self._numerical_features:
            if feature not in current_df.columns:
                logger.warning(
                    "Feature '%s' missing from current data — skipping.", feature
                )
                continue

            ref_vals = self._reference_df[feature].dropna().values
            cur_vals = current_df[feature].dropna().values

            if len(ref_vals) == 0 or len(cur_vals) == 0:
                logger.warning("Feature '%s' has no valid values — skipping.", feature)
                continue

            psi_score = self._compute_single_psi(ref_vals, cur_vals)
            is_drifted = bool(psi_score > psi_threshold)

            result = FeatureDriftResult(
                feature_name=feature,
                drift_score=round(psi_score, 4),
                is_drifted=is_drifted,
                method="psi",
                threshold=psi_threshold,
                details={
                    "reference_mean": float(np.mean(ref_vals)),
                    "current_mean": float(np.mean(cur_vals)),
                    "reference_std": float(np.std(ref_vals)),
                    "current_std": float(np.std(cur_vals)),
                },
            )
            results.append(result)

            if is_drifted:
                logger.warning(
                    "PSI DRIFT detected for '%s': %.4f > %.4f threshold",
                    feature,
                    psi_score,
                    psi_threshold,
                )
            else:
                logger.debug("PSI for '%s': %.4f (no drift)", feature, psi_score)

        return results

    # -----------------------------------------------------------------------
    # Chi-squared test for categorical features
    # -----------------------------------------------------------------------

    def compute_categorical_drift(
        self, current_df: pd.DataFrame
    ) -> list[FeatureDriftResult]:
        """
        Compute chi-squared drift test for all categorical features.

        Tests whether the category distribution in current data differs
        significantly from the reference distribution. A low p-value
        indicates the distributions are different.

        Args:
            current_df: Current production data.

        Returns:
            List of FeatureDriftResult, one per categorical feature.
        """
        alpha = self._cfg.monitoring.chi_squared_alpha
        results = []

        for feature in self._categorical_features:
            if feature not in current_df.columns:
                logger.warning(
                    "Feature '%s' missing from current data — skipping.", feature
                )
                continue

            # Get value counts for both distributions
            ref_counts = self._reference_df[feature].value_counts()
            cur_counts = current_df[feature].value_counts()

            # Align categories — union of all categories in both datasets
            all_categories = sorted(set(ref_counts.index) | set(cur_counts.index))
            ref_aligned = np.array([ref_counts.get(c, 0) for c in all_categories])
            cur_aligned = np.array([cur_counts.get(c, 0) for c in all_categories])

            # Skip if either distribution is empty or has a single category
            if len(all_categories) < 2:
                continue
            if ref_aligned.sum() == 0 or cur_aligned.sum() == 0:
                continue

            # Chi-squared test: compare observed (current) against
            # expected (reference scaled to current's total count).
            expected = ref_aligned * (cur_aligned.sum() / ref_aligned.sum())
            # Avoid division by zero in expected
            expected = np.where(expected < _PSI_EPSILON, _PSI_EPSILON, expected)

            chi2_stat, p_value = stats.chisquare(cur_aligned, f_exp=expected)

            is_drifted = bool(p_value < alpha)

            result = FeatureDriftResult(
                feature_name=feature,
                drift_score=round(float(p_value), 6),
                is_drifted=is_drifted,
                method="chi_squared",
                threshold=alpha,
                details={
                    "chi2_statistic": round(float(chi2_stat), 4),
                    "p_value": round(float(p_value), 6),
                    "n_categories": len(all_categories),
                },
            )
            results.append(result)

            if is_drifted:
                logger.warning(
                    "Chi-squared DRIFT detected for '%s': p=%.6f < %.2f alpha",
                    feature,
                    p_value,
                    alpha,
                )
            else:
                logger.debug(
                    "Chi-squared for '%s': p=%.6f (no drift)", feature, p_value
                )

        return results

    # -----------------------------------------------------------------------
    # Prediction drift — rolling churn probability shift
    # -----------------------------------------------------------------------

    @staticmethod
    def compute_prediction_drift(
        reference_proba: np.ndarray,
        current_proba: np.ndarray,
    ) -> float:
        """
        Compute relative shift in mean churn probability.

        Formula: |mean(current) - mean(reference)| / mean(reference)

        A 15% relative shift (configurable via monitoring_config.yaml)
        indicates that the model's output distribution has changed
        meaningfully — either the data has shifted or the model is
        degrading.

        Args:
            reference_proba: Churn probabilities on training/reference data.
            current_proba: Churn probabilities on current production data.

        Returns:
            Relative drift magnitude (0.0 = no change, 0.15 = 15% shift).
        """
        ref_mean = float(np.mean(reference_proba))
        cur_mean = float(np.mean(current_proba))

        if ref_mean == 0:
            return 0.0

        drift = abs(cur_mean - ref_mean) / ref_mean
        return round(float(drift), 4)

    # -----------------------------------------------------------------------
    # Full report generation
    # -----------------------------------------------------------------------

    def generate_report(
        self,
        current_df: pd.DataFrame,
        reference_proba: Optional[np.ndarray] = None,
        current_proba: Optional[np.ndarray] = None,
    ) -> DriftReport:
        """
        Run all drift checks and produce a comprehensive report.

        Args:
            current_df: Current production data (features only, no target).
            reference_proba: Predicted probabilities on the reference dataset.
            current_proba: Predicted probabilities on the current dataset.

        Returns:
            DriftReport with all per-feature results and overall verdict.
        """
        logger.info(
            "Generating drift report — reference: %s, current: %s",
            self._reference_df.shape,
            current_df.shape,
        )

        # Run all checks
        numerical_drift = self.compute_psi(current_df)
        categorical_drift = self.compute_categorical_drift(current_df)

        # Prediction drift (optional — requires probabilities)
        pred_drift_score: Optional[float] = None
        pred_is_drifted = False

        if reference_proba is not None and current_proba is not None:
            pred_drift_score = self.compute_prediction_drift(
                reference_proba, current_proba
            )
            pred_threshold = self._cfg.monitoring.prediction_drift_threshold
            pred_is_drifted = bool(pred_drift_score > pred_threshold)
            logger.info(
                "Prediction drift: %.4f (threshold: %.2f, drifted: %s)",
                pred_drift_score,
                pred_threshold,
                pred_is_drifted,
            )

        # Collect drifted features
        drifted_features = [r.feature_name for r in numerical_drift if r.is_drifted]
        drifted_features += [r.feature_name for r in categorical_drift if r.is_drifted]

        overall_drift = len(drifted_features) > 0 or pred_is_drifted

        # Build summary message
        if overall_drift:
            summary = (
                f"DRIFT DETECTED — {len(drifted_features)} feature(s) drifted"
                f"{', prediction drift detected' if pred_is_drifted else ''}. "
                f"Drifted features: {drifted_features}. "
                "Consider investigating data pipeline and scheduling retraining."
            )
            logger.warning(summary)
        else:
            summary = (
                "No significant drift detected. "
                f"Monitored {len(numerical_drift)} numerical and "
                f"{len(categorical_drift)} categorical features."
            )
            logger.info(summary)

        report = DriftReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            reference_shape=self._reference_df.shape,
            current_shape=current_df.shape,
            numerical_drift=numerical_drift,
            categorical_drift=categorical_drift,
            prediction_drift_score=pred_drift_score,
            prediction_is_drifted=pred_is_drifted,
            overall_drift_detected=overall_drift,
            drifted_features=drifted_features,
            summary=summary,
        )

        return report
