"""
Conformal Prediction Module — Uncertainty Quantification for Churn.

Traditional ML returns point estimates: "85% churn probability."
Conformal Prediction provides mathematically guaranteed prediction sets:
    "At 95% confidence, this customer WILL churn (set = {1})."
    "At 95% confidence, we are UNCERTAIN (set = {0, 1})."

This module wraps the existing calibrated LightGBM pipeline with MAPIE's
SplitConformalClassifier to produce valid prediction sets with
finite-sample coverage guarantees — no distributional assumptions
required.

Theory:
    Conformal prediction guarantees that the true label is contained
    in the prediction set with probability >= 1 - alpha, for ANY
    data distribution. This is a distribution-free, model-agnostic
    guarantee.

    We use the LAC (Least Ambiguous set-valued Classifier) method
    which minimises prediction set size while maintaining coverage.

References:
    - Vovk et al., "Algorithmic Learning in a Random World" (2005)
    - Angelopoulos & Bates, "A Gentle Introduction to Conformal
      Prediction" (2023), https://arxiv.org/abs/2107.07511
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from mapie.classification import SplitConformalClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Default artifacts directory
CONFORMAL_ARTIFACTS_DIR = Path("models/conformal")


class _DataFrameWrapper(BaseEstimator, ClassifierMixin):
    """
    Thin wrapper that converts numpy arrays back to DataFrames.

    MAPIE internally converts input data to numpy arrays, but our
    sklearn Pipeline requires pandas DataFrames with column names
    (because ColumnTransformer uses column name selection). This
    wrapper stores the column names from the calibration set and
    reconstructs DataFrames before calling the inner estimator.
    """

    def __init__(self, estimator: Any, columns: list[str]):
        self.estimator = estimator
        self.columns = columns

    def _to_df(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(np.asarray(X), columns=self.columns)

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> Any:
        """Pass-through fit (estimator is already fitted)."""
        return self

    def predict(self, X: Any) -> Any:
        """Predict with DataFrame conversion."""
        return self.estimator.predict(self._to_df(X))

    def predict_proba(self, X: Any) -> Any:
        """Predict probabilities with DataFrame conversion."""
        return self.estimator.predict_proba(self._to_df(X))

    @property
    def classes_(self) -> Any:
        """Expose classes_ from inner estimator."""
        return self.estimator.classes_

    def __sklearn_is_fitted__(self) -> bool:
        """Tell sklearn this estimator is already fitted."""
        return True


class ConformalChurnPredictor:
    """
    Conformal prediction wrapper for the churn classifier.

    Wraps a fitted sklearn Pipeline with MAPIE's
    SplitConformalClassifier to produce valid prediction sets
    at configurable confidence levels (default: 90% and 95%).

    Usage:
        conformal = ConformalChurnPredictor()
        conformal.calibrate(pipeline, X_cal, y_cal)
        result = conformal.predict_single(X_new)
    """

    def __init__(
        self,
        confidence_levels: list[float] | None = None,
    ):
        """
        Args:
            confidence_levels: List of confidence levels
                (e.g., [0.90, 0.95]). Default: [0.90, 0.95].
        """
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95]
        self.confidence_levels = confidence_levels
        self.mapie_clf: SplitConformalClassifier | None = None
        self.is_fitted = False

    def calibrate(
        self,
        pipeline: Any,
        X_cal: Any,
        y_cal: Any,
    ) -> "ConformalChurnPredictor":
        """
        Calibrate the conformal predictor on a held-out set.

        The calibration set must be HELD OUT from training.
        Using training data here invalidates the coverage
        guarantee — this is the most common mistake in
        conformal prediction implementations.

        Args:
            pipeline: A FITTED sklearn Pipeline.
            X_cal: Calibration features (held-out).
            y_cal: Calibration labels.

        Returns:
            self (for method chaining).
        """
        logger.info(
            "Calibrating conformal predictor on %d samples "
            "(confidence_levels=%s)...",
            len(X_cal),
            self.confidence_levels,
        )

        # Wrap the pipeline so MAPIE's numpy-based internals
        # don't break our column-name-dependent Pipeline.
        columns = list(X_cal.columns) if hasattr(X_cal, "columns") else []
        wrapped = _DataFrameWrapper(pipeline, columns)

        self.mapie_clf = SplitConformalClassifier(
            estimator=wrapped,
            confidence_level=self.confidence_levels,
            prefit=True,
        )
        # With prefit=True, skip fit() → conformalize() directly
        self.mapie_clf.conformalize(X_cal, y_cal)
        self.is_fitted = True

        logger.info("Conformal predictor calibrated successfully.")
        return self

    def predict(self, X: Any) -> dict:
        """
        Produce conformal prediction sets for new data.

        Returns a dict with results for each confidence level.

        Interpretation of prediction_sets[i]:
            [False, True]  -> set = {churn}    (confident)
            [True, False]  -> set = {no_churn} (confident)
            [True, True]   -> set = {both}     (uncertain)
            [False, False] -> empty set        (rare edge case)
        """
        if not self.is_fitted or self.mapie_clf is None:
            raise RuntimeError(
                "Conformal predictor must be calibrated "
                "before predicting. Call .calibrate() first."
            )

        _, prediction_sets = self.mapie_clf.predict_set(X)

        results: dict[str, dict[str, Any]] = {}
        for i, confidence in enumerate(self.confidence_levels):
            sets_at_level = prediction_sets[:, :, i]
            set_sizes = sets_at_level.sum(axis=1)
            results[str(confidence)] = {
                "prediction_sets": sets_at_level.tolist(),
                "set_sizes": set_sizes.tolist(),
            }

        return results

    def predict_single(self, X: Any) -> dict:
        """
        Produce a human-readable conformal result for one sample.

        Returns:
            {
                "confidence_90": {
                    "includes_churn": bool,
                    "includes_no_churn": bool,
                    "is_uncertain": bool,
                    "set_size": int,
                },
                "confidence_95": { ... },
            }
        """
        raw = self.predict(X)
        result = {}
        for confidence in self.confidence_levels:
            data = raw[str(confidence)]
            pred_set = data["prediction_sets"][0]
            includes_no_churn = bool(pred_set[0])
            includes_churn = bool(pred_set[1])
            set_size = int(data["set_sizes"][0])

            key = f"confidence_{int(confidence * 100)}"
            result[key] = {
                "includes_churn": includes_churn,
                "includes_no_churn": includes_no_churn,
                "is_uncertain": set_size == 2,
                "set_size": set_size,
            }
        return result

    def save(self, path: Path | None = None) -> Path:
        """Save the fitted conformal predictor to disk."""
        if path is None:
            path = CONFORMAL_ARTIFACTS_DIR
        path.mkdir(parents=True, exist_ok=True)

        model_path = path / "conformal_model.joblib"
        joblib.dump(self.mapie_clf, model_path)

        meta = {
            "confidence_levels": self.confidence_levels,
        }
        meta_path = path / "conformal_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Conformal artifacts saved to %s", path)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "ConformalChurnPredictor":
        """Load a fitted conformal predictor from disk."""
        if path is None:
            path = CONFORMAL_ARTIFACTS_DIR

        meta_path = path / "conformal_meta.json"
        model_path = path / "conformal_model.joblib"

        with open(meta_path) as f:
            meta = json.load(f)

        instance = cls(
            confidence_levels=meta["confidence_levels"],
        )
        instance.mapie_clf = joblib.load(model_path)
        instance.is_fitted = True

        logger.info("Conformal predictor loaded from %s", path)
        return instance
