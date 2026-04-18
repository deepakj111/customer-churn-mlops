"""
Causal Inference / Uplift Modeling Module.

While traditional ML predicts *who* will churn (P(Y=1|X)), Causal ML predicts
*who will change their mind* if given an intervention. We model the
Conditional Average Treatment Effect (CATE):
    CATE(x) = E[Y | X=x, T=1] - E[Y | X=x, T=0]

A negative CATE means the treatment (e.g. adding TechSupport) reduces churn probability.
This allows the business to target "Persuadables" and avoid "Sleeping Dogs".
"""

import numpy as np
import pandas as pd
from econml.metalearners import TLearner
from lightgbm import LGBMClassifier

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ChurnUpliftModel:
    """
    T-Learner Causal Model for Churn Prevention.

    The T-Learner (Two-Learner) fits two separate models:
        1. Model 0 (Control): predicting churn for untreated customers.
        2. Model 1 (Treated): predicting churn for treated customers.

    We use LightGBM classifiers as the base learners.
    """

    def __init__(self, random_state: int = 42):
        # We specify the models for T=0 and T=1
        self.model = TLearner(
            models=[
                LGBMClassifier(random_state=random_state, n_jobs=-1, verbose=-1),
                LGBMClassifier(random_state=random_state, n_jobs=-1, verbose=-1),
            ]
        )
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, T: pd.Series, y: pd.Series):
        """
        Fit the causal meta-learner.

        Args:
            X: Covariates (features) for all users.
            T: Treatment array (binary 0/1).
            y: Outcome array (binary 0/1 — 1 means churned).
        """
        logger.info(f"Fitting T-Learner on {len(X)} samples with {T.sum()} treated...")
        # econml expects Y, T, X
        self.model.fit(Y=y.values, T=T.values, X=X.values)
        self.is_fitted = True
        logger.info("T-Learner successfully fitted.")
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict the Conditional Average Treatment Effect (CATE).

        A negative value indicates the treatment decreases the probability of churn.

        Args:
            X: Covariates.

        Returns:
            np.ndarray of shape (N,) containing CATE estimates.
        """
        if not self.is_fitted:
            raise RuntimeError("Uplift model must be fitted before predicting.")

        # effect() returns the predicted CATE
        cate: np.ndarray = np.asarray(self.model.effect(X.values))
        return cate

    def segment_customers(
        self, X: pd.DataFrame, cate: np.ndarray, base_churn_prob: pd.Series
    ) -> pd.DataFrame:
        """
        Divide customers into classic uplift quadrants based on predictions.

        Quadrants:
            1. Persuadables: High base churn risk, negative CATE (treatment works).
               -> TARGET THESE!
            2. Sure Things: Low base churn risk, negative/neutral CATE.
               -> DO NOT TARGET (wasted money).
            3. Lost Causes: High base churn risk, positive/neutral CATE.
               -> DO NOT TARGET (treatment won't save them).
            4. Sleeping Dogs: Low base churn risk, positive CATE.
               -> AVOID AT ALL COSTS (intervention triggers them to leave).
        """
        df = X.copy()
        df["base_churn_prob"] = base_churn_prob.values
        df["uplift_cate"] = cate

        conditions = [
            (df["base_churn_prob"] >= 0.5)
            & (df["uplift_cate"] <= -0.05),  # Persuadables
            (df["base_churn_prob"] < 0.5) & (df["uplift_cate"] <= 0.0),  # Sure Things
            (df["base_churn_prob"] >= 0.5) & (df["uplift_cate"] > -0.05),  # Lost Causes
            (df["base_churn_prob"] < 0.5) & (df["uplift_cate"] > 0.0),  # Sleeping Dogs
        ]

        choices = [
            "Persuadable (Target)",
            "Sure Thing (Ignore)",
            "Lost Cause (Ignore)",
            "Sleeping Dog (Do Not Disturb)",
        ]
        df["uplift_segment"] = np.select(conditions, choices, default="Unknown")

        return df
