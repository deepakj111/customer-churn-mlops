"""
TabPFN Wrapper for Benchmarking.

TabPFN is a prior-data fitted network (a Neural Network trained on
synthetic datasets) that performs classification entirely in a
forward pass, making it extremely fast for small datasets without
needing traditional back-propagation tuning.

Because TabPFN natively struggles with strings and expects numeric
arrays, we wrap it cleanly to bridge with our existing data schemas.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

try:
    from tabpfn import TabPFNClassifier
except ImportError:
    TabPFNClassifier = None


class TabPFNWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper for TabPFNClassifier to ensure Scikit-Learn compatibility.

    Usage:
        model = TabPFNWrapper()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize the TabPFN wrapper.

        Args:
            device: Accelerator to use ('cuda' or 'cpu').
        """
        if TabPFNClassifier is None:
            raise ImportError("TabPFN is not installed. Run `poetry add tabpfn`")

        self.device = device
        # Create TabPFN internally
        self.model = TabPFNClassifier(
            device=self.device,
        )

    def fit(self, X: Any, y: Any) -> "TabPFNWrapper":
        """
        Pass-through wrapper ensuring data formats are array-compliant.
        TabPFN automatically scales and processes inputs internally.
        """
        # Convert DataFrames to numpy matrices, TabPFN demands it
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=np.float32)
        if hasattr(y, "values"):
            y = y.values

        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: Any) -> Any:
        # Convert DataFrames to numpy matrices
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=np.float32)
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> Any:
        # Convert DataFrames to numpy matrices
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=np.float32)
        return self.model.predict_proba(X)
