"""
FT-Transformer Wrapper for Benchmarking.

FT-Transformer (Feature Tokenizer Transformer) is a deep learning
architecture specialized for tabular data. It tokenizes numerical
and categorical datasets, passing them through self-attention layers
similar to NLP Transformers.

This wrapper utilizes the PyTorch Tabular framework mapping to create
an sklearn-compatible Estimator.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from src.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from pytorch_tabular import TabularModel
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
    from pytorch_tabular.models import FTTransformerConfig
except ImportError:
    TabularModel = None


class FTTransformerWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper for PyTorch-Tabular's FT-Transformer.

    Usage:
        model = FTTransformerWrapper(categorical_cols=[...], numerical_cols=[...])
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    """

    def __init__(
        self,
        categorical_cols: list[str] | None = None,
        numerical_cols: list[str] | None = None,
        epochs: int = 15,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
    ):
        """
        Initialize the FT-Transformer wrapper.
        """
        if TabularModel is None:
            raise ImportError(
                "PyTorch Tabular is not installed. Run `poetry add pytorch-tabular`"
            )

        self.categorical_cols = categorical_cols or []
        self.numerical_cols = numerical_cols or []
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tabular_model = None

    def _prepare_configs(self) -> Any:
        data_config = DataConfig(
            target=["target"],
            continuous_cols=self.numerical_cols,
            categorical_cols=self.categorical_cols,
            num_workers=0,
        )

        trainer_config = TrainerConfig(
            batch_size=self.batch_size,
            max_epochs=self.epochs,
            accelerator="auto",
            fast_dev_run=False,
            checkpoints_path=os.path.join("models", "ft_transformer"),
            early_stopping_patience=3,  # prevent massive overfitting
        )

        optimizer_config = OptimizerConfig(
            optimizer="AdamW",
            optimizer_params={"weight_decay": 1e-5},
            lr_scheduler="ReduceLROnPlateau",
            lr_scheduler_params={"patience": 2, "mode": "min"},
        )

        model_config = FTTransformerConfig(
            task="classification",
            learning_rate=self.learning_rate,
            num_attn_blocks=2,  # Keep it lightweight for this scale
            num_heads=4,
            attn_dropout=0.1,
            ff_dropout=0.1,
        )

        return data_config, trainer_config, optimizer_config, model_config

    def _ensure_dataframe(self, X: Any, y: Any = None) -> pd.DataFrame:
        """Converts inputs to a fused DataFrame, which PyTorch Tabular expects."""
        if not isinstance(X, pd.DataFrame):
            # Try to build generic numbered columns if none exist
            cols = self.categorical_cols + self.numerical_cols
            if not cols:
                cols = [f"col_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=cols)
        else:
            X_df = X.copy()

        if y is not None:
            if hasattr(y, "values"):
                y = y.values
            X_df["target"] = y
        return X_df

    def fit(self, X: Any, y: Any) -> "FTTransformerWrapper":
        """
        Fits the FT-Transformer modeling pipeline.
        """
        # Auto-detect columns if not specified
        if isinstance(X, pd.DataFrame) and not (
            self.categorical_cols or self.numerical_cols
        ):
            self.categorical_cols = list(
                X.select_dtypes(include=["object", "category", "bool"]).columns
            )
            self.numerical_cols = list(
                X.select_dtypes(exclude=["object", "category", "bool"]).columns
            )

        data_config, trainer_config, optimizer_config, model_config = (
            self._prepare_configs()
        )

        self.tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
            verbose=False,
            suppress_lightning_logger=True,
        )

        train_df = self._ensure_dataframe(X, y)

        logger.info(
            "Fitting FT-Transformer on %d samples, %d features " "(%d cat, %d num).",
            train_df.shape[0],
            len(self.categorical_cols) + len(self.numerical_cols),
            len(self.categorical_cols),
            len(self.numerical_cols),
        )

        assert self.tabular_model is not None, "TabularModel not initialized"
        self.tabular_model.fit(train=train_df)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X: Any) -> Any:
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X: Any) -> Any:
        df = self._ensure_dataframe(X)
        assert self.tabular_model is not None, "Must call fit() before predict"
        pred_df = self.tabular_model.predict(df, ret_logits=False)

        preds = np.zeros((len(df), 2))
        prob_cols = [c for c in pred_df.columns if "probability" in c]
        if len(prob_cols) == 2:
            preds[:, 0] = pred_df[prob_cols[0]].values
            preds[:, 1] = pred_df[prob_cols[1]].values
        else:
            # Fallback if probability names aren't straightforward
            preds[:, 1] = pred_df.iloc[:, -1].values
            preds[:, 0] = 1 - preds[:, 1]
        return preds
