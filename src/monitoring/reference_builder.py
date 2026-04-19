"""
Reference data builder for drift monitoring.

Creates a snapshot of the training data that serves as the baseline
distribution for drift comparison. The reference dataset is saved to
data/reference/ in Parquet format for efficient storage and fast loading.

Why Parquet?
    - Preserves dtypes exactly (unlike CSV which loses int vs. float)
    - Column-oriented: fast partial reads (only load monitored features)
    - 3-5x smaller than CSV for this dataset
    - Industry standard for ML data storage

The reference dataset includes:
    - All feature columns (after preprocessing and feature engineering)
    - Reference predictions (churn probabilities from the current model)
    - Metadata (creation timestamp, model version, row count)

Public API:
    build_reference(pipeline, X, y)  → saves to data/reference/
    load_reference()                 → pd.DataFrame
    get_reference_metadata()         → dict
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Default paths for reference data storage
REFERENCE_DATA_PATH = Path("data/reference/reference_features.parquet")
REFERENCE_PROBA_PATH = Path("data/reference/reference_probabilities.npy")
REFERENCE_METADATA_PATH = Path("data/reference/reference_metadata.json")


def build_reference(
    X: pd.DataFrame,
    y: pd.Series,
    predictions: Optional[np.ndarray] = None,
    model_version: str = "unknown",
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Build and save the reference dataset for drift monitoring.

    Saves three files:
        - reference_features.parquet: Feature DataFrame
        - reference_probabilities.npy: Model predictions (if provided)
        - reference_metadata.json: Creation metadata

    Args:
        X: Feature DataFrame (from preprocessing pipeline).
        y: Target Series (for baseline churn rate tracking).
        predictions: Optional model predictions on the reference data.
        model_version: Identifier for the model that generated predictions.
        output_dir: Override the default data/reference/ directory.

    Returns:
        Path to the output directory.
    """
    if output_dir is None:
        output_dir = REFERENCE_DATA_PATH.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save features
    features_path = output_dir / "reference_features.parquet"
    X.to_parquet(features_path, index=False, engine="pyarrow")
    logger.info(
        "Reference features saved — shape: %s, path: %s", X.shape, features_path
    )

    # Save predictions (if provided)
    if predictions is not None:
        proba_path = output_dir / "reference_probabilities.npy"
        np.save(proba_path, predictions)
        logger.info("Reference probabilities saved — %d values.", len(predictions))

    # Save metadata
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "n_rows": len(X),
        "n_features": len(X.columns),
        "feature_names": X.columns.tolist(),
        "churn_rate": round(float(y.mean()), 4) if y is not None else None,
        "mean_prediction": (
            round(float(np.mean(predictions)), 4) if predictions is not None else None
        ),
    }

    metadata_path = output_dir / "reference_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Reference metadata saved — %s", metadata_path)

    return output_dir


def load_reference(
    data_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, Optional[np.ndarray]]:
    """
    Load the reference dataset from disk.

    Args:
        data_dir: Override the default data/reference/ directory.

    Returns:
        Tuple of (features_df, predictions_array).
        predictions_array is None if no probabilities were saved.

    Raises:
        FileNotFoundError: If the reference features file does not exist.
    """
    if data_dir is None:
        data_dir = REFERENCE_DATA_PATH.parent

    features_path = data_dir / "reference_features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(
            f"Reference data not found at {features_path.resolve()}. "
            "Run 'python scripts/save_reference_data.py' first."
        )

    df = pd.read_parquet(features_path, engine="pyarrow")
    logger.info("Reference features loaded — shape: %s", df.shape)

    # Load predictions if available
    proba_path = data_dir / "reference_probabilities.npy"
    predictions = None
    if proba_path.exists():
        predictions = np.load(proba_path)
        logger.info("Reference probabilities loaded — %d values.", len(predictions))

    return df, predictions


def get_reference_metadata(
    data_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Load reference dataset metadata.

    Args:
        data_dir: Override the default data/reference/ directory.

    Returns:
        Metadata dict with creation timestamp, model version, etc.

    Raises:
        FileNotFoundError: If the metadata file does not exist.
    """
    if data_dir is None:
        data_dir = REFERENCE_METADATA_PATH.parent

    metadata_path = data_dir / "reference_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Reference metadata not found at {metadata_path.resolve()}."
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.debug(
        "Reference metadata loaded — created at: %s", metadata.get("created_at")
    )
    return cast(dict[str, Any], metadata)
