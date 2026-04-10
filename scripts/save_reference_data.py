#!/usr/bin/env python
"""
Save the training data as the drift reference snapshot.

This script runs the full data pipeline (load → validate → preprocess →
feature engineer) and saves the result to data/reference/ for use by
the drift monitoring module.

Run this after each model retraining to update the reference baseline:
    poetry run python scripts/save_reference_data.py

Or via Makefile:
    make save-reference
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for `src.*` imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ingest import load_for_training  # noqa: E402
from src.data.preprocess import run_preprocessing  # noqa: E402
from src.data.validate import validate_raw_data  # noqa: E402
from src.features.feature_store import engineer_features  # noqa: E402
from src.monitoring.reference_builder import build_reference  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    """Build and save the reference dataset from the current training data."""
    logger.info("Building reference dataset for drift monitoring...")

    # Run the full data pipeline
    raw_df = load_for_training()
    validated_df = validate_raw_data(raw_df)
    X, y = run_preprocessing(validated_df)

    # Apply feature engineering (same as the Pipeline's FunctionTransformer)
    X_engineered = engineer_features(X)

    # Save reference
    output_dir = build_reference(
        X=X_engineered,
        y=y,
        predictions=None,  # Predictions will be added after model training
        model_version="manual",
    )

    logger.info("Reference dataset saved to: %s", output_dir.resolve())
    print(f"\n✅ Reference dataset saved to {output_dir.resolve()}")
    print(f"   Rows: {len(X_engineered)}")
    print(f"   Features: {len(X_engineered.columns)}")
    print(f"   Churn rate: {y.mean():.2%}")


if __name__ == "__main__":
    main()
