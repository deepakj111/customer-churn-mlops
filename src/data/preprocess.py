"""
Data preprocessing layer.

Responsibility: Transform a validated raw DataFrame into a clean DataFrame
ready for feature engineering. This module assumes validate_raw_data()
has already run successfully on the input — it does not re-validate.

What this module does:
    - Encode the binary target column (Churn: Yes/No → 1/0)
    - Cast SeniorCitizen to string category (it's 0/1 int in raw data,
      but semantically it's a category like all other binary columns)
    - Remove the customerID column (no predictive signal, leaks identity)
    - Produce a clean split: features (X) and target (y)

What this module does NOT do:
    - Feature engineering (that's src/features/)
    - Scaling or encoding categorical columns (that's the sklearn Pipeline)
    - Any transformation that needs to be fitted on training data
      (those belong inside the sklearn Pipeline to prevent leakage)

Public API:
    preprocess(df)              → cleaned full DataFrame
    split_features_target(df)  → (X DataFrame, y Series)
    run_preprocessing(df)      → convenience: preprocess + split in one call
"""

import pandas as pd

from src.utils.config_loader import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all stateless cleaning transformations to a validated DataFrame.

    Stateless means: nothing is fitted on the data. Every transformation
    here applies the same rule regardless of the data's distribution.
    This means preprocess() can be called identically on training data,
    validation data, test data, and inference requests without any risk
    of data leakage.

    Transformations applied (in order):
        1. Encode target column: Churn Yes→1, No→0
        2. Cast SeniorCitizen from int to str ("0"/"1") so the downstream
           sklearn Pipeline treats it as a categorical column alongside
           all other binary string columns
        3. Drop customerID (defined in feature_config.yaml)

    Args:
        df: Validated raw DataFrame. Must have passed validate_raw_data().

    Returns:
        Cleaned DataFrame. Same row count as input, column count reduced
        by the number of dropped columns (customerID).
    """
    cfg = get_config()
    df = df.copy()

    logger.info("Starting preprocessing on DataFrame with shape %s.", df.shape)

    # Step 1 — Encode target column
    # Map Yes→1, No→0 for binary classification.
    # Do this before dropping any columns so the target is always present.
    target = cfg.features.target_column
    if target in df.columns:
        original_yes_count = (df[target] == "Yes").sum()
        df[target] = df[target].map({"Yes": 1, "No": 0}).astype(int)
        logger.info(
            "Encoded target '%s': %d positives (churn=1), %d negatives (churn=0).",
            target,
            original_yes_count,
            len(df) - original_yes_count,
        )
    else:
        # Target column absent is valid at inference time (we're predicting it)
        logger.debug("Target column '%s' not found — inference mode assumed.", target)

    # Step 2 — Cast SeniorCitizen to string
    # In the raw data SeniorCitizen is 0/1 integer, unlike every other
    # binary feature which is "Yes"/"No" string. Casting it to "0"/"1"
    # string means the downstream ColumnTransformer can apply the same
    # OneHotEncoder to ALL categorical columns uniformly, without needing
    # a special numeric branch just for this one column.
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
        logger.debug("Cast SeniorCitizen to string dtype.")

    # Step 3 — Drop columns with no predictive signal
    cols_to_drop = [col for col in cfg.features.features_to_drop if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info("Dropped columns: %s", cols_to_drop)

    logger.info("Preprocessing complete. Output shape: %s.", df.shape)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a preprocessed DataFrame into features (X) and target (y).

    The target column name is read from feature_config.yaml so it is
    never hardcoded in the training pipeline.

    Args:
        df: Preprocessed DataFrame that includes the target column.

    Returns:
        Tuple of (X, y) where:
            X: DataFrame with all feature columns (target excluded)
            y: Series with the binary target values (0 or 1)

    Raises:
        KeyError: If the target column is not present in the DataFrame.
    """
    cfg = get_config()
    target = cfg.features.target_column

    if target not in df.columns:
        raise KeyError(
            f"Target column '{target}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    X = df.drop(columns=[target])
    y = df[target]

    logger.info(
        "Split complete — X shape: %s, y shape: %s, positive rate: %.2f%%.",
        X.shape,
        y.shape,
        y.mean() * 100,
    )

    return X, y


def run_preprocessing(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Run the full preprocessing sequence in one call.

    Convenience function used by the training pipeline:
        df_clean = preprocess(df)
        X, y = split_features_target(df_clean)

    Combines both steps so the training pipeline reads as:
        X, y = run_preprocessing(validated_df)

    Args:
        df: Validated raw DataFrame.

    Returns:
        Tuple of (X, y) ready for feature engineering and model training.
    """
    df_clean = preprocess(df)
    return split_features_target(df_clean)
