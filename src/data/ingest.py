"""
Data ingestion layer.

Responsibility: Load raw data from disk into a pandas DataFrame.
Nothing else. No cleaning, no feature engineering, no validation.

The separation between loading (here) and validating (validate.py)
and cleaning (preprocess.py) is intentional. Each module has exactly
one job. This makes failures easy to locate — if data doesn't load,
it's an ingest problem. If it loads but fails schema checks, it's
a validation problem. If it loads and validates but has dirty values,
it's a preprocessing problem.

Public API:
    load_raw_data(path)   → raw DataFrame, exactly as stored on disk
    load_for_training()   → convenience wrapper using the default data path
"""

from pathlib import Path

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_RAW_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """
    Load raw Telco churn data from a CSV file into a DataFrame.

    Does the minimum necessary:
        - Read the CSV with explicit dtype for SeniorCitizen (int)
        - Strip leading/trailing whitespace from string columns
        - Log shape and column count for traceability

    Does NOT clean, fix, validate, or transform anything.
    Call validate_raw_data() from src.data.validate next.

    Args:
        path: Path to the CSV file. Accepts str or Path.

    Returns:
        Raw DataFrame with all original columns intact.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If the file is empty or cannot be parsed as CSV.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path.resolve()}\n"
            f"Run 'poetry run dvc pull' to download the dataset from DagsHub."
        )

    if path.stat().st_size == 0:
        raise ValueError(f"Data file is empty: {path.resolve()}")

    logger.info("Loading raw data from: %s", path.resolve())

    df = pd.read_csv(
        path,
        dtype={"SeniorCitizen": int},
    )

    # Strip whitespace from all string columns.
    # The Telco CSV has no leading/trailing spaces in practice, but this
    # defensive step costs nothing and prevents subtle comparison bugs
    # (e.g. "Yes " != "Yes") that are nearly impossible to debug later.
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    logger.info(
        "Loaded raw data — shape: %s, columns: %d, memory: %.1f MB",
        df.shape,
        len(df.columns),
        df.memory_usage(deep=True).sum() / 1024**2,
    )

    return df


def load_for_training(path: str | Path = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """
    Convenience wrapper for the training pipeline.

    Loads the default Kaggle CSV from the DVC-managed data/raw/ directory.
    The training pipeline and notebooks call this instead of load_raw_data()
    directly so the default path is defined in one place.

    Args:
        path: Override the default path if needed (e.g. for testing).

    Returns:
        Raw DataFrame ready for validation and preprocessing.
    """
    logger.info("Loading training data from default path.")
    return load_raw_data(path)
