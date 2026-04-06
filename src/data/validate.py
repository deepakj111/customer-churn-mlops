"""
Data validation layer for the Churn MLOps pipeline.

Every DataFrame that enters the training pipeline or inference API
passes through these validators before touching any model logic.
Catching schema problems here — at the boundary — prevents silent
corruption from propagating into features, training, or predictions.

Two validators are defined:
    RawDataSchema     — validates the original Kaggle CSV as loaded from disk
    InferenceSchema   — validates a single customer's features at API request time

Known data quality issue in the Telco dataset:
    TotalCharges contains 11 blank strings ("") for customers with tenure == 0.
    These are NOT null — they are empty strings that pandas reads as object dtype.
    fix_total_charges() handles this before schema validation runs.
"""

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Known valid category sets — defined once, reused in both schemas
# ---------------------------------------------------------------------------

YES_NO = {"Yes", "No"}
YES_NO_NO_PHONE = {"Yes", "No", "No phone service"}
YES_NO_NO_INTERNET = {"Yes", "No", "No internet service"}
INTERNET_SERVICE_TYPES = {"DSL", "Fiber optic", "No"}
CONTRACT_TYPES = {"Month-to-month", "One year", "Two year"}
PAYMENT_METHODS = {
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
}


# ---------------------------------------------------------------------------
# Raw data schema — validates the CSV exactly as loaded from disk
# ---------------------------------------------------------------------------

RawDataSchema = DataFrameSchema(
    columns={
        "customerID": Column(
            str,
            nullable=False,
            description="Unique customer identifier. Not used in model training.",
        ),
        "gender": Column(
            str,
            checks=Check.isin({"Male", "Female"}),
            nullable=False,
            description="Customer gender.",
        ),
        "SeniorCitizen": Column(
            int,
            checks=Check.isin({0, 1}),
            nullable=False,
            description="1 if the customer is a senior citizen, else 0.",
        ),
        "Partner": Column(
            str,
            checks=Check.isin(YES_NO),
            nullable=False,
        ),
        "Dependents": Column(
            str,
            checks=Check.isin(YES_NO),
            nullable=False,
        ),
        "tenure": Column(
            int,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(100),
            ],
            nullable=False,
            description="Number of months the customer has been with the company.",
        ),
        "PhoneService": Column(
            str,
            checks=Check.isin(YES_NO),
            nullable=False,
        ),
        "MultipleLines": Column(
            str,
            checks=Check.isin(YES_NO_NO_PHONE),
            nullable=False,
        ),
        "InternetService": Column(
            str,
            checks=Check.isin(INTERNET_SERVICE_TYPES),
            nullable=False,
        ),
        "OnlineSecurity": Column(
            str,
            checks=Check.isin(YES_NO_NO_INTERNET),
            nullable=False,
        ),
        "OnlineBackup": Column(
            str,
            checks=Check.isin(YES_NO_NO_INTERNET),
            nullable=False,
        ),
        "DeviceProtection": Column(
            str,
            checks=Check.isin(YES_NO_NO_INTERNET),
            nullable=False,
        ),
        "TechSupport": Column(
            str,
            checks=Check.isin(YES_NO_NO_INTERNET),
            nullable=False,
        ),
        "StreamingTV": Column(
            str,
            checks=Check.isin(YES_NO_NO_INTERNET),
            nullable=False,
        ),
        "StreamingMovies": Column(
            str,
            checks=Check.isin(YES_NO_NO_INTERNET),
            nullable=False,
        ),
        "Contract": Column(
            str,
            checks=Check.isin(CONTRACT_TYPES),
            nullable=False,
        ),
        "PaperlessBilling": Column(
            str,
            checks=Check.isin(YES_NO),
            nullable=False,
        ),
        "PaymentMethod": Column(
            str,
            checks=Check.isin(PAYMENT_METHODS),
            nullable=False,
        ),
        "MonthlyCharges": Column(
            float,
            checks=Check.greater_than(0),
            nullable=False,
            description="Current monthly charge amount in dollars.",
        ),
        "TotalCharges": Column(
            float,
            checks=Check.greater_than_or_equal_to(0),
            nullable=False,
            description=(
                "Total amount charged over tenure. "
                "Must call fix_total_charges() before validating — "
                "the raw CSV contains blank strings for tenure==0 rows."
            ),
        ),
        "Churn": Column(
            str,
            checks=Check.isin(YES_NO),
            nullable=False,
            description="Target variable. 'Yes' = customer churned.",
        ),
    },
    # Allow extra columns to pass through without error.
    # This protects against future dataset versions adding columns
    # without breaking the validation pipeline.
    strict=False,
    name="RawDataSchema",
)


# ---------------------------------------------------------------------------
# Inference schema — validates a SINGLE customer's features at API request time.
# The Churn column is absent (that's what we're predicting).
# customerID is optional — the API may or may not receive it.
# ---------------------------------------------------------------------------

InferenceSchema = DataFrameSchema(
    columns={
        "gender": Column(str, checks=Check.isin({"Male", "Female"}), nullable=False),
        "SeniorCitizen": Column(int, checks=Check.isin({0, 1}), nullable=False),
        "Partner": Column(str, checks=Check.isin(YES_NO), nullable=False),
        "Dependents": Column(str, checks=Check.isin(YES_NO), nullable=False),
        "tenure": Column(
            int,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(100),
            ],
            nullable=False,
        ),
        "PhoneService": Column(str, checks=Check.isin(YES_NO), nullable=False),
        "MultipleLines": Column(
            str, checks=Check.isin(YES_NO_NO_PHONE), nullable=False
        ),
        "InternetService": Column(
            str, checks=Check.isin(INTERNET_SERVICE_TYPES), nullable=False
        ),
        "OnlineSecurity": Column(
            str, checks=Check.isin(YES_NO_NO_INTERNET), nullable=False
        ),
        "OnlineBackup": Column(
            str, checks=Check.isin(YES_NO_NO_INTERNET), nullable=False
        ),
        "DeviceProtection": Column(
            str, checks=Check.isin(YES_NO_NO_INTERNET), nullable=False
        ),
        "TechSupport": Column(
            str, checks=Check.isin(YES_NO_NO_INTERNET), nullable=False
        ),
        "StreamingTV": Column(
            str, checks=Check.isin(YES_NO_NO_INTERNET), nullable=False
        ),
        "StreamingMovies": Column(
            str, checks=Check.isin(YES_NO_NO_INTERNET), nullable=False
        ),
        "Contract": Column(str, checks=Check.isin(CONTRACT_TYPES), nullable=False),
        "PaperlessBilling": Column(str, checks=Check.isin(YES_NO), nullable=False),
        "PaymentMethod": Column(
            str, checks=Check.isin(PAYMENT_METHODS), nullable=False
        ),
        "MonthlyCharges": Column(float, checks=Check.greater_than(0), nullable=False),
        "TotalCharges": Column(
            float,
            checks=Check.greater_than_or_equal_to(0),
            nullable=False,
        ),
    },
    strict=False,
    name="InferenceSchema",
)


# ---------------------------------------------------------------------------
# Pre-validation fixers — run BEFORE schema validation
# ---------------------------------------------------------------------------


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the known blank-string issue in TotalCharges before validation.

    The raw Telco CSV contains 11 rows where TotalCharges is an empty
    string "". These are new customers (tenure == 0) who have not yet
    been billed a full month. Pandas reads the column as object dtype
    because of these strings.

    Fix applied:
        1. Replace blank strings with NaN so pd.to_numeric can handle them.
        2. Convert the column to float.
        3. Impute NaN values with MonthlyCharges (logical: first month = monthly rate).
        4. Log how many rows were fixed so the fix is always auditable.

    This is a documented, intentional imputation — not a hidden hack.
    The decision is captured here so any future engineer can read why.

    Args:
        df: Raw DataFrame loaded from the Telco CSV.

    Returns:
        DataFrame with TotalCharges as a clean float column.
    """
    df = df.copy()

    original_dtype = df["TotalCharges"].dtype

    # Replace blank strings and whitespace-only strings with NaN
    df["TotalCharges"] = df["TotalCharges"].replace(r"^\s*$", pd.NA, regex=True)

    # Convert to numeric — any remaining non-numeric values become NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    null_count = df["TotalCharges"].isna().sum()

    if null_count > 0:
        logger.info(
            "fix_total_charges: found %d blank TotalCharges rows (dtype was %s). "
            "Imputing with MonthlyCharges.",
            null_count,
            original_dtype,
        )
        df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
    else:
        logger.debug("fix_total_charges: no blank TotalCharges found.")

    return df


# ---------------------------------------------------------------------------
# Public validation functions — call these from pipelines and tests
# ---------------------------------------------------------------------------


def validate_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the raw Telco DataFrame against RawDataSchema.

    Applies fix_total_charges() first, then runs Pandera validation.
    Returns the validated (and fixed) DataFrame on success.
    Raises pandera.errors.SchemaError on any violation.

    This is the entry point for the training pipeline's data layer.
    Call this immediately after loading the CSV — before any other
    transformation touches the data.

    Args:
        df: Raw DataFrame loaded from the Telco CSV.

    Returns:
        Validated DataFrame with TotalCharges fixed to float.

    Raises:
        pandera.errors.SchemaError: If any column fails its checks.
    """
    logger.info("Starting raw data validation on DataFrame with shape %s.", df.shape)

    df = fix_total_charges(df)
    validated_df = RawDataSchema.validate(df)

    logger.info("Raw data validation passed. Shape: %s.", validated_df.shape)
    return validated_df


def validate_inference_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a customer feature DataFrame before running inference.

    Used by the FastAPI predictor to ensure incoming request data
    is structurally sound before the sklearn Pipeline processes it.
    This is the inference-time equivalent of validate_raw_data().

    Args:
        df: DataFrame built from the API request body (one or more rows).

    Returns:
        Validated DataFrame.

    Raises:
        pandera.errors.SchemaError: If any field fails its checks.
    """
    logger.info(
        "Starting inference data validation on DataFrame with shape %s.", df.shape
    )

    validated_df = InferenceSchema.validate(df)

    logger.info("Inference data validation passed.")
    return validated_df


def get_validation_report(df: pd.DataFrame) -> dict:
    """
    Run validation and return a structured report instead of raising an exception.

    Useful for monitoring and alerting pipelines where you want to collect
    ALL violations at once (not just fail on the first one) and log or
    report them without crashing the pipeline.

    Args:
        df: DataFrame to validate.

    Returns:
        A dict with keys:
            "is_valid" (bool): True if all checks passed.
            "row_count" (int): Number of rows in the input.
            "errors" (list): List of error message strings. Empty if valid.
            "shape" (tuple): Shape of the input DataFrame.
    """
    df = fix_total_charges(df)

    report = {
        "is_valid": False,
        "row_count": len(df),
        "shape": df.shape,
        "errors": [],
    }

    try:
        RawDataSchema.validate(df, lazy=True)
        report["is_valid"] = True
        logger.info("Validation report: PASSED for %d rows.", len(df))
    except pa.errors.SchemaErrors as exc:
        error_messages = exc.failure_cases["failure_case"].astype(str).tolist()
        report["errors"] = error_messages
        logger.warning(
            "Validation report: FAILED with %d error(s).", len(error_messages)
        )

    return report
