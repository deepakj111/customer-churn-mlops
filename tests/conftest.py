"""
Shared test fixtures and configuration for the entire test suite.

Provides:
    - Automatic ConfigLoader singleton reset between tests
    - Shared DataFrame fixtures used by multiple test modules
"""

import pandas as pd
import pytest

from src.utils.config_loader import reset_config


@pytest.fixture(autouse=True)
def _reset_config_singleton():
    """
    Reset the ConfigLoader singleton before each test.

    Without this, a test that calls get_config("/tmp/custom") would pollute
    the singleton for all subsequent tests in the same process. autouse=True
    means every test in the suite gets this cleanup automatically.
    """
    reset_config()
    yield
    reset_config()


@pytest.fixture
def valid_raw_row() -> dict:
    """One perfectly valid raw Telco row — reusable across test modules."""
    return {
        "customerID": "1234-ABCDE",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.20,
        "Churn": "No",
    }


@pytest.fixture
def valid_raw_df(valid_raw_row) -> pd.DataFrame:
    """Single-row DataFrame representing a valid raw data record."""
    return pd.DataFrame([valid_raw_row])


@pytest.fixture
def preprocessed_base_row() -> dict:
    """
    One fully valid preprocessed row (post-preprocess.py).

    SeniorCitizen is cast to string, customerID is dropped,
    Churn is not present (features only).
    """
    return {
        "gender": "Male",
        "SeniorCitizen": "0",
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.20,
    }
