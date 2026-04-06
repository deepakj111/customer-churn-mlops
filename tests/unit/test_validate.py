import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from src.data.validate import (
    fix_total_charges,
    get_validation_report,
    validate_inference_data,
    validate_raw_data,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_raw_row() -> dict:
    """One perfectly valid raw Telco row — the happy path baseline."""
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
def valid_inference_row() -> dict:
    """Valid row without customerID and Churn — as the API would receive it."""
    return {
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No internet service",
        "StreamingTV": "No internet service",
        "StreamingMovies": "No internet service",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 53.85,
        "TotalCharges": 108.15,
    }


@pytest.fixture
def valid_inference_df(valid_inference_row) -> pd.DataFrame:
    return pd.DataFrame([valid_inference_row])


# ---------------------------------------------------------------------------
# fix_total_charges tests
# ---------------------------------------------------------------------------


class TestFixTotalCharges:

    def test_blank_string_replaced_with_monthly_charges(self):
        df = pd.DataFrame(
            {
                "tenure": [0],
                "MonthlyCharges": [29.85],
                "TotalCharges": [""],
            }
        )
        result = fix_total_charges(df)
        assert result["TotalCharges"].iloc[0] == 29.85

    def test_whitespace_only_string_treated_as_blank(self):
        df = pd.DataFrame(
            {
                "tenure": [0],
                "MonthlyCharges": [55.0],
                "TotalCharges": ["   "],
            }
        )
        result = fix_total_charges(df)
        assert result["TotalCharges"].iloc[0] == 55.0

    def test_valid_numeric_string_preserved(self):
        df = pd.DataFrame(
            {
                "tenure": [24],
                "MonthlyCharges": [70.0],
                "TotalCharges": ["1680.00"],
            }
        )
        result = fix_total_charges(df)
        assert result["TotalCharges"].iloc[0] == 1680.0

    def test_already_float_column_unchanged(self):
        df = pd.DataFrame(
            {
                "tenure": [12],
                "MonthlyCharges": [60.0],
                "TotalCharges": [720.0],
            }
        )
        result = fix_total_charges(df)
        assert result["TotalCharges"].iloc[0] == 720.0

    def test_output_dtype_is_float(self):
        df = pd.DataFrame(
            {
                "tenure": [0, 5],
                "MonthlyCharges": [29.85, 55.0],
                "TotalCharges": ["", "275.0"],
            }
        )
        result = fix_total_charges(df)
        assert result["TotalCharges"].dtype in [np.float64, np.float32]

    def test_does_not_mutate_input_dataframe(self):
        df = pd.DataFrame(
            {
                "tenure": [0],
                "MonthlyCharges": [29.85],
                "TotalCharges": [""],
            }
        )
        original_value = df["TotalCharges"].iloc[0]
        fix_total_charges(df)
        assert df["TotalCharges"].iloc[0] == original_value

    def test_multiple_blank_rows_all_fixed(self):
        df = pd.DataFrame(
            {
                "tenure": [0, 0, 0],
                "MonthlyCharges": [20.0, 30.0, 40.0],
                "TotalCharges": ["", "", ""],
            }
        )
        result = fix_total_charges(df)
        assert result["TotalCharges"].isna().sum() == 0
        assert list(result["TotalCharges"]) == [20.0, 30.0, 40.0]


# ---------------------------------------------------------------------------
# validate_raw_data tests
# ---------------------------------------------------------------------------


class TestValidateRawData:

    def test_valid_dataframe_passes(self, valid_raw_df):
        result = validate_raw_data(valid_raw_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_total_charges_blank_string_fixed_and_passes(self):
        df = pd.DataFrame(
            [
                {
                    "customerID": "0000-BLANK",
                    "gender": "Male",
                    "SeniorCitizen": 0,
                    "Partner": "No",
                    "Dependents": "No",
                    "tenure": 0,
                    "PhoneService": "No",
                    "MultipleLines": "No phone service",
                    "InternetService": "No",
                    "OnlineSecurity": "No internet service",
                    "OnlineBackup": "No internet service",
                    "DeviceProtection": "No internet service",
                    "TechSupport": "No internet service",
                    "StreamingTV": "No internet service",
                    "StreamingMovies": "No internet service",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 29.85,
                    "TotalCharges": "",  # the known Telco data quality issue
                    "Churn": "No",
                }
            ]
        )
        result = validate_raw_data(df)
        assert result["TotalCharges"].iloc[0] == 29.85

    def test_invalid_gender_raises_schema_error(self, valid_raw_df):
        valid_raw_df["gender"] = "Unknown"
        with pytest.raises(pa.errors.SchemaError):
            validate_raw_data(valid_raw_df)

    def test_invalid_contract_raises_schema_error(self, valid_raw_df):
        valid_raw_df["Contract"] = "Weekly"
        with pytest.raises(pa.errors.SchemaError):
            validate_raw_data(valid_raw_df)

    def test_negative_tenure_raises_schema_error(self, valid_raw_df):
        valid_raw_df["tenure"] = -1
        with pytest.raises(pa.errors.SchemaError):
            validate_raw_data(valid_raw_df)

    def test_negative_monthly_charges_raises_schema_error(self, valid_raw_df):
        valid_raw_df["MonthlyCharges"] = -50.0
        with pytest.raises(pa.errors.SchemaError):
            validate_raw_data(valid_raw_df)

    def test_null_churn_raises_schema_error(self, valid_raw_df):
        valid_raw_df["Churn"] = None
        with pytest.raises(pa.errors.SchemaError):
            validate_raw_data(valid_raw_df)

    def test_invalid_payment_method_raises_schema_error(self, valid_raw_df):
        valid_raw_df["PaymentMethod"] = "Cash"
        with pytest.raises(pa.errors.SchemaError):
            validate_raw_data(valid_raw_df)

    def test_returns_dataframe_not_none(self, valid_raw_df):
        result = validate_raw_data(valid_raw_df)
        assert result is not None

    def test_extra_columns_allowed(self, valid_raw_df):
        valid_raw_df["extra_column"] = "some_value"
        # strict=False means extra columns don't cause failure
        result = validate_raw_data(valid_raw_df)
        assert "extra_column" in result.columns


# ---------------------------------------------------------------------------
# validate_inference_data tests
# ---------------------------------------------------------------------------


class TestValidateInferenceData:

    def test_valid_inference_row_passes(self, valid_inference_df):
        result = validate_inference_data(valid_inference_df)
        assert isinstance(result, pd.DataFrame)

    def test_invalid_internet_service_raises_error(self, valid_inference_df):
        valid_inference_df["InternetService"] = "Satellite"
        with pytest.raises(pa.errors.SchemaError):
            validate_inference_data(valid_inference_df)

    def test_senior_citizen_invalid_value_raises_error(self, valid_inference_df):
        valid_inference_df["SeniorCitizen"] = 2
        with pytest.raises(pa.errors.SchemaError):
            validate_inference_data(valid_inference_df)

    def test_zero_monthly_charges_raises_error(self, valid_inference_df):
        valid_inference_df["MonthlyCharges"] = 0.0
        with pytest.raises(pa.errors.SchemaError):
            validate_inference_data(valid_inference_df)


# ---------------------------------------------------------------------------
# get_validation_report tests
# ---------------------------------------------------------------------------


class TestGetValidationReport:

    def test_valid_data_returns_is_valid_true(self, valid_raw_df):
        report = get_validation_report(valid_raw_df)
        assert report["is_valid"] is True
        assert report["errors"] == []

    def test_invalid_data_returns_is_valid_false(self, valid_raw_df):
        valid_raw_df["gender"] = "Robot"
        report = get_validation_report(valid_raw_df)
        assert report["is_valid"] is False
        assert len(report["errors"]) > 0

    def test_report_contains_row_count(self, valid_raw_df):
        report = get_validation_report(valid_raw_df)
        assert report["row_count"] == 1

    def test_report_contains_shape(self, valid_raw_df):
        report = get_validation_report(valid_raw_df)
        assert isinstance(report["shape"], tuple)
        assert len(report["shape"]) == 2

    def test_errors_is_list(self, valid_raw_df):
        valid_raw_df["Contract"] = "Daily"
        report = get_validation_report(valid_raw_df)
        assert isinstance(report["errors"], list)
