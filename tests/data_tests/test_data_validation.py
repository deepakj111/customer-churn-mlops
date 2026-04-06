"""
Data validation tests against the real Telco dataset on disk.

These tests only run if the dataset is present (managed by DVC).
They verify the actual downloaded CSV passes all schema checks after
the fix_total_charges pre-processor runs — confirming the known
blank-string bug is handled correctly in the real file.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data.validate import (
    fix_total_charges,
    get_validation_report,
    validate_raw_data,
)

DATASET_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# Skip all tests in this file if the CSV hasn't been pulled from DVC yet
pytestmark = pytest.mark.skipif(
    not DATASET_PATH.exists(),
    reason=(
        "Real dataset not found. Run 'poetry run dvc pull' to download it "
        "before running data validation tests."
    ),
)


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    """Load the actual Kaggle CSV once for all tests in this module."""
    return pd.read_csv(DATASET_PATH)


class TestRealDataset:

    def test_dataset_loads_without_error(self, raw_df):
        assert raw_df is not None
        assert len(raw_df) > 0

    def test_expected_row_count(self, raw_df):
        # The Telco dataset always has exactly 7,043 rows
        assert len(raw_df) == 7043

    def test_expected_column_count(self, raw_df):
        assert len(raw_df.columns) == 21

    def test_known_blank_total_charges_rows_exist_in_raw(self, raw_df):
        # Confirm the data quality issue actually exists in the file
        blank_mask = raw_df["TotalCharges"].astype(str).str.strip() == ""
        assert blank_mask.sum() == 11, (
            f"Expected 11 blank TotalCharges rows, found {blank_mask.sum()}. "
            "The dataset may have changed."
        )

    def test_blank_total_charges_rows_have_zero_tenure(self, raw_df):
        blank_mask = raw_df["TotalCharges"].astype(str).str.strip() == ""
        assert (raw_df[blank_mask]["tenure"] == 0).all()

    def test_full_dataset_passes_validation_after_fix(self, raw_df):
        # This is the end-to-end proof: real data → fix → validate → no errors
        validated = validate_raw_data(raw_df)
        assert len(validated) == 7043

    def test_churn_rate_is_approximately_26_percent(self, raw_df):
        fixed = fix_total_charges(raw_df)
        churn_rate = (fixed["Churn"] == "Yes").mean()
        # Actual rate is 26.54% — allow ±1% tolerance
        assert 0.25 < churn_rate < 0.28, f"Unexpected churn rate: {churn_rate:.2%}"

    def test_tenure_range_is_valid(self, raw_df):
        assert raw_df["tenure"].min() >= 0
        assert raw_df["tenure"].max() <= 100

    def test_no_nulls_in_critical_columns(self, raw_df):
        critical_cols = ["customerID", "tenure", "MonthlyCharges", "Churn"]
        for col in critical_cols:
            null_count = raw_df[col].isna().sum()
            assert null_count == 0, f"Found {null_count} nulls in {col}"

    def test_validation_report_on_real_data_is_valid(self, raw_df):
        report = get_validation_report(raw_df)
        assert report["is_valid"] is True
        assert report["row_count"] == 7043
        assert report["errors"] == []
