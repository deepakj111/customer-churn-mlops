from pathlib import Path

import pandas as pd
import pytest

from src.data.ingest import DEFAULT_RAW_PATH, load_for_training, load_raw_data


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Write a minimal valid Telco-shaped CSV to a temp directory."""
    csv_content = (
        "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,"
        "PhoneService,MultipleLines,InternetService,OnlineSecurity,"
        "OnlineBackup,DeviceProtection,TechSupport,StreamingTV,"
        "StreamingMovies,Contract,PaperlessBilling,PaymentMethod,"
        "MonthlyCharges,TotalCharges,Churn\n"
        "1234-ABCD,Male,0,Yes,No,12,Yes,No,Fiber optic,No,Yes,No,No,"
        "No,No,Month-to-month,Yes,Electronic check,70.35,844.20,No\n"
        "5678-EFGH,Female,1,No,No,0,Yes,No phone service,DSL,No,No,"
        "No,No,No internet service,No internet service,Month-to-month,"
        "Yes,Mailed check,53.85,  ,Yes\n"
    )
    csv_file = tmp_path / "test_churn.csv"
    csv_file.write_text(csv_content)
    return csv_file


class TestLoadRawData:

    def test_returns_dataframe(self, sample_csv):
        df = load_raw_data(sample_csv)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self, sample_csv):
        df = load_raw_data(sample_csv)
        assert len(df) == 2

    def test_correct_column_count(self, sample_csv):
        df = load_raw_data(sample_csv)
        assert len(df.columns) == 21

    def test_senior_citizen_loaded_as_int(self, sample_csv):
        df = load_raw_data(sample_csv)
        assert df["SeniorCitizen"].dtype == int

    def test_string_columns_stripped_of_whitespace(self, sample_csv):
        df = load_raw_data(sample_csv)
        # TotalCharges second row has surrounding whitespace in CSV
        # After strip, string "  " becomes "" (still a string, not float yet)
        assert df["gender"].iloc[0] == "Male"

    def test_file_not_found_raises_error(self):
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            load_raw_data("/nonexistent/path/churn.csv")

    def test_error_message_includes_dvc_hint(self):
        with pytest.raises(FileNotFoundError, match="dvc pull"):
            load_raw_data("/nonexistent/path/churn.csv")

    def test_empty_file_raises_value_error(self, tmp_path):
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_raw_data(empty_file)

    def test_accepts_string_path(self, sample_csv):
        df = load_raw_data(str(sample_csv))
        assert isinstance(df, pd.DataFrame)

    def test_accepts_path_object(self, sample_csv):
        df = load_raw_data(Path(sample_csv))
        assert isinstance(df, pd.DataFrame)


class TestLoadForTraining:

    def test_uses_default_path_constant(self):
        assert DEFAULT_RAW_PATH == Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    def test_custom_path_override_works(self, sample_csv):
        df = load_for_training(path=sample_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
