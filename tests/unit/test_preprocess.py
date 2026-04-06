import pandas as pd
import pytest

from src.data.preprocess import preprocess, run_preprocessing, split_features_target


@pytest.fixture
def valid_preprocessable_df() -> pd.DataFrame:
    """
    A minimal validated DataFrame ready for preprocessing.
    TotalCharges is already float (post fix_total_charges).
    """
    return pd.DataFrame(
        [
            {
                "customerID": "1234-ABCD",
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
                "Churn": "Yes",
            },
            {
                "customerID": "5678-EFGH",
                "gender": "Female",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 24,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 55.90,
                "TotalCharges": 1341.60,
                "Churn": "No",
            },
        ]
    )


class TestPreprocess:

    def test_returns_dataframe(self, valid_preprocessable_df):
        result = preprocess(valid_preprocessable_df)
        assert isinstance(result, pd.DataFrame)

    def test_churn_yes_encoded_as_1(self, valid_preprocessable_df):
        result = preprocess(valid_preprocessable_df)
        assert result["Churn"].iloc[0] == 1

    def test_churn_no_encoded_as_0(self, valid_preprocessable_df):
        result = preprocess(valid_preprocessable_df)
        assert result["Churn"].iloc[1] == 0

    def test_churn_column_dtype_is_int(self, valid_preprocessable_df):
        result = preprocess(valid_preprocessable_df)
        assert result["Churn"].dtype == int

    def test_senior_citizen_cast_to_string(self, valid_preprocessable_df):
        result = preprocess(valid_preprocessable_df)
        assert result["SeniorCitizen"].dtype == object
        assert result["SeniorCitizen"].iloc[0] == "0"
        assert result["SeniorCitizen"].iloc[1] == "1"

    def test_customer_id_dropped(self, valid_preprocessable_df):
        result = preprocess(valid_preprocessable_df)
        assert "customerID" not in result.columns

    def test_row_count_unchanged(self, valid_preprocessable_df):
        result = preprocess(valid_preprocessable_df)
        assert len(result) == len(valid_preprocessable_df)

    def test_does_not_mutate_input(self, valid_preprocessable_df):
        original_churn = valid_preprocessable_df["Churn"].iloc[0]
        preprocess(valid_preprocessable_df)
        assert valid_preprocessable_df["Churn"].iloc[0] == original_churn

    def test_inference_mode_no_churn_column(self, valid_preprocessable_df):
        df_no_target = valid_preprocessable_df.drop(columns=["Churn"])
        result = preprocess(df_no_target)
        assert "Churn" not in result.columns
        assert "customerID" not in result.columns

    def test_column_count_reduced_by_dropped_columns(self, valid_preprocessable_df):
        original_cols = len(valid_preprocessable_df.columns)
        result = preprocess(valid_preprocessable_df)
        # customerID is dropped, so one column less
        assert len(result.columns) == original_cols - 1


class TestSplitFeaturesTarget:

    def test_returns_tuple_of_two(self, valid_preprocessable_df):
        df = preprocess(valid_preprocessable_df)
        result = split_features_target(df)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_X_is_dataframe(self, valid_preprocessable_df):
        df = preprocess(valid_preprocessable_df)
        X, y = split_features_target(df)
        assert isinstance(X, pd.DataFrame)

    def test_y_is_series(self, valid_preprocessable_df):
        df = preprocess(valid_preprocessable_df)
        X, y = split_features_target(df)
        assert isinstance(y, pd.Series)

    def test_target_not_in_X(self, valid_preprocessable_df):
        df = preprocess(valid_preprocessable_df)
        X, y = split_features_target(df)
        assert "Churn" not in X.columns

    def test_y_has_correct_values(self, valid_preprocessable_df):
        df = preprocess(valid_preprocessable_df)
        X, y = split_features_target(df)
        assert list(y) == [1, 0]

    def test_X_and_y_have_same_row_count(self, valid_preprocessable_df):
        df = preprocess(valid_preprocessable_df)
        X, y = split_features_target(df)
        assert len(X) == len(y)

    def test_missing_target_raises_key_error(self, valid_preprocessable_df):
        df = valid_preprocessable_df.drop(columns=["Churn"])
        with pytest.raises(KeyError, match="Churn"):
            split_features_target(df)


class TestRunPreprocessing:

    def test_returns_X_and_y(self, valid_preprocessable_df):
        X, y = run_preprocessing(valid_preprocessable_df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_same_result_as_separate_calls(self, valid_preprocessable_df):
        X1, y1 = run_preprocessing(valid_preprocessable_df)

        df_clean = preprocess(valid_preprocessable_df)
        X2, y2 = split_features_target(df_clean)

        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)

    def test_positive_rate_is_correct(self, valid_preprocessable_df):
        X, y = run_preprocessing(valid_preprocessable_df)
        assert y.mean() == 0.5  # 1 churn out of 2 rows
