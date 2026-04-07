import pandas as pd
import pytest

from src.features.feature_store import (
    AVG_CHARGE_PER_SERVICE,
    CHARGE_TO_TENURE_RATIO,
    ENGINEERED_FEATURES,
    HAS_PROTECTION_BUNDLE,
    IS_FIBER_OPTIC,
    IS_MONTH_TO_MONTH,
    SERVICE_ADOPTION_COUNT,
    TENURE_GROUP,
    TENURE_GROUP_ORDER,
    _add_avg_charge_per_service,
    _add_charge_to_tenure_ratio,
    _add_has_protection_bundle,
    _add_is_fiber_optic,
    _add_is_month_to_month,
    _add_service_adoption_count,
    _add_tenure_group,
    engineer_features,
    get_binary_features,
    get_categorical_features,
    get_engineered_feature_names,
    get_feature_names,
    get_numerical_features,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_row() -> dict:
    """A fully valid preprocessed row — the happy-path baseline."""
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
        "Churn": 1,
    }


@pytest.fixture
def base_df(base_row) -> pd.DataFrame:
    return pd.DataFrame([base_row])


@pytest.fixture
def multi_row_df(base_row) -> pd.DataFrame:
    """Multi-row DataFrame covering diverse scenarios."""
    rows = [
        # Row 0 — Month-to-month, Fiber, new customer, high charge, no protection
        base_row,
        # Row 1 — Two year, DSL, long tenure, low charge, full protection bundle
        {
            **base_row,
            "tenure": 60,
            "Contract": "Two year",
            "InternetService": "DSL",
            "TechSupport": "Yes",
            "OnlineSecurity": "Yes",
            "MonthlyCharges": 45.0,
            "TotalCharges": 2700.0,
            "Churn": 0,
        },
        # Row 2 — One year, No internet, medium tenure, moderate charge
        {
            **base_row,
            "tenure": 24,
            "Contract": "One year",
            "InternetService": "No",
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "OnlineSecurity": "No internet service",
            "OnlineBackup": "No internet service",
            "DeviceProtection": "No internet service",
            "TechSupport": "No internet service",
            "StreamingTV": "No internet service",
            "StreamingMovies": "No internet service",
            "MonthlyCharges": 20.0,
            "TotalCharges": 480.0,
            "Churn": 0,
        },
        # Row 3 — New customer, tenure=0, edge case for ratio features
        {
            **base_row,
            "tenure": 0,
            "MonthlyCharges": 29.85,
            "TotalCharges": 29.85,
            "Churn": 1,
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tenure group tests
# ---------------------------------------------------------------------------


class TestAddTenureGroup:

    @pytest.mark.parametrize(
        "tenure,expected_group",
        [
            (0, "0-6m"),
            (3, "0-6m"),
            (6, "0-6m"),
            (7, "7-12m"),
            (12, "7-12m"),
            (13, "13-24m"),
            (24, "13-24m"),
            (25, "25-48m"),
            (48, "25-48m"),
            (49, "49+m"),
            (72, "49+m"),
        ],
    )
    def test_tenure_group_boundaries(self, base_df, tenure, expected_group):
        base_df["tenure"] = tenure
        result = _add_tenure_group(base_df)
        assert result[TENURE_GROUP].iloc[0] == expected_group

    def test_tenure_group_column_created(self, base_df):
        result = _add_tenure_group(base_df)
        assert TENURE_GROUP in result.columns

    def test_tenure_group_values_are_strings(self, base_df):
        result = _add_tenure_group(base_df)
        assert result[TENURE_GROUP].dtype == object

    def test_all_groups_valid(self, multi_row_df):
        result = _add_tenure_group(multi_row_df)
        assert result[TENURE_GROUP].isin(TENURE_GROUP_ORDER).all()

    def test_adds_exactly_one_column(self, base_df):
        original_col_count = len(base_df.columns)
        result = _add_tenure_group(base_df)
        assert len(result.columns) == original_col_count + 1


# ---------------------------------------------------------------------------
# is_month_to_month tests
# ---------------------------------------------------------------------------


class TestAddIsMonthToMonth:

    def test_month_to_month_returns_1(self, base_df):
        base_df["Contract"] = "Month-to-month"
        result = _add_is_month_to_month(base_df)
        assert result[IS_MONTH_TO_MONTH].iloc[0] == 1

    def test_one_year_returns_0(self, base_df):
        base_df["Contract"] = "One year"
        result = _add_is_month_to_month(base_df)
        assert result[IS_MONTH_TO_MONTH].iloc[0] == 0

    def test_two_year_returns_0(self, base_df):
        base_df["Contract"] = "Two year"
        result = _add_is_month_to_month(base_df)
        assert result[IS_MONTH_TO_MONTH].iloc[0] == 0

    def test_output_dtype_is_int(self, base_df):
        result = _add_is_month_to_month(base_df)
        assert result[IS_MONTH_TO_MONTH].dtype == int

    def test_only_0_and_1_in_output(self, multi_row_df):
        result = _add_is_month_to_month(multi_row_df)
        assert set(result[IS_MONTH_TO_MONTH].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# service_adoption_count tests
# ---------------------------------------------------------------------------


class TestAddServiceAdoptionCount:

    def test_count_increases_with_services(self, base_df):
        base_df["PhoneService"] = "Yes"
        base_df["MultipleLines"] = "Yes"
        base_df["InternetService"] = "Fiber optic"
        result = _add_service_adoption_count(base_df)
        # Phone(1) + MultipleLines(1) + Internet(1) + OnlineBackup(1) = 4
        assert result[SERVICE_ADOPTION_COUNT].iloc[0] == 4

    def test_no_services_returns_0(self, base_df):
        base_df["PhoneService"] = "No"
        base_df["MultipleLines"] = "No phone service"
        base_df["InternetService"] = "No"
        base_df["OnlineSecurity"] = "No internet service"
        base_df["OnlineBackup"] = "No internet service"
        base_df["DeviceProtection"] = "No internet service"
        base_df["TechSupport"] = "No internet service"
        base_df["StreamingTV"] = "No internet service"
        base_df["StreamingMovies"] = "No internet service"
        result = _add_service_adoption_count(base_df)
        assert result[SERVICE_ADOPTION_COUNT].iloc[0] == 0

    def test_all_services_returns_9(self, base_df):
        base_df["PhoneService"] = "Yes"
        base_df["MultipleLines"] = "Yes"
        base_df["InternetService"] = "Fiber optic"
        base_df["OnlineSecurity"] = "Yes"
        base_df["OnlineBackup"] = "Yes"
        base_df["DeviceProtection"] = "Yes"
        base_df["TechSupport"] = "Yes"
        base_df["StreamingTV"] = "Yes"
        base_df["StreamingMovies"] = "Yes"
        result = _add_service_adoption_count(base_df)
        assert result[SERVICE_ADOPTION_COUNT].iloc[0] == 9

    def test_count_is_non_negative(self, multi_row_df):
        result = _add_service_adoption_count(multi_row_df)
        assert (result[SERVICE_ADOPTION_COUNT] >= 0).all()

    def test_count_never_exceeds_9(self, multi_row_df):
        result = _add_service_adoption_count(multi_row_df)
        assert (result[SERVICE_ADOPTION_COUNT] <= 9).all()


# ---------------------------------------------------------------------------
# has_protection_bundle tests
# ---------------------------------------------------------------------------


class TestAddHasProtectionBundle:

    def test_both_yes_returns_1(self, base_df):
        base_df["TechSupport"] = "Yes"
        base_df["OnlineSecurity"] = "Yes"
        result = _add_has_protection_bundle(base_df)
        assert result[HAS_PROTECTION_BUNDLE].iloc[0] == 1

    def test_only_tech_support_returns_0(self, base_df):
        base_df["TechSupport"] = "Yes"
        base_df["OnlineSecurity"] = "No"
        result = _add_has_protection_bundle(base_df)
        assert result[HAS_PROTECTION_BUNDLE].iloc[0] == 0

    def test_only_online_security_returns_0(self, base_df):
        base_df["TechSupport"] = "No"
        base_df["OnlineSecurity"] = "Yes"
        result = _add_has_protection_bundle(base_df)
        assert result[HAS_PROTECTION_BUNDLE].iloc[0] == 0

    def test_neither_returns_0(self, base_df):
        base_df["TechSupport"] = "No internet service"
        base_df["OnlineSecurity"] = "No internet service"
        result = _add_has_protection_bundle(base_df)
        assert result[HAS_PROTECTION_BUNDLE].iloc[0] == 0

    def test_output_dtype_is_int(self, base_df):
        result = _add_has_protection_bundle(base_df)
        assert result[HAS_PROTECTION_BUNDLE].dtype == int


# ---------------------------------------------------------------------------
# charge_to_tenure_ratio tests
# ---------------------------------------------------------------------------


class TestAddChargeToTenureRatio:

    def test_formula_is_correct(self, base_df):
        base_df["MonthlyCharges"] = 70.0
        base_df["tenure"] = 12
        result = _add_charge_to_tenure_ratio(base_df)
        expected = 70.0 / (12 + 1)
        assert abs(result[CHARGE_TO_TENURE_RATIO].iloc[0] - expected) < 1e-9

    def test_tenure_zero_no_division_error(self, base_df):
        base_df["tenure"] = 0
        base_df["MonthlyCharges"] = 29.85
        result = _add_charge_to_tenure_ratio(base_df)
        expected = 29.85 / 1
        assert abs(result[CHARGE_TO_TENURE_RATIO].iloc[0] - expected) < 1e-9

    def test_higher_charge_higher_ratio(self, base_df):
        df_low = base_df.copy()
        df_high = base_df.copy()
        df_low["MonthlyCharges"] = 30.0
        df_high["MonthlyCharges"] = 90.0
        result_low = _add_charge_to_tenure_ratio(df_low)
        result_high = _add_charge_to_tenure_ratio(df_high)
        assert (
            result_high[CHARGE_TO_TENURE_RATIO].iloc[0]
            > result_low[CHARGE_TO_TENURE_RATIO].iloc[0]
        )

    def test_longer_tenure_lower_ratio(self, base_df):
        df_short = base_df.copy()
        df_long = base_df.copy()
        df_short["tenure"] = 1
        df_long["tenure"] = 60
        result_short = _add_charge_to_tenure_ratio(df_short)
        result_long = _add_charge_to_tenure_ratio(df_long)
        assert (
            result_short[CHARGE_TO_TENURE_RATIO].iloc[0]
            > result_long[CHARGE_TO_TENURE_RATIO].iloc[0]
        )

    def test_output_is_float(self, base_df):
        result = _add_charge_to_tenure_ratio(base_df)
        assert result[CHARGE_TO_TENURE_RATIO].dtype == float


# ---------------------------------------------------------------------------
# is_fiber_optic tests
# ---------------------------------------------------------------------------


class TestAddIsFiberOptic:

    def test_fiber_optic_returns_1(self, base_df):
        base_df["InternetService"] = "Fiber optic"
        result = _add_is_fiber_optic(base_df)
        assert result[IS_FIBER_OPTIC].iloc[0] == 1

    def test_dsl_returns_0(self, base_df):
        base_df["InternetService"] = "DSL"
        result = _add_is_fiber_optic(base_df)
        assert result[IS_FIBER_OPTIC].iloc[0] == 0

    def test_no_internet_returns_0(self, base_df):
        base_df["InternetService"] = "No"
        result = _add_is_fiber_optic(base_df)
        assert result[IS_FIBER_OPTIC].iloc[0] == 0

    def test_output_dtype_is_int(self, base_df):
        result = _add_is_fiber_optic(base_df)
        assert result[IS_FIBER_OPTIC].dtype == int


# ---------------------------------------------------------------------------
# avg_charge_per_service tests
# ---------------------------------------------------------------------------


class TestAddAvgChargePerService:

    def test_formula_is_correct(self, base_df):
        base_df["MonthlyCharges"] = 70.0
        base_df[SERVICE_ADOPTION_COUNT] = 6
        result = _add_avg_charge_per_service(base_df)
        expected = 70.0 / (6 + 1)
        assert abs(result[AVG_CHARGE_PER_SERVICE].iloc[0] - expected) < 1e-9

    def test_raises_if_service_count_missing(self, base_df):
        with pytest.raises(ValueError, match=SERVICE_ADOPTION_COUNT):
            _add_avg_charge_per_service(base_df)

    def test_zero_services_no_division_error(self, base_df):
        base_df[SERVICE_ADOPTION_COUNT] = 0
        base_df["MonthlyCharges"] = 20.0
        result = _add_avg_charge_per_service(base_df)
        assert result[AVG_CHARGE_PER_SERVICE].iloc[0] == 20.0

    def test_more_services_lower_ratio(self, base_df):
        df_few = base_df.copy()
        df_many = base_df.copy()
        df_few[SERVICE_ADOPTION_COUNT] = 1
        df_many[SERVICE_ADOPTION_COUNT] = 8
        result_few = _add_avg_charge_per_service(df_few)
        result_many = _add_avg_charge_per_service(df_many)
        assert (
            result_few[AVG_CHARGE_PER_SERVICE].iloc[0]
            > result_many[AVG_CHARGE_PER_SERVICE].iloc[0]
        )


# ---------------------------------------------------------------------------
# engineer_features (full pipeline) tests
# ---------------------------------------------------------------------------


class TestEngineerFeatures:

    def test_returns_dataframe(self, base_df):
        result = engineer_features(base_df)
        assert isinstance(result, pd.DataFrame)

    def test_all_engineered_columns_present(self, base_df):
        result = engineer_features(base_df)
        for col in ENGINEERED_FEATURES:
            assert col in result.columns, f"Missing engineered column: {col}"

    def test_row_count_unchanged(self, multi_row_df):
        result = engineer_features(multi_row_df)
        assert len(result) == len(multi_row_df)

    def test_original_columns_preserved(self, base_df):
        original_cols = set(base_df.columns)
        result = engineer_features(base_df)
        assert original_cols.issubset(set(result.columns))

    def test_column_count_increases_by_7(self, base_df):
        result = engineer_features(base_df)
        assert len(result.columns) == len(base_df.columns) + 7

    def test_does_not_mutate_input(self, base_df):
        original_shape = base_df.shape
        original_cols = list(base_df.columns)
        engineer_features(base_df)
        assert base_df.shape == original_shape
        assert list(base_df.columns) == original_cols

    def test_missing_required_column_raises_key_error(self, base_df):
        base_df = base_df.drop(columns=["tenure"])
        with pytest.raises(KeyError, match="tenure"):
            engineer_features(base_df)

    def test_no_nulls_in_engineered_features(self, multi_row_df):
        result = engineer_features(multi_row_df)
        for col in ENGINEERED_FEATURES:
            null_count = result[col].isna().sum()
            assert null_count == 0, f"Found {null_count} nulls in {col}"

    def test_high_risk_profile_produces_expected_values(self, base_df):
        # M2M + Fiber optic + new customer + high charge = the highest risk segment
        base_df["tenure"] = 3
        base_df["Contract"] = "Month-to-month"
        base_df["InternetService"] = "Fiber optic"
        base_df["MonthlyCharges"] = 85.0
        base_df["TechSupport"] = "No"
        base_df["OnlineSecurity"] = "No"
        result = engineer_features(base_df)

        assert result[IS_MONTH_TO_MONTH].iloc[0] == 1
        assert result[IS_FIBER_OPTIC].iloc[0] == 1
        assert result[HAS_PROTECTION_BUNDLE].iloc[0] == 0
        assert result[TENURE_GROUP].iloc[0] == "0-6m"
        assert result[CHARGE_TO_TENURE_RATIO].iloc[0] == 85.0 / 4

    def test_low_risk_profile_produces_expected_values(self, base_df):
        # Two year + DSL + long tenure + low charge + full bundle = lowest risk
        base_df["tenure"] = 60
        base_df["Contract"] = "Two year"
        base_df["InternetService"] = "DSL"
        base_df["MonthlyCharges"] = 45.0
        base_df["TechSupport"] = "Yes"
        base_df["OnlineSecurity"] = "Yes"
        result = engineer_features(base_df)

        assert result[IS_MONTH_TO_MONTH].iloc[0] == 0
        assert result[IS_FIBER_OPTIC].iloc[0] == 0
        assert result[HAS_PROTECTION_BUNDLE].iloc[0] == 1
        assert result[TENURE_GROUP].iloc[0] == "49+m"


# ---------------------------------------------------------------------------
# Feature name accessor tests
# ---------------------------------------------------------------------------


class TestFeatureNameAccessors:

    def test_get_feature_names_returns_list(self):
        assert isinstance(get_feature_names(), list)

    def test_get_feature_names_non_empty(self):
        assert len(get_feature_names()) > 0

    def test_get_categorical_features_returns_list(self):
        assert isinstance(get_categorical_features(), list)

    def test_get_numerical_features_returns_list(self):
        assert isinstance(get_numerical_features(), list)

    def test_get_binary_features_returns_list(self):
        assert isinstance(get_binary_features(), list)

    def test_engineered_features_in_full_feature_list(self):
        all_features = get_feature_names()
        for feat in get_engineered_feature_names():
            assert feat in all_features

    def test_no_overlap_between_categorical_and_numerical(self):
        cat = set(get_categorical_features())
        num = set(get_numerical_features())
        assert len(cat & num) == 0, f"Overlap found: {cat & num}"

    def test_accessor_returns_new_list_not_reference(self):
        list1 = get_feature_names()
        list2 = get_feature_names()
        list1.append("dummy")
        assert "dummy" not in list2

    def test_get_engineered_feature_names_has_exactly_7(self):
        assert len(get_engineered_feature_names()) == 7
