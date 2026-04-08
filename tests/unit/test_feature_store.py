import pandas as pd
import pytest

from src.features.feature_store import (  # Constants; Functions
    AVG_CHARGE_PER_SERVICE,
    CHARGE_TO_TENURE_RATIO,
    CHARGES_GAP,
    COMPOSITE_CHURN_RISK,
    CONTRACT_NUMERIC,
    ENGINEERED_FEATURES,
    FIBER_NO_SECURITY,
    HAS_FAMILY,
    HAS_FULL_STREAMING,
    HAS_INTERNET,
    HAS_NO_INTERNET_ADDONS,
    HAS_PROTECTION_BUNDLE,
    INTERNET_ADDON_COUNT,
    IS_AUTO_PAY,
    IS_COMMITTED_CUSTOMER,
    IS_ELECTRONIC_CHECK,
    IS_FIBER_OPTIC,
    IS_HIGH_MONTHLY_CHARGE,
    IS_ISOLATED,
    IS_MONTH_TO_MONTH,
    M2M_AND_FIBER,
    MONTHLY_CHARGE_BIN,
    MONTHLY_CHARGE_BIN_ORDER,
    NEW_HIGH_VALUE_CUSTOMER,
    PAPERLESS_AND_ECHECK,
    SECURITY_SERVICE_COUNT,
    SENIOR_NO_SUPPORT,
    SERVICE_ADOPTION_COUNT,
    STREAMING_COUNT,
    TENURE_GROUP,
    TENURE_GROUP_ORDER,
    engineer_features,
    get_binary_features,
    get_categorical_features,
    get_engineered_feature_names,
    get_feature_names,
    get_numerical_features,
)

# ---------------------------------------------------------------------------
# Shared test fixture — one fully valid preprocessed row
# ---------------------------------------------------------------------------


@pytest.fixture
def base_row() -> dict:
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


@pytest.fixture
def base_df(base_row) -> pd.DataFrame:
    return pd.DataFrame([base_row])


@pytest.fixture
def multi_row_df(base_row) -> pd.DataFrame:
    senior_auto_pay = {
        **base_row,
        "SeniorCitizen": "1",
        "TechSupport": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "Contract": "Two year",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 0,
        "TotalCharges": 0.0,
    }
    return pd.DataFrame([base_row, senior_auto_pay])


# ---------------------------------------------------------------------------
# Test: engineer_features output shape
# ---------------------------------------------------------------------------


class TestEngineerFeaturesShape:

    def test_row_count_unchanged(self, multi_row_df):
        result = engineer_features(multi_row_df)
        assert len(result) == len(multi_row_df)

    def test_adds_exactly_28_new_columns(self, base_df):
        result = engineer_features(base_df)
        new_cols = [c for c in result.columns if c not in base_df.columns]
        assert len(new_cols) == 28

    def test_all_engineered_feature_names_present(self, base_df):
        result = engineer_features(base_df)
        for feat in ENGINEERED_FEATURES:
            assert feat in result.columns, f"Missing engineered feature: {feat}"

    def test_does_not_mutate_input(self, base_df):
        original_cols = list(base_df.columns)
        engineer_features(base_df)
        assert list(base_df.columns) == original_cols

    def test_missing_required_column_raises_key_error(self, base_df):
        df_bad = base_df.drop(columns=["tenure"])
        with pytest.raises(KeyError, match="tenure"):
            engineer_features(df_bad)

    def test_engineered_features_constant_has_28_items(self):
        assert len(ENGINEERED_FEATURES) == 28


# ---------------------------------------------------------------------------
# Test: Original 7 features
# ---------------------------------------------------------------------------


class TestOriginalSevenFeatures:

    def test_tenure_group_values_are_valid(self, multi_row_df):
        result = engineer_features(multi_row_df)
        valid = set(TENURE_GROUP_ORDER)
        assert set(result[TENURE_GROUP].unique()).issubset(valid)

    def test_tenure_group_7_12_for_tenure_12(self, base_df):
        result = engineer_features(base_df)
        assert result[TENURE_GROUP].iloc[0] == "7-12m"

    def test_tenure_group_0_6m_for_tenure_0(self, multi_row_df):
        result = engineer_features(multi_row_df)
        assert result[TENURE_GROUP].iloc[1] == "0-6m"

    def test_is_month_to_month_flag(self, base_df):
        result = engineer_features(base_df)
        assert result[IS_MONTH_TO_MONTH].iloc[0] == 1

    def test_is_month_to_month_zero_for_two_year(self, multi_row_df):
        result = engineer_features(multi_row_df)
        assert result[IS_MONTH_TO_MONTH].iloc[1] == 0

    def test_service_adoption_count_nonnegative(self, multi_row_df):
        result = engineer_features(multi_row_df)
        assert (result[SERVICE_ADOPTION_COUNT] >= 0).all()

    def test_service_adoption_count_max_nine(self, multi_row_df):
        result = engineer_features(multi_row_df)
        assert (result[SERVICE_ADOPTION_COUNT] <= 9).all()

    def test_has_protection_bundle_requires_both(self, base_df):
        result = engineer_features(base_df)
        assert result[HAS_PROTECTION_BUNDLE].iloc[0] == 0

    def test_charge_to_tenure_ratio_positive(self, multi_row_df):
        result = engineer_features(multi_row_df)
        assert (result[CHARGE_TO_TENURE_RATIO] > 0).all()

    def test_is_fiber_optic_flag(self, base_df):
        result = engineer_features(base_df)
        assert result[IS_FIBER_OPTIC].iloc[0] == 1

    def test_avg_charge_per_service_positive(self, base_df):
        result = engineer_features(base_df)
        assert result[AVG_CHARGE_PER_SERVICE].iloc[0] > 0


# ---------------------------------------------------------------------------
# Test: Group A — Demographic
# ---------------------------------------------------------------------------


class TestGroupADemographic:

    def test_has_family_when_partner_yes(self, base_df):
        result = engineer_features(base_df)
        assert result[HAS_FAMILY].iloc[0] == 1

    def test_is_isolated_when_no_family(self, multi_row_df):
        result = engineer_features(multi_row_df)
        assert result[IS_ISOLATED].iloc[1] == 1

    def test_has_family_and_is_isolated_are_complements(self, multi_row_df):
        result = engineer_features(multi_row_df)
        for _, row in result.iterrows():
            assert row[HAS_FAMILY] + row[IS_ISOLATED] == 1

    def test_senior_no_support_for_senior_without_tech(self, multi_row_df):
        result = engineer_features(multi_row_df)
        assert result[SENIOR_NO_SUPPORT].iloc[1] == 1

    def test_senior_no_support_zero_for_non_senior(self, base_df):
        result = engineer_features(base_df)
        assert result[SENIOR_NO_SUPPORT].iloc[0] == 0


# ---------------------------------------------------------------------------
# Test: Group B — Billing / Payment
# ---------------------------------------------------------------------------


class TestGroupBBilling:

    def test_is_electronic_check_flag(self, base_df):
        result = engineer_features(base_df)
        assert result[IS_ELECTRONIC_CHECK].iloc[0] == 1

    def test_is_auto_pay_for_bank_transfer(self, multi_row_df):
        result = engineer_features(multi_row_df)
        assert result[IS_AUTO_PAY].iloc[1] == 1

    def test_is_auto_pay_zero_for_electronic_check(self, base_df):
        result = engineer_features(base_df)
        assert result[IS_AUTO_PAY].iloc[0] == 0

    def test_is_high_monthly_charge_flag(self, base_df):
        result = engineer_features(base_df)
        assert result[IS_HIGH_MONTHLY_CHARGE].iloc[0] == 1

    def test_monthly_charge_bin_valid_values(self, multi_row_df):
        result = engineer_features(multi_row_df)
        assert set(result[MONTHLY_CHARGE_BIN].unique()).issubset(
            set(MONTHLY_CHARGE_BIN_ORDER)
        )

    def test_charges_gap_zero_for_new_customer(self, multi_row_df):
        result = engineer_features(multi_row_df)
        gap = result[CHARGES_GAP].iloc[1]
        assert gap == pytest.approx(0.0, abs=1e-6)

    def test_charges_gap_consistent_billing(self, base_df):
        result = engineer_features(base_df)
        gap = result[CHARGES_GAP].iloc[0]
        expected = 844.20 - (70.35 * 12)
        assert gap == pytest.approx(expected, abs=1e-3)


# ---------------------------------------------------------------------------
# Test: Group C — Service Depth
# ---------------------------------------------------------------------------


class TestGroupCServiceDepth:

    def test_has_internet_fiber_optic(self, base_df):
        result = engineer_features(base_df)
        assert result[HAS_INTERNET].iloc[0] == 1

    def test_streaming_count_range(self, multi_row_df):
        result = engineer_features(multi_row_df)
        assert result[STREAMING_COUNT].between(0, 2).all()

    def test_security_service_count_range(self, multi_row_df):
        result = engineer_features(multi_row_df)
        assert result[SECURITY_SERVICE_COUNT].between(0, 3).all()

    def test_has_full_streaming_requires_both(self, base_df):
        result = engineer_features(base_df)
        assert result[HAS_FULL_STREAMING].iloc[0] == 0

    def test_internet_addon_count_range(self, multi_row_df):
        result = engineer_features(multi_row_df)
        assert result[INTERNET_ADDON_COUNT].between(0, 6).all()

    def test_has_no_internet_addons_only_when_has_internet(self, multi_row_df):
        result = engineer_features(multi_row_df)
        no_internet_rows = result[result[HAS_INTERNET] == 0]
        assert (no_internet_rows[HAS_NO_INTERNET_ADDONS] == 0).all()


# ---------------------------------------------------------------------------
# Test: Group D — Contract / Commitment
# ---------------------------------------------------------------------------


class TestGroupDContract:

    def test_contract_numeric_m2m_is_zero(self, base_df):
        result = engineer_features(base_df)
        assert result[CONTRACT_NUMERIC].iloc[0] == 0

    def test_contract_numeric_two_year_is_two(self, multi_row_df):
        result = engineer_features(multi_row_df)
        assert result[CONTRACT_NUMERIC].iloc[1] == 2

    def test_is_committed_customer_for_two_year(self, multi_row_df):
        result = engineer_features(multi_row_df)
        assert result[IS_COMMITTED_CUSTOMER].iloc[1] == 1

    def test_is_committed_customer_zero_for_m2m(self, base_df):
        result = engineer_features(base_df)
        assert result[IS_COMMITTED_CUSTOMER].iloc[0] == 0


# ---------------------------------------------------------------------------
# Test: Group E — Interaction / Compound
# ---------------------------------------------------------------------------


class TestGroupEInteraction:

    def test_m2m_and_fiber_flag(self, base_df):
        result = engineer_features(base_df)
        assert result[M2M_AND_FIBER].iloc[0] == 1

    def test_m2m_and_fiber_zero_when_two_year(self, multi_row_df):
        result = engineer_features(multi_row_df)
        assert result[M2M_AND_FIBER].iloc[1] == 0

    # explicitly create a new + high-value customer
    def test_new_high_value_customer_flag(self, base_row):
        # base_row has tenure=12 which does NOT satisfy tenure <= 6.
        # Override tenure to 3 to create a genuine new-customer scenario.
        new_high_value_row = {**base_row, "tenure": 3}
        result = engineer_features(pd.DataFrame([new_high_value_row]))
        assert result[NEW_HIGH_VALUE_CUSTOMER].iloc[0] == 1

    def test_new_high_value_customer_zero_for_established(self, base_df):
        # base_df tenure=12 > 6 threshold → NOT a new customer → must be 0
        result = engineer_features(base_df)
        assert result[NEW_HIGH_VALUE_CUSTOMER].iloc[0] == 0

    def test_new_high_value_customer_zero_for_low_charges(self, base_row):
        # tenure <= 6 but MonthlyCharges <= 70 → NOT high-value → must be 0
        low_charge_new_row = {**base_row, "tenure": 2, "MonthlyCharges": 45.0}
        result = engineer_features(pd.DataFrame([low_charge_new_row]))
        assert result[NEW_HIGH_VALUE_CUSTOMER].iloc[0] == 0

    def test_fiber_no_security_flag(self, base_df):
        result = engineer_features(base_df)
        assert result[FIBER_NO_SECURITY].iloc[0] == 1

    def test_fiber_no_security_zero_when_dsl(self, multi_row_df):
        result = engineer_features(multi_row_df)
        assert result[FIBER_NO_SECURITY].iloc[1] == 0

    def test_paperless_and_echeck_flag(self, base_df):
        result = engineer_features(base_df)
        assert result[PAPERLESS_AND_ECHECK].iloc[0] == 1


# ---------------------------------------------------------------------------
# Test: Group F — Composite Score
# ---------------------------------------------------------------------------


class TestGroupFComposite:

    def test_composite_risk_range_0_to_7(self, multi_row_df):
        result = engineer_features(multi_row_df)
        assert result[COMPOSITE_CHURN_RISK].between(0, 7).all()

    def test_composite_risk_high_for_all_risk_factors(self):
        high_risk = pd.DataFrame(
            [
                {
                    "gender": "Male",
                    "SeniorCitizen": "0",
                    "Partner": "No",
                    "Dependents": "No",
                    "tenure": 1,
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "No",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 90.0,
                    "TotalCharges": 90.0,
                }
            ]
        )
        result = engineer_features(high_risk)
        assert result[COMPOSITE_CHURN_RISK].iloc[0] >= 5

    def test_composite_risk_low_for_loyal_customer(self):
        loyal = pd.DataFrame(
            [
                {
                    "gender": "Female",
                    "SeniorCitizen": "0",
                    "Partner": "Yes",
                    "Dependents": "Yes",
                    "tenure": 60,
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "DSL",
                    "OnlineSecurity": "Yes",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "Yes",
                    "TechSupport": "Yes",
                    "StreamingTV": "Yes",
                    "StreamingMovies": "Yes",
                    "Contract": "Two year",
                    "PaperlessBilling": "No",
                    "PaymentMethod": "Bank transfer (automatic)",
                    "MonthlyCharges": 50.0,
                    "TotalCharges": 3000.0,
                }
            ]
        )
        result = engineer_features(loyal)
        assert result[COMPOSITE_CHURN_RISK].iloc[0] <= 2


# ---------------------------------------------------------------------------
# Test: Public API functions
# ---------------------------------------------------------------------------


class TestPublicAPI:

    def test_get_feature_names_returns_list(self):
        assert isinstance(get_feature_names(), list)

    def test_get_categorical_features_returns_list(self):
        assert isinstance(get_categorical_features(), list)

    def test_get_numerical_features_returns_list(self):
        assert isinstance(get_numerical_features(), list)

    def test_get_binary_features_returns_list(self):
        assert isinstance(get_binary_features(), list)

    def test_no_feature_in_two_routing_lists(self):
        cat = set(get_categorical_features())
        num = set(get_numerical_features())
        bin_ = set(get_binary_features())
        assert len(cat & num) == 0, f"Overlap cat∩num: {cat & num}"
        assert len(cat & bin_) == 0, f"Overlap cat∩bin: {cat & bin_}"
        assert len(num & bin_) == 0, f"Overlap num∩bin: {num & bin_}"

    def test_get_engineered_feature_names_returns_28(self):
        assert len(get_engineered_feature_names()) == 28

    def test_get_feature_names_covers_all_engineered(self):
        all_names = set(get_feature_names())
        for feat in ENGINEERED_FEATURES:
            assert (
                feat in all_names
            ), f"Engineered feature {feat} missing from get_feature_names()"
