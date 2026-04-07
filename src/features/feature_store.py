"""
Feature store — Phase 1 (Centralized Module).

This is the single source of truth for all feature definitions in
the churn prediction system. Every feature used by the model is
defined here and only here. The training pipeline, inference API,
and monitoring pipeline all import from this module — never
define features in multiple places.

Why this matters:
    Training-serving skew is one of the most common and hardest-to-debug
    production ML failures. It happens when the feature logic in the
    training pipeline differs (even slightly) from the feature logic at
    inference time. By centralising all definitions here, we make skew
    structurally impossible — there is only one implementation.

Architecture note (Phase 1 vs Phase 2):
    Phase 1 (this file): A centralized Python module. Simple, fast to build,
    zero infrastructure dependencies. Sufficient for this dataset size.

    Phase 2 (future): Replace with Feast feature store when the team needs
    shared feature serving across multiple models, or when online/offline
    feature freshness requirements diverge. The function signatures here
    are designed to make that migration straightforward — each function
    takes a DataFrame and returns a DataFrame.

Feature inventory (7 engineered features, all justified by EDA):
    1. tenure_group           — Captures the non-linear tenure/churn relationship
    2. is_month_to_month      — Distils the strongest single predictor to a boolean
    3. service_adoption_count — Measures customer stickiness via service breadth
    4. has_protection_bundle  — Specific high-retention service combination
    5. charge_to_tenure_ratio — Identifies overpriced new customers (highest risk)
    6. is_fiber_optic         — Isolates the high-churn infrastructure segment
    7. avg_charge_per_service — Perceived value: cost divided by services received

Public API:
    engineer_features(df)       → df with all 7 new columns added
    get_feature_names()         → list of all feature column names after engineering
    get_categorical_features()  → list of categorical feature names
    get_numerical_features()    → list of numerical feature names
    get_engineered_feature_names() → list of only the 7 new columns
"""

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Feature name constants
# Defined as constants so they can be imported and used in tests,
# the sklearn Pipeline's ColumnTransformer, and the SHAP explainer
# without any risk of typos causing silent KeyErrors.
# ---------------------------------------------------------------------------

# Engineered feature names
TENURE_GROUP = "tenure_group"
IS_MONTH_TO_MONTH = "is_month_to_month"
SERVICE_ADOPTION_COUNT = "service_adoption_count"
HAS_PROTECTION_BUNDLE = "has_protection_bundle"
CHARGE_TO_TENURE_RATIO = "charge_to_tenure_ratio"
IS_FIBER_OPTIC = "is_fiber_optic"
AVG_CHARGE_PER_SERVICE = "avg_charge_per_service"

ENGINEERED_FEATURES = [
    TENURE_GROUP,
    IS_MONTH_TO_MONTH,
    SERVICE_ADOPTION_COUNT,
    HAS_PROTECTION_BUNDLE,
    CHARGE_TO_TENURE_RATIO,
    IS_FIBER_OPTIC,
    AVG_CHARGE_PER_SERVICE,
]

# Original features kept as-is (post-preprocess, customerID already dropped)
BASE_CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",  # cast to str "0"/"1" by preprocess.py
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_group",  # engineered — ordinal category
]

BASE_NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "service_adoption_count",  # engineered — integer count
    "charge_to_tenure_ratio",  # engineered — float ratio
    "avg_charge_per_service",  # engineered — float ratio
]

BASE_BINARY_FEATURES = [
    "is_month_to_month",  # engineered — 0/1 int
    "has_protection_bundle",  # engineered — 0/1 int
    "is_fiber_optic",  # engineered — 0/1 int
]

# Tenure group ordering for ordinal encoding downstream
TENURE_GROUP_ORDER = ["0-6m", "7-12m", "13-24m", "25-48m", "49+m"]


# ---------------------------------------------------------------------------
# Individual feature engineering functions
# Each function is pure: takes a DataFrame, returns a DataFrame.
# Each is independently testable.
# ---------------------------------------------------------------------------


def _add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin tenure (months) into 5 ordinal groups.

    EDA finding: Churn is overwhelmingly a new-customer problem.
    0-6 months: ~47% churn. 49+ months: <10% churn. The relationship
    is non-linear and monotonically decreasing, making a binned ordinal
    feature more expressive than raw tenure for tree-based models.

    Raw tenure is also kept — this feature adds signal, not replaces it.

    Bins:
        0-6m   → customers in their first half-year (highest risk)
        7-12m  → completing their first year
        13-24m → second year (churn risk declining sharply)
        25-48m → established customers
        49+m   → loyal long-term customers (lowest risk)
    """
    df[TENURE_GROUP] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, float("inf")],
        labels=TENURE_GROUP_ORDER,
        right=True,
    ).astype(str)

    return df


def _add_is_month_to_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary flag: 1 if the customer is on a month-to-month contract.

    EDA finding: Month-to-month customers churn at 42.7% vs 11.3%
    (one year) and 2.8% (two year). This is the strongest single
    predictor in the dataset. Extracting it as an explicit binary
    feature concentrates that signal into a form that every model
    architecture (including linear models) can leverage directly.
    """
    df[IS_MONTH_TO_MONTH] = (df["Contract"] == "Month-to-month").astype(int)
    return df


def _add_service_adoption_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number of active services a customer subscribes to.

    Services counted (8 possible):
        PhoneService, MultipleLines, InternetService,
        OnlineSecurity, OnlineBackup, DeviceProtection,
        TechSupport, StreamingTV, StreamingMovies

    "Active" definition:
        - PhoneService: "Yes"
        - MultipleLines: "Yes" (not "No phone service")
        - InternetService: "DSL" or "Fiber optic" (not "No")
        - All others: "Yes" (not "No" or "No internet service")

    EDA finding: The relationship between service count and churn is
    non-linear — not simply "more services = lower churn." But this
    raw count is still useful in combination with other features,
    particularly as the denominator in avg_charge_per_service.
    """
    count = pd.Series(0, index=df.index)

    count += (df["PhoneService"] == "Yes").astype(int)
    count += (df["MultipleLines"] == "Yes").astype(int)
    count += (df["InternetService"] != "No").astype(int)

    internet_addons = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for col in internet_addons:
        count += (df[col] == "Yes").astype(int)

    df[SERVICE_ADOPTION_COUNT] = count
    return df


def _add_has_protection_bundle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary flag: 1 if the customer subscribes to both TechSupport
    AND OnlineSecurity.

    EDA finding: The service count vs churn relationship is non-linear,
    but specific high-retention services drive that pattern. Customers
    with TechSupport + OnlineSecurity — a "protection bundle" — show
    significantly lower churn than other multi-service customers.
    This interaction is unlikely to be learned efficiently from the
    raw categorical columns alone.

    Note: DeviceProtection was considered for inclusion but its
    marginal contribution after TechSupport + OnlineSecurity was
    negligible in EDA correlation analysis.
    """
    df[HAS_PROTECTION_BUNDLE] = (
        (df["TechSupport"] == "Yes") & (df["OnlineSecurity"] == "Yes")
    ).astype(int)
    return df


def _add_charge_to_tenure_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ratio of monthly charge to tenure, adjusted for new customers.

    Formula: MonthlyCharges / (tenure + 1)

    The +1 prevents division by zero for tenure=0 customers and also
    provides sensible scaling: a new customer (tenure=0) paying $80/month
    gets a ratio of 80.0, which is extreme relative to a 24-month customer
    paying the same amount (ratio = 3.2).

    EDA finding: Churned customers have significantly higher MonthlyCharges
    ($79.49 median) than retained customers ($61.27) AND lower tenure.
    This ratio directly captures the "paying a lot but haven't committed"
    profile — the highest-risk customer segment.

    Business interpretation: High ratio = paying more than long-term value
    justifies. These customers haven't yet built the switching-cost inertia
    that long-tenured customers have.
    """
    df[CHARGE_TO_TENURE_RATIO] = df["MonthlyCharges"] / (df["tenure"] + 1)
    return df


def _add_is_fiber_optic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary flag: 1 if the customer has Fiber optic internet service.

    EDA finding: Fiber optic customers churn at 41.9% — nearly 3x the
    rate of DSL customers (18.9%) and 10x customers with no internet
    (7.4%). This signals a service quality or value perception problem
    specific to the Fiber product.

    While InternetService is already a categorical feature, the extreme
    churn rate of the Fiber segment justifies making it an explicit
    binary flag so that models with less capacity (logistic regression,
    shallow trees) can capture this signal without needing to learn
    a multi-class contrast.
    """
    df[IS_FIBER_OPTIC] = (df["InternetService"] == "Fiber optic").astype(int)
    return df


def _add_avg_charge_per_service(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly charge divided by the number of active services.

    Formula: MonthlyCharges / (service_adoption_count + 1)

    The +1 prevents division by zero for customers with no active services
    and also smooths the ratio for single-service customers.

    Requires: _add_service_adoption_count must run first.

    Business interpretation: A customer paying $80/month for 8 services
    is getting good value (ratio = ~8.9). A customer paying $80/month
    for 1 service is paying 9x more per service — high perceived cost,
    high churn risk. This feature captures value-for-money perception,
    which is one of the top drivers of telecom churn in the literature.
    """
    if SERVICE_ADOPTION_COUNT not in df.columns:
        raise ValueError(
            f"'{SERVICE_ADOPTION_COUNT}' column missing. "
            "Call _add_service_adoption_count() before _add_avg_charge_per_service()."
        )

    df[AVG_CHARGE_PER_SERVICE] = df["MonthlyCharges"] / (df[SERVICE_ADOPTION_COUNT] + 1)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all 7 feature engineering transformations to the input DataFrame.

    This is the single function called by the training pipeline and the
    inference API. The order of operations is fixed and must not be changed:
    _add_service_adoption_count() must run before _add_avg_charge_per_service()
    because the latter depends on the column created by the former.

    This function is STATELESS — it never fits anything to the data.
    No mean, no median, no percentile is computed here. Every transformation
    is a deterministic rule applied identically to any input row.

    This means it can be safely called on training data, validation data,
    test data, and individual inference requests without any risk of leakage.

    Args:
        df: Preprocessed DataFrame (output of src.data.preprocess.preprocess).
            Must contain all original Telco feature columns.
            customerID must already be dropped (handled by preprocess).

    Returns:
        DataFrame with all original columns plus 7 new engineered columns.
        Row count is unchanged.

    Raises:
        KeyError: If a required source column is missing.
        ValueError: If column dependency order is violated.
    """
    required_cols = [
        "tenure",
        "Contract",
        "MonthlyCharges",
        "InternetService",
        "TechSupport",
        "OnlineSecurity",
        "PhoneService",
        "MultipleLines",
        "OnlineBackup",
        "DeviceProtection",
        "StreamingTV",
        "StreamingMovies",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"engineer_features: missing required columns: {missing}. "
            f"Ensure preprocess() has been called before engineer_features()."
        )

    df = df.copy()

    logger.info("Engineering features for DataFrame with shape %s.", df.shape)

    # Order matters: service_adoption_count before avg_charge_per_service
    df = _add_tenure_group(df)
    df = _add_is_month_to_month(df)
    df = _add_service_adoption_count(df)
    df = _add_has_protection_bundle(df)
    df = _add_charge_to_tenure_ratio(df)
    df = _add_is_fiber_optic(df)
    df = _add_avg_charge_per_service(df)

    logger.info(
        "Feature engineering complete. Output shape: %s. " "New columns added: %s",
        df.shape,
        ENGINEERED_FEATURES,
    )

    return df


def get_feature_names() -> list[str]:
    """
    Return the complete list of feature column names after engineering.

    Used by the sklearn ColumnTransformer to route columns to the
    correct transformer (OneHotEncoder for categoricals, StandardScaler
    for numericals, passthrough for binaries).

    Returns:
        Flat list of all feature column names (categorical + numerical + binary).
    """
    return BASE_CATEGORICAL_FEATURES + BASE_NUMERICAL_FEATURES + BASE_BINARY_FEATURES


def get_categorical_features() -> list[str]:
    """
    Return categorical feature names for OneHotEncoder routing.

    Includes tenure_group (engineered ordinal) alongside the original
    categorical columns. The sklearn Pipeline will apply OneHotEncoder
    to all of these.
    """
    return BASE_CATEGORICAL_FEATURES.copy()


def get_numerical_features() -> list[str]:
    """
    Return numerical feature names for StandardScaler routing.

    Includes the three engineered ratio/count features alongside
    the original numeric columns.
    """
    return BASE_NUMERICAL_FEATURES.copy()


def get_binary_features() -> list[str]:
    """
    Return binary (0/1) engineered feature names.

    These are passed through the ColumnTransformer without scaling
    or encoding — they are already in the correct numeric form.
    """
    return BASE_BINARY_FEATURES.copy()


def get_engineered_feature_names() -> list[str]:
    """
    Return only the names of the 7 newly engineered features.

    Used in tests and explainability analysis to distinguish
    engineered features from original features.
    """
    return ENGINEERED_FEATURES.copy()
