"""Feature store — Phase 1 Centralized Module.

This is the single source of truth for ALL feature definitions in the churn
prediction system. The training pipeline, inference API, and monitoring pipeline
all import from this module — never define features in multiple places.

Why this matters:
    Training-serving skew is one of the most common and hardest-to-debug
    production ML failures. By centralising all definitions here, we make
    skew structurally impossible — there is only one implementation.

Feature inventory — 28 engineered features across 6 groups:

Original 7 (Phase 1, retained):
    1.  tenure_group           Non-linear tenure → churn relationship (ordinal bins)
    2.  is_month_to_month      Strongest single predictor → binary flag
    3.  service_adoption_count Customer stickiness via service breadth
    4.  has_protection_bundle  High-retention TechSupport + OnlineSecurity combo
    5.  charge_to_tenure_ratio Overpriced-new-customer risk ratio
    6.  is_fiber_optic         High-churn infrastructure segment flag
    7.  avg_charge_per_service Perceived value: cost divided by services

Group A — Demographic / Lifecycle (3 new):
    8.  has_family             Partner OR Dependents → lower switching friction
    9.  is_isolated            Neither Partner nor Dependents → lone churner
    10. senior_no_support      Senior citizen without TechSupport → friction risk

Group B — Billing / Payment (5 new):
    11. is_auto_pay             Automatic payment → commitment signal
    12. is_electronic_check     Highest-churn payment method (45.3%)
    13. is_high_monthly_charge  MonthlyCharges > 70 → price-pressure segment
    14. monthly_charge_bin      Low / Medium / High pricing tier (categorical)
    15. charges_gap             TotalCharges vs expected billing (anomaly detector)

Group C — Service Depth (6 new):
    16. has_internet            Any internet service → add-on eligibility
    17. streaming_count         TV + Movies subs (0-2) → entertainment stickiness
    18. security_service_count  OnlineSecurity + OnlineBackup + DeviceProtection (0-3)
    19. has_full_streaming      Both StreamingTV AND StreamingMovies
    20. internet_addon_count    All 6 internet add-ons count (0-6)
    21. has_no_internet_addons  Internet but ZERO add-ons → disengaged high-risk

Group D — Contract / Commitment (2 new):
    22. contract_numeric        M2M=0, 1yr=1, 2yr=2 → ordinal commitment scale
    23. is_committed_customer   Non-M2M contract → contractual barrier to churn

Group E — Interaction / Compound (4 new):
    24. m2m_and_fiber           M2M × Fiber (highest-risk combo from EDA heatmap)
    25. new_high_value_customer tenure ≤ 6 AND
        MonthlyCharges > 70 → top retention target
    26. fiber_no_security       Fiber optic AND no OnlineSecurity → poaching risk
    27. paperless_and_echeck    PaperlessBilling AND Electronic check → dual risk flag

Group F — Composite Score (1 new):
    28. composite_churn_risk    Weighted risk score 0–7 → interpretable risk tier

Public API:
    engineer_features(df)          → df with all 28 new columns added
    get_feature_names()            → list of all feature columns after engineering
    get_categorical_features()     → list for OneHotEncoder routing
    get_numerical_features()       → list for StandardScaler routing
    get_binary_features()          → list for passthrough routing
    get_engineered_feature_names() → list of only the 28 new columns
"""

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Feature name constants — the ONLY place these strings are defined.
# Import these constants everywhere instead of using raw strings.
# ---------------------------------------------------------------------------

# ── Original 7 ──────────────────────────────────────────────────────────────
TENURE_GROUP = "tenure_group"
IS_MONTH_TO_MONTH = "is_month_to_month"
SERVICE_ADOPTION_COUNT = "service_adoption_count"
HAS_PROTECTION_BUNDLE = "has_protection_bundle"
CHARGE_TO_TENURE_RATIO = "charge_to_tenure_ratio"
IS_FIBER_OPTIC = "is_fiber_optic"
AVG_CHARGE_PER_SERVICE = "avg_charge_per_service"

# ── Group A — Demographic / Lifecycle ────────────────────────────────────────
HAS_FAMILY = "has_family"
IS_ISOLATED = "is_isolated"
SENIOR_NO_SUPPORT = "senior_no_support"

# ── Group B — Billing / Payment ──────────────────────────────────────────────
IS_AUTO_PAY = "is_auto_pay"
IS_ELECTRONIC_CHECK = "is_electronic_check"
IS_HIGH_MONTHLY_CHARGE = "is_high_monthly_charge"
MONTHLY_CHARGE_BIN = "monthly_charge_bin"
CHARGES_GAP = "charges_gap"

# ── Group C — Service Depth ──────────────────────────────────────────────────
HAS_INTERNET = "has_internet"
STREAMING_COUNT = "streaming_count"
SECURITY_SERVICE_COUNT = "security_service_count"
HAS_FULL_STREAMING = "has_full_streaming"
INTERNET_ADDON_COUNT = "internet_addon_count"
HAS_NO_INTERNET_ADDONS = "has_no_internet_addons"

# ── Group D — Contract / Commitment ──────────────────────────────────────────
CONTRACT_NUMERIC = "contract_numeric"
IS_COMMITTED_CUSTOMER = "is_committed_customer"

# ── Group E — Interaction / Compound ─────────────────────────────────────────
M2M_AND_FIBER = "m2m_and_fiber"
NEW_HIGH_VALUE_CUSTOMER = "new_high_value_customer"
FIBER_NO_SECURITY = "fiber_no_security"
PAPERLESS_AND_ECHECK = "paperless_and_echeck"

# ── Group F — Composite Score ─────────────────────────────────────────────────
COMPOSITE_CHURN_RISK = "composite_churn_risk"

# ---------------------------------------------------------------------------
# Master list of all 28 engineered feature names (new columns only).
# Used by the notebook to distinguish engineered vs. original columns.
# ---------------------------------------------------------------------------
ENGINEERED_FEATURES: list[str] = [
    # Original 7
    TENURE_GROUP,
    IS_MONTH_TO_MONTH,
    SERVICE_ADOPTION_COUNT,
    HAS_PROTECTION_BUNDLE,
    CHARGE_TO_TENURE_RATIO,
    IS_FIBER_OPTIC,
    AVG_CHARGE_PER_SERVICE,
    # Group A
    HAS_FAMILY,
    IS_ISOLATED,
    SENIOR_NO_SUPPORT,
    # Group B
    IS_AUTO_PAY,
    IS_ELECTRONIC_CHECK,
    IS_HIGH_MONTHLY_CHARGE,
    MONTHLY_CHARGE_BIN,
    CHARGES_GAP,
    # Group C
    HAS_INTERNET,
    STREAMING_COUNT,
    SECURITY_SERVICE_COUNT,
    HAS_FULL_STREAMING,
    INTERNET_ADDON_COUNT,
    HAS_NO_INTERNET_ADDONS,
    # Group D
    CONTRACT_NUMERIC,
    IS_COMMITTED_CUSTOMER,
    # Group E
    M2M_AND_FIBER,
    NEW_HIGH_VALUE_CUSTOMER,
    FIBER_NO_SECURITY,
    PAPERLESS_AND_ECHECK,
    # Group F
    COMPOSITE_CHURN_RISK,
]

# ---------------------------------------------------------------------------
# Feature routing lists — used by ColumnTransformer in pipeline.py.
# Rule: each feature appears in EXACTLY ONE of these three lists.
# ---------------------------------------------------------------------------

BASE_CATEGORICAL_FEATURES: list[str] = [
    # Original raw categoricals
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
    # Engineered categoricals
    TENURE_GROUP,  # ordinal: "0-6m" / "7-12m" / … / "49m+"
    MONTHLY_CHARGE_BIN,  # ordinal: "Low" / "Medium" / "High"
]

BASE_NUMERICAL_FEATURES: list[str] = [
    # Original numerics
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    # Engineered numerics (scaled by StandardScaler)
    SERVICE_ADOPTION_COUNT,
    CHARGE_TO_TENURE_RATIO,
    AVG_CHARGE_PER_SERVICE,
    CHARGES_GAP,
    STREAMING_COUNT,
    SECURITY_SERVICE_COUNT,
    INTERNET_ADDON_COUNT,
    CONTRACT_NUMERIC,
    COMPOSITE_CHURN_RISK,
]

BASE_BINARY_FEATURES: list[str] = [
    # Original 7 binary engineered (passthrough — already 0/1 ints)
    IS_MONTH_TO_MONTH,
    HAS_PROTECTION_BUNDLE,
    IS_FIBER_OPTIC,
    # Group A
    HAS_FAMILY,
    IS_ISOLATED,
    SENIOR_NO_SUPPORT,
    # Group B
    IS_AUTO_PAY,
    IS_ELECTRONIC_CHECK,
    IS_HIGH_MONTHLY_CHARGE,
    # Group C
    HAS_INTERNET,
    HAS_FULL_STREAMING,
    HAS_NO_INTERNET_ADDONS,
    # Group D
    IS_COMMITTED_CUSTOMER,
    # Group E
    M2M_AND_FIBER,
    NEW_HIGH_VALUE_CUSTOMER,
    FIBER_NO_SECURITY,
    PAPERLESS_AND_ECHECK,
]

# Lookup constants used in feature functions
TENURE_GROUP_ORDER: list[str] = ["0-6m", "7-12m", "13-24m", "25-48m", "49m+"]
MONTHLY_CHARGE_BIN_ORDER: list[str] = ["Low", "Medium", "High"]
AUTO_PAY_METHODS: frozenset[str] = frozenset(
    {
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    }
)
CONTRACT_MAP: dict[str, int] = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2,
}
HIGH_MONTHLY_CHARGE_THRESHOLD: float = 70.0
NEW_CUSTOMER_TENURE_THRESHOLD: int = 6

# ---------------------------------------------------------------------------
# Individual feature engineering functions — one function per feature.
# Each function:
#   - Takes a DataFrame
#   - Adds exactly one column
#   - Returns the DataFrame
#   - Is STATELESS (no fitting, no distribution-dependent computation)
# ---------------------------------------------------------------------------

# ── Original 7 ──────────────────────────────────────────────────────────────


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bin tenure (months) into 5 ordinal groups.

    EDA finding: Churn is overwhelmingly a new-customer problem.
    0-6 months → 47% churn.  49+ months → 10% churn.
    The relationship is non-linear and monotonically decreasing,
    making a binned ordinal feature more expressive than raw tenure
    for tree-based models. Raw tenure is also kept — this adds signal.

    Bins:
        0-6m   customers in their first half-year (highest risk)
        7-12m  completing their first year
        13-24m second year — churn risk declining sharply
        25-48m established customers
        49m+   loyal long-term customers (lowest risk)
    """
    df[TENURE_GROUP] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, float("inf")],
        labels=TENURE_GROUP_ORDER,
        right=True,
    ).astype(str)
    return df


def add_is_month_to_month(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag: 1 if the customer is on a Month-to-month contract.

    EDA finding: M2M customers churn at 42.7% vs 11.3% (One year)
    and 2.8% (Two year). This is the strongest single predictor.
    Extracting it as a binary concentrates the signal for linear models.
    """
    df[IS_MONTH_TO_MONTH] = (df["Contract"] == "Month-to-month").astype(int)
    return df


def add_service_adoption_count(df: pd.DataFrame) -> pd.DataFrame:
    """Count the number of active services a customer subscribes to (0-9).

    Services counted (8 possible):
        PhoneService, MultipleLines, InternetService,
        OnlineSecurity, OnlineBackup, DeviceProtection,
        TechSupport, StreamingTV, StreamingMovies

    Active definition:
        PhoneService    == "Yes"
        MultipleLines   == "Yes"  (not "No phone service")
        InternetService != "No"
        All others      == "Yes"  (not "No" or "No internet service")

    EDA: Non-linear stickiness signal. Serves as denominator for
    avg_charge_per_service. Must run BEFORE add_avg_charge_per_service.
    """
    count = pd.Series(0, index=df.index)
    count += (df["PhoneService"] == "Yes").astype(int)
    count += (df["MultipleLines"] == "Yes").astype(int)
    count += (df["InternetService"] != "No").astype(int)
    for col in [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]:
        count += (df[col] == "Yes").astype(int)
    df[SERVICE_ADOPTION_COUNT] = count
    return df


def add_has_protection_bundle(df: pd.DataFrame) -> pd.DataFrame:
    """Binary: 1 if customer subscribes to BOTH TechSupport AND OnlineSecurity.

    EDA: This combination drives meaningfully lower churn.
    The interaction is unlikely to be learned efficiently from raw
    categorical columns alone by linear models.
    """
    df[HAS_PROTECTION_BUNDLE] = (
        (df["TechSupport"] == "Yes") & (df["OnlineSecurity"] == "Yes")
    ).astype(int)
    return df


def add_charge_to_tenure_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Ratio of monthly charge to tenure+1.

    Formula: MonthlyCharges / (tenure + 1)
    The +1 prevents division by zero for tenure=0 customers and provides
    sensible scaling — a new customer (tenure=0) paying $80/month gets
    ratio=80.0 vs a 24-month customer ratio=3.2.

    EDA: Churned customers have significantly higher MonthlyCharges
    ($79.49 median) AND lower tenure. This ratio captures the
    "paying a lot but haven't committed" profile — the highest-risk segment.
    """
    df[CHARGE_TO_TENURE_RATIO] = df["MonthlyCharges"] / (df["tenure"] + 1)
    return df


def add_is_fiber_optic(df: pd.DataFrame) -> pd.DataFrame:
    """Binary: 1 if customer has Fiber optic internet service.

    EDA: Fiber customers churn at 41.9% — nearly 3x DSL (18.9%)
    and 10x customers with no internet (7.4%). Making this an explicit
    binary helps linear models that cannot learn multi-class contrasts.
    """
    df[IS_FIBER_OPTIC] = (df["InternetService"] == "Fiber optic").astype(int)
    return df


def add_avg_charge_per_service(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly charge divided by number of active services.

    Formula: MonthlyCharges / (service_adoption_count + 1)
    Captures perceived value — paying more per service → higher churn.

    DEPENDENCY: add_service_adoption_count must run first.
    """
    df[AVG_CHARGE_PER_SERVICE] = df["MonthlyCharges"] / (df[SERVICE_ADOPTION_COUNT] + 1)
    return df


# ── Group A — Demographic / Lifecycle ────────────────────────────────────────


def add_has_family(df: pd.DataFrame) -> pd.DataFrame:
    """Binary: 1 if the customer has a Partner OR Dependents (or both).

    Business rationale: Family customers have shared accounts, bundled
    services, and higher switching friction. A partner or children mean
    the entire household would need to switch — a significant barrier.
    Also proxy for financial stability (lower price sensitivity).
    """
    df[HAS_FAMILY] = ((df["Partner"] == "Yes") | (df["Dependents"] == "Yes")).astype(
        int
    )
    return df


def add_is_isolated(df: pd.DataFrame) -> pd.DataFrame:
    """Binary: 1 if the customer has neither Partner nor Dependents.

    Logical complement of has_family. Kept as an explicit feature for
    interpretability in SHAP / linear model coefficients — "isolated
    customer" is a more actionable retention segment label than "not
    has_family". Note: is_isolated == 1 - has_family, so the correlation
    filter in feature_selector.py will catch one of these if they are
    perfectly redundant at selection time.
    """
    df[IS_ISOLATED] = ((df["Partner"] != "Yes") & (df["Dependents"] != "Yes")).astype(
        int
    )
    return df


def add_senior_no_support(df: pd.DataFrame) -> pd.DataFrame:
    """Binary: 1 if customer is Senior Citizen AND does NOT have TechSupport.

    Business rationale: Senior citizens are a vulnerable segment. Without
    TechSupport they experience higher friction with digital services, which
    can trigger cancellation especially when combined with billing confusion.
    This interaction captures a specific underserved-customer profile.

    Note: SeniorCitizen is stored as str "0"/"1" after preprocess.py casts it.
    """
    df[SENIOR_NO_SUPPORT] = (
        (df["SeniorCitizen"].astype(str) == "1") & (df["TechSupport"] != "Yes")
    ).astype(int)
    return df


# ── Group B — Billing / Payment ──────────────────────────────────────────────


def add_is_auto_pay(df: pd.DataFrame) -> pd.DataFrame:
    """Binary: 1 if the customer uses an automatic payment method.

    Auto-pay methods: Bank transfer (automatic), Credit card (automatic).
    Business rationale: Auto-pay customers show stronger commitment.
    Payment is frictionless — no monthly action required to stay subscribed.
    Manual payers (especially electronic check) must actively choose to pay,
    creating natural cancellation decision points each billing cycle.
    """
    df[IS_AUTO_PAY] = df["PaymentMethod"].isin(AUTO_PAY_METHODS).astype(int)
    return df


def add_is_electronic_check(df: pd.DataFrame) -> pd.DataFrame:
    """Binary: 1 if payment method is Electronic check.

    EDA: Electronic check customers churn at 45.3% — the highest of
    all payment methods, nearly 2x the dataset average (26.5%).
    While PaymentMethod is already a categorical feature, making this
    an explicit binary amplifies the signal for linear models that cannot
    efficiently learn a 4-class split where one class dominates.
    """
    df[IS_ELECTRONIC_CHECK] = (df["PaymentMethod"] == "Electronic check").astype(int)
    return df


def add_is_high_monthly_charge(df: pd.DataFrame) -> pd.DataFrame:
    """Binary: 1 if MonthlyCharges > 70.

    EDA: Median MonthlyCharges for churned customers is $79.49 vs
    $61.27 for retained (t-test p < 0.001). Threshold of $70 captures
    the top ~35% of monthly payers where the churn signal is strongest.
    Provides a non-linear discretisation that complements the raw numeric.
    """
    df[IS_HIGH_MONTHLY_CHARGE] = (
        df["MonthlyCharges"] > HIGH_MONTHLY_CHARGE_THRESHOLD
    ).astype(int)
    return df


def add_monthly_charge_bin(df: pd.DataFrame) -> pd.DataFrame:
    """Bin MonthlyCharges into Low / Medium / High categorical.

    Bins:
        Low    (≤ $35):  phone-only or entry-level internet customers
        Medium ($35-70): mid-tier service bundle customers
        High   (> $70):  premium / Fiber optic customers (highest churn)

    Business rationale: Provides an interpretable pricing-tier signal.
    Helps linear models learn non-monotonic pricing effects without
    relying on polynomial terms. Complements is_high_monthly_charge
    and charge_to_tenure_ratio.
    """
    df[MONTHLY_CHARGE_BIN] = pd.cut(
        df["MonthlyCharges"],
        bins=[-float("inf"), 35.0, 70.0, float("inf")],
        labels=MONTHLY_CHARGE_BIN_ORDER,
        right=True,
    ).astype(str)
    return df


def add_charges_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Billing anomaly score: TotalCharges − (MonthlyCharges × tenure).

    For a consistently-billed customer: TotalCharges ≈ MonthlyCharges × tenure.
    Deviations reveal billing history patterns:
        Positive gap: Customer previously paid higher monthly charges
                      (price increase or plan upgrade resentment signal).
        Negative gap: Customer received discounts or credits in the past
                      (higher satisfaction proxy — they got a good deal).
        Near zero:    Consistent billing history throughout tenure.

    Note: For tenure=0 this equals TotalCharges directly (near zero for
    new customers who haven't completed their first billing cycle).
    """
    df[CHARGES_GAP] = df["TotalCharges"] - (df["MonthlyCharges"] * df["tenure"])
    return df


# ── Group C — Service Depth ──────────────────────────────────────────────────


def add_has_internet(df: pd.DataFrame) -> pd.DataFrame:
    """Binary: 1 if the customer has any internet service (DSL or Fiber).

    Required by: add_has_no_internet_addons (dependency).
    Business rationale: Internet customers can subscribe to add-ons —
    segmenting internet vs. phone-only customers enables add-on ratio
    features that would be meaningless for phone-only customers.
    """
    df[HAS_INTERNET] = (df["InternetService"] != "No").astype(int)
    return df


def add_streaming_count(df: pd.DataFrame) -> pd.DataFrame:
    """Count of streaming services subscribed: StreamingTV + StreamingMovies (0-2).

    Business rationale: Each streaming subscription ties the customer to
    the provider's entertainment bundle. Two streaming services creates
    stronger lock-in than one — switching means losing both TV AND movies.
    """
    count = (df["StreamingTV"] == "Yes").astype(int)
    count += (df["StreamingMovies"] == "Yes").astype(int)
    df[STREAMING_COUNT] = count
    return df


def add_security_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """Count of security/protection services subscribed (0-3).

    Services: OnlineSecurity + OnlineBackup + DeviceProtection.
    Business rationale: Each security add-on increases switching cost —
    customer data, backup history, and device protection are tied to
    the provider. Three security services means the customer has
    significant digital infrastructure invested.
    """
    count = (df["OnlineSecurity"] == "Yes").astype(int)
    count += (df["OnlineBackup"] == "Yes").astype(int)
    count += (df["DeviceProtection"] == "Yes").astype(int)
    df[SECURITY_SERVICE_COUNT] = count
    return df


def add_has_full_streaming(df: pd.DataFrame) -> pd.DataFrame:
    """Binary: 1 if customer subscribes to BOTH StreamingTV AND StreamingMovies.

    Business rationale: Full streaming bundle customers are entertainment-
    dependent on the provider. The AND interaction may have non-linear
    effects beyond what streaming_count (0/1/2) captures alone —
    specifically, the marginal switching cost of losing BOTH services
    simultaneously is higher than twice the cost of losing one.
    """
    df[HAS_FULL_STREAMING] = (
        (df["StreamingTV"] == "Yes") & (df["StreamingMovies"] == "Yes")
    ).astype(int)
    return df


def add_internet_addon_count(df: pd.DataFrame) -> pd.DataFrame:
    """Count of all 6 internet add-on services subscribed (0-6).

    Add-ons: OnlineSecurity, OnlineBackup, DeviceProtection,
             TechSupport, StreamingTV, StreamingMovies.

    Business rationale: More internet add-ons = deeper product integration =
    higher switching cost. Focuses specifically on internet-tier stickiness
    (unlike service_adoption_count which includes phone too).

    Required by: add_has_no_internet_addons (dependency).
    """
    addons = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    count = pd.Series(0, index=df.index)
    for col in addons:
        count += (df[col] == "Yes").astype(int)
    df[INTERNET_ADDON_COUNT] = count
    return df


def add_has_no_internet_addons(df: pd.DataFrame) -> pd.DataFrame:
    """Binary: 1 if customer HAS internet but ZERO internet add-ons.

    Business rationale: Raw-connectivity customers have no bundled
    stickiness. They pay for internet alone with no switching cost
    from add-ons — highly price-sensitive and easy to poach.
    This is a critical at-risk segment for targeted retention offers.

    DEPENDENCY: add_has_internet and add_internet_addon_count must run first.
    """
    df[HAS_NO_INTERNET_ADDONS] = (
        (df[HAS_INTERNET] == 1) & (df[INTERNET_ADDON_COUNT] == 0)
    ).astype(int)
    return df


# ── Group D — Contract / Commitment ──────────────────────────────────────────


def add_contract_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal encoding of Contract: Month-to-month=0, One year=1, Two year=2.

    Business rationale: Contract length is an ordinal scale of commitment.
    Churn rates: 42.7% (M2M) → 11.3% (1yr) → 2.8% (2yr).
    The numeric encoding lets gradient boosters and linear models learn
    a monotonic commitment → retention relationship more efficiently
    than OHE which loses ordinality.

    Complements is_month_to_month (binary) with the full 3-level signal.
    Required by: add_is_committed_customer (dependency).
    """
    df[CONTRACT_NUMERIC] = df["Contract"].map(CONTRACT_MAP).fillna(0).astype(int)
    return df


def add_is_committed_customer(df: pd.DataFrame) -> pd.DataFrame:
    """Binary: 1 if the customer is on a 1-year or 2-year contract.

    Positive-class framing of is_month_to_month (is_committed = 1 - is_m2m).
    Business rationale: Committed customers have contractual switching
    barriers — early termination fees, re-signup friction.
    Kept as explicit feature for interpretability in SHAP and linear models.
    Feature selection will drop this if it is redundant with is_month_to_month.

    DEPENDENCY: add_contract_numeric must run first.
    """
    df[IS_COMMITTED_CUSTOMER] = (df[CONTRACT_NUMERIC] > 0).astype(int)
    return df


# ── Group E — Interaction / Compound ─────────────────────────────────────────


def add_m2m_and_fiber(df: pd.DataFrame) -> pd.DataFrame:
    """Binary: 1 if customer is BOTH Month-to-month AND Fiber optic.

    Business rationale: EDA Contract×InternetService heatmap confirmed
    this is the single highest-risk cell. M2M customers churn at 42.7%,
    Fiber customers at 41.9% — their intersection compounds both risks.
    Linear models CANNOT learn multiplicative interactions natively.
    This feature explicitly provides what the EDA heatmap shows.

    DEPENDENCY: add_is_month_to_month and add_is_fiber_optic must run first.
    """
    df[M2M_AND_FIBER] = (
        (df[IS_MONTH_TO_MONTH] == 1) & (df[IS_FIBER_OPTIC] == 1)
    ).astype(int)
    return df


def add_new_high_value_customer(df: pd.DataFrame) -> pd.DataFrame:
    """Binary: 1 if NEW (tenure ≤ 6 months) AND HIGH-VALUE (MonthlyCharges > 70).

    Business rationale: These customers are the most critical retention
    targets. They generate significant revenue but have not built switching-
    cost inertia. charge_to_tenure_ratio captures similar logic but as a
    continuous score — this explicit binary flag makes the segment directly
    actionable for targeted retention campaigns.
    """
    df[NEW_HIGH_VALUE_CUSTOMER] = (
        (df["tenure"] <= NEW_CUSTOMER_TENURE_THRESHOLD)
        & (df["MonthlyCharges"] > HIGH_MONTHLY_CHARGE_THRESHOLD)
    ).astype(int)
    return df


def add_fiber_no_security(df: pd.DataFrame) -> pd.DataFrame:
    """Binary: 1 if Fiber optic internet AND no OnlineSecurity.

    Business rationale: Fiber customers without OnlineSecurity may feel
    underserved — paying a premium price point but lacking digital protection.
    If a competitor offers bundled security at a similar price, these customers
    have high poaching risk. Captures a specific frustration-prone profile
    that neither is_fiber_optic nor OnlineSecurity alone identifies.

    DEPENDENCY: add_is_fiber_optic must run first.
    """
    df[FIBER_NO_SECURITY] = (
        (df[IS_FIBER_OPTIC] == 1) & (df["OnlineSecurity"] != "Yes")
    ).astype(int)
    return df


def add_paperless_and_echeck(df: pd.DataFrame) -> pd.DataFrame:
    """Binary: 1 if BOTH PaperlessBilling=Yes AND PaymentMethod=Electronic check.

    Both signals individually indicate higher churn:
        PaperlessBilling: 33.6% churn vs 16.3% with paper billing
        Electronic check: 45.3% churn — highest payment method
    Their combination identifies a digitally-engaged but uncommitted segment.
    Concentrates both signals into one actionable retention flag.
    """
    df[PAPERLESS_AND_ECHECK] = (
        (df["PaperlessBilling"] == "Yes") & (df["PaymentMethod"] == "Electronic check")
    ).astype(int)
    return df


# ── Group F — Composite Risk Score ───────────────────────────────────────────


def add_composite_churn_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Composite churn risk score (integer 0-7).

    Components — each contributes 1 point to the score:
        1. is_month_to_month      Strongest single predictor
        2. is_fiber_optic         High-churn infrastructure
        3. is_electronic_check    Highest-churn payment method
        4. ~has_protection_bundle Absence of the key retention bundle
        5. is_isolated            No family-based switching friction
        6. is_high_monthly_charge High price pressure on unrooted customer
        7. has_no_internet_addons Internet without add-ons = disengaged

    Score 0 = all retention factors present, minimal churn risk.
    Score 7 = all high-risk factors combined, maximum churn risk.

    Business use:
        - Segment customers into risk tiers for targeted campaigns
        - Produce explainable risk reports for business stakeholders
        - Validate model: this engineered score should rank highly in
          feature importance if its component features are meaningful

    DEPENDENCY: Must run LAST. Depends on 7 previously computed columns.
    """
    score = (
        df[IS_MONTH_TO_MONTH]
        + df[IS_FIBER_OPTIC]
        + df[IS_ELECTRONIC_CHECK]
        + (1 - df[HAS_PROTECTION_BUNDLE])
        + df[IS_ISOLATED]
        + df[IS_HIGH_MONTHLY_CHARGE]
        + df[HAS_NO_INTERNET_ADDONS]
    )
    df[COMPOSITE_CHURN_RISK] = score.astype(int)
    return df


# ---------------------------------------------------------------------------
# Master engineering function — single entry point for the pipeline.
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all 28 feature engineering transformations to the input DataFrame.

    Execution order is fixed and must not be changed.
    Dependencies are documented inline:

        service_adoption_count  →  avg_charge_per_service
        has_internet            →  has_no_internet_addons
        internet_addon_count    →  has_no_internet_addons
        is_fiber_optic          →  m2m_and_fiber, fiber_no_security
        is_month_to_month       →  m2m_and_fiber
        contract_numeric        →  is_committed_customer
        is_electronic_check     →  (used by composite_churn_risk)
        is_high_monthly_charge  →  (used by composite_churn_risk)
        has_protection_bundle   →  (used by composite_churn_risk)
        is_isolated             →  (used by composite_churn_risk)
        has_no_internet_addons  →  (used by composite_churn_risk)
        composite_churn_risk    →  MUST BE LAST

    This function is STATELESS — it never fits, computes statistics from,
    or stores any information about the data it processes. Safe to call
    identically on training, validation, test, and live inference data.

    Args:
        df: Preprocessed DataFrame (output of src.data.preprocess.preprocess).
            Must contain all original Telco feature columns. customerID
            must already be dropped (handled by preprocess).

    Returns:
        DataFrame with all original columns plus 28 new engineered columns.
        Row count is unchanged.

    Raises:
        KeyError: If a required source column is missing.
    """
    required = [
        "tenure",
        "Contract",
        "MonthlyCharges",
        "TotalCharges",
        "InternetService",
        "TechSupport",
        "OnlineSecurity",
        "PhoneService",
        "MultipleLines",
        "OnlineBackup",
        "DeviceProtection",
        "StreamingTV",
        "StreamingMovies",
        "Partner",
        "Dependents",
        "SeniorCitizen",
        "PaymentMethod",
        "PaperlessBilling",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"engineer_features: missing required columns {missing}. "
            "Ensure preprocess() has been called before engineer_features()."
        )

    df = df.copy()
    logger.info("Engineering features for DataFrame with shape %s.", df.shape)

    # ── Original 7 ──────────────────────────────────────────────────────────
    df = add_tenure_group(df)
    df = add_is_month_to_month(df)
    df = add_service_adoption_count(df)
    df = add_has_protection_bundle(df)
    df = add_charge_to_tenure_ratio(df)
    df = add_is_fiber_optic(df)
    df = add_avg_charge_per_service(df)  # needs service_adoption_count

    # ── Group A — Demographic ────────────────────────────────────────────────
    df = add_has_family(df)
    df = add_is_isolated(df)
    df = add_senior_no_support(df)

    # ── Group B — Billing / Payment ──────────────────────────────────────────
    df = add_is_auto_pay(df)
    df = add_is_electronic_check(df)
    df = add_is_high_monthly_charge(df)
    df = add_monthly_charge_bin(df)
    df = add_charges_gap(df)

    # ── Group C — Service Depth (sub-ordering matters) ───────────────────────
    df = add_has_internet(df)
    df = add_streaming_count(df)
    df = add_security_service_count(df)
    df = add_has_full_streaming(df)
    df = add_internet_addon_count(df)
    df = add_has_no_internet_addons(df)  # needs has_internet + internet_addon_count

    # ── Group D — Contract (sub-ordering matters) ────────────────────────────
    df = add_contract_numeric(df)
    df = add_is_committed_customer(df)  # needs contract_numeric

    # ── Group E — Interaction ────────────────────────────────────────────────
    df = add_m2m_and_fiber(df)  # needs is_month_to_month + is_fiber_optic
    df = add_new_high_value_customer(df)
    df = add_fiber_no_security(df)  # needs is_fiber_optic
    df = add_paperless_and_echeck(df)

    # ── Group F — Composite (MUST BE LAST) ──────────────────────────────────
    df = add_composite_churn_risk(df)

    logger.info(
        "Feature engineering complete. Output shape %s. Added %d new columns.",
        df.shape,
        len(ENGINEERED_FEATURES),
    )
    return df


# ---------------------------------------------------------------------------
# Public API for feature routing — called by pipeline.py ColumnTransformer
# ---------------------------------------------------------------------------


def get_feature_names() -> list[str]:
    """Return the complete list of feature column names after engineering."""
    return BASE_CATEGORICAL_FEATURES + BASE_NUMERICAL_FEATURES + BASE_BINARY_FEATURES


def get_categorical_features() -> list[str]:
    """Return categorical feature names for OneHotEncoder routing."""
    return BASE_CATEGORICAL_FEATURES.copy()


def get_numerical_features() -> list[str]:
    """Return numerical feature names for StandardScaler routing."""
    return BASE_NUMERICAL_FEATURES.copy()


def get_binary_features() -> list[str]:
    """Return binary 0/1 feature names for passthrough routing."""
    return BASE_BINARY_FEATURES.copy()


def get_engineered_feature_names() -> list[str]:
    """Return only the names of the 28 newly engineered columns."""
    return ENGINEERED_FEATURES.copy()
