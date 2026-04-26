"""
Pydantic v2 request and response schemas for the prediction API.

Every field has a description and example so the auto-generated OpenAPI
docs at /docs are immediately usable by frontend engineers, QA, and
business stakeholders — no separate API documentation needed.

Validation strategy (three layers):
    Layer 1 — Pydantic: Type coercion, range checks, enum membership.
              Runs automatically before the endpoint body executes.
              Returns 422 with structured error on failure.
    Layer 2 — Pandera InferenceSchema: Business-rule validation on the
              DataFrame built from the Pydantic model. Catches inter-field
              inconsistencies that single-field validators can't.
    Layer 3 — sklearn Pipeline: Feature engineering + model inference.
              If something slips through layers 1-2, the Pipeline's
              ColumnTransformer will raise on unexpected column types.

Public API:
    CustomerFeatures         — single customer input schema
    PredictionResponse       — single prediction output
    BatchPredictionRequest   — list of customers (max 100)
    BatchPredictionResponse  — list of predictions + summary
    HealthResponse           — health check output
    ModelInfoResponse        — model metadata output
    ErrorDetail              — structured error body
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator

from src.utils.logging import get_logger

_validator_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Enums — constrain string fields to known valid values
# ---------------------------------------------------------------------------


class Gender(str, Enum):
    """Allowed values for the gender field."""

    MALE = "Male"
    FEMALE = "Female"


class YesNo(str, Enum):
    """Binary Yes/No fields used by multiple Telco columns."""

    YES = "Yes"
    NO = "No"


class YesNoNoPhone(str, Enum):
    """MultipleLines can be Yes, No, or 'No phone service'."""

    YES = "Yes"
    NO = "No"
    NO_PHONE = "No phone service"


class YesNoNoInternet(str, Enum):
    """Internet-dependent services can be Yes, No, or 'No internet service'."""

    YES = "Yes"
    NO = "No"
    NO_INTERNET = "No internet service"


class InternetServiceType(str, Enum):
    """Types of internet service available."""

    DSL = "DSL"
    FIBER = "Fiber optic"
    NO = "No"


class ContractType(str, Enum):
    """Customer contract types, ordered by commitment level."""

    MONTH_TO_MONTH = "Month-to-month"
    ONE_YEAR = "One year"
    TWO_YEAR = "Two year"


class PaymentMethodType(str, Enum):
    """Available payment methods."""

    ELECTRONIC_CHECK = "Electronic check"
    MAILED_CHECK = "Mailed check"
    BANK_TRANSFER = "Bank transfer (automatic)"
    CREDIT_CARD = "Credit card (automatic)"


class RiskTier(str, Enum):
    """Churn risk tiers returned by the prediction API."""

    HIGH = "HIGH_RISK"
    MEDIUM = "MEDIUM_RISK"
    LOW = "LOW_RISK"


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class CustomerFeatures(BaseModel):
    """
    Input schema for a single customer's features.

    Maps 1:1 to the Telco dataset columns (minus customerID and Churn).
    All fields are required — the retention team's CRM system must provide
    a complete customer profile for accurate prediction.

    Field descriptions serve as inline documentation in the OpenAPI /docs page.
    """

    gender: Gender = Field(
        ...,
        description="Customer gender.",
        examples=["Male"],
    )
    SeniorCitizen: int = Field(
        ...,
        ge=0,
        le=1,
        description="1 if the customer is 65 or older, else 0.",
        examples=[0],
    )
    Partner: YesNo = Field(
        ...,
        description="Whether the customer has a partner.",
        examples=["Yes"],
    )
    Dependents: YesNo = Field(
        ...,
        description="Whether the customer has dependents.",
        examples=["No"],
    )
    tenure: int = Field(
        ...,
        ge=0,
        le=100,
        description="Number of months the customer has been with the company.",
        examples=[12],
    )
    PhoneService: YesNo = Field(
        ...,
        description="Whether the customer has phone service.",
        examples=["Yes"],
    )
    MultipleLines: YesNoNoPhone = Field(
        ...,
        description="Whether the customer has multiple phone lines.",
        examples=["No"],
    )
    InternetService: InternetServiceType = Field(
        ...,
        description="Type of internet service: DSL, Fiber optic, or No.",
        examples=["Fiber optic"],
    )
    OnlineSecurity: YesNoNoInternet = Field(
        ...,
        description="Whether the customer has online security add-on.",
        examples=["No"],
    )
    OnlineBackup: YesNoNoInternet = Field(
        ...,
        description="Whether the customer has online backup add-on.",
        examples=["Yes"],
    )
    DeviceProtection: YesNoNoInternet = Field(
        ...,
        description="Whether the customer has device protection add-on.",
        examples=["No"],
    )
    TechSupport: YesNoNoInternet = Field(
        ...,
        description="Whether the customer has tech support add-on.",
        examples=["No"],
    )
    StreamingTV: YesNoNoInternet = Field(
        ...,
        description="Whether the customer has streaming TV add-on.",
        examples=["No"],
    )
    StreamingMovies: YesNoNoInternet = Field(
        ...,
        description="Whether the customer has streaming movies add-on.",
        examples=["No"],
    )
    Contract: ContractType = Field(
        ...,
        description="Customer's contract type.",
        examples=["Month-to-month"],
    )
    PaperlessBilling: YesNo = Field(
        ...,
        description="Whether the customer uses paperless billing.",
        examples=["Yes"],
    )
    PaymentMethod: PaymentMethodType = Field(
        ...,
        description="Customer's payment method.",
        examples=["Electronic check"],
    )
    MonthlyCharges: float = Field(
        ...,
        gt=0,
        description="Current monthly charge amount in dollars.",
        examples=[70.35],
    )
    TotalCharges: float = Field(
        ...,
        ge=0,
        description="Total amount charged over the customer's tenure.",
        examples=[844.20],
    )

    # Cross-field validation: TotalCharges should be plausible given tenure
    @model_validator(mode="after")
    def validate_charges_consistency(self) -> "CustomerFeatures":
        """
        Sanity check: TotalCharges should not exceed:
        MonthlyCharges × (tenure + 1) × 1.5.

        The 1.5 multiplier accounts for plan upgrades, one-time fees, and
        minor billing variations. This catches data entry errors like
        TotalCharges = 100000 for a 2-month customer.
        """
        if self.tenure > 0:
            max_plausible = self.MonthlyCharges * (self.tenure + 1) * 1.5
            if self.TotalCharges > max_plausible:
                _validator_logger.warning(
                    "TotalCharges (%.2f) exceeds plausible maximum (%.2f) "
                    "for tenure=%d, MonthlyCharges=%.2f. "
                    "Prediction may be unreliable.",
                    self.TotalCharges,
                    max_plausible,
                    self.tenure,
                    self.MonthlyCharges,
                )
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
                }
            ]
        }
    }


class BatchPredictionRequest(BaseModel):
    """
    Batch prediction request containing 1–100 customer records.

    The 100-customer limit prevents the API from becoming a batch-processing
    bottleneck. For larger batches, use the training pipeline directly.
    """

    customers: list[CustomerFeatures] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of customer features (max 100).",
    )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class PredictionResponse(BaseModel):
    """
    Single customer prediction response.

    Returns the churn probability, a business-friendly risk tier,
    and the binary prediction at the model's optimal threshold.
    """

    churn_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted probability of churn (0.0–1.0).",
        examples=[0.73],
    )
    risk_tier: RiskTier = Field(
        ...,
        description=(
            "Business risk tier: HIGH_RISK (≥0.60), "
            "MEDIUM_RISK (≥0.35), LOW_RISK (<0.35)."
        ),
        examples=["HIGH_RISK"],
    )
    will_churn: bool = Field(
        ...,
        description="Binary prediction at the cost-optimal threshold.",
        examples=[True],
    )
    threshold_used: float = Field(
        ...,
        description="Decision threshold used for the binary prediction.",
        examples=[0.34],
    )
    explainability: dict[str, float] | None = Field(
        default=None,
        description="Top features driving the prediction, mapped to their SHAP values.",
        examples=[{"is_month_to_month": 0.45, "TotalCharges": -0.12, "tenure": 0.08}],
    )
    conformal_prediction: dict[str, dict] | None = Field(
        default=None,
        description=(
            "Conformal prediction sets at 90%/95% "
            "confidence levels. includes_churn and "
            "includes_no_churn indicate which classes are "
            "in the guaranteed prediction set. is_uncertain "
            "is True when both classes are included."
        ),
        examples=[
            {
                "confidence_90": {
                    "includes_churn": True,
                    "includes_no_churn": False,
                    "is_uncertain": False,
                    "set_size": 1,
                },
                "confidence_95": {
                    "includes_churn": True,
                    "includes_no_churn": True,
                    "is_uncertain": True,
                    "set_size": 2,
                },
            }
        ],
    )
    request_id: str | None = Field(
        default=None,
        description="Unique request identifier for traceability.",
    )


class BatchPredictionResponse(BaseModel):
    """
    Batch prediction response with per-customer results and summary statistics.

    The summary section gives the retention team an at-a-glance view:
    how many customers are at risk, what's the average churn probability,
    and how many need immediate outreach.
    """

    predictions: list[PredictionResponse] = Field(
        ...,
        description="Per-customer prediction results.",
    )
    total_customers: int = Field(
        ...,
        description="Number of customers processed.",
        examples=[25],
    )
    high_risk_count: int = Field(
        ...,
        description="Number of customers classified as HIGH_RISK.",
        examples=[8],
    )
    medium_risk_count: int = Field(
        ...,
        description="Number of customers classified as MEDIUM_RISK.",
        examples=[7],
    )
    low_risk_count: int = Field(
        ...,
        description="Number of customers classified as LOW_RISK.",
        examples=[10],
    )
    avg_churn_probability: float = Field(
        ...,
        description="Average churn probability across the batch.",
        examples=[0.42],
    )
    request_id: str | None = Field(
        default=None,
        description="Unique request identifier for the batch.",
    )


class HealthResponse(BaseModel):
    """Health check response — used by load balancers and k8s probes."""

    status: str = Field(
        ...,
        description="Service status: 'healthy' or 'degraded'.",
        examples=["healthy"],
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the prediction model is loaded and ready.",
        examples=[True],
    )
    version: str = Field(
        ...,
        description="API version string.",
        examples=["0.1.0"],
    )


class ModelInfoResponse(BaseModel):
    """Model metadata response — for debugging and audit trails."""

    model_name: str = Field(
        ...,
        description="Registered model name.",
        examples=["customer-churn-lgbm"],
    )
    model_version: str | None = Field(
        default=None,
        description="Model version from the registry.",
        examples=["1"],
    )
    optimal_threshold: float = Field(
        ...,
        description="Cost-optimal decision threshold.",
        examples=[0.34],
    )
    algorithm: str = Field(
        ...,
        description="ML algorithm used in the pipeline.",
        examples=["LightGBM"],
    )
    feature_count: int = Field(
        ...,
        description="Number of input features expected.",
        examples=[19],
    )
    mlflow_run_id: str | None = Field(
        default=None,
        description="MLflow run ID that produced this model.",
    )


class ErrorDetail(BaseModel):
    """Structured error response body."""

    error: str = Field(
        ...,
        description="Error category.",
        examples=["validation_error"],
    )
    message: str = Field(
        ...,
        description="Human-readable error description.",
        examples=["MonthlyCharges must be greater than 0."],
    )
    request_id: str | None = Field(
        default=None,
        description="Request identifier for support reference.",
    )
