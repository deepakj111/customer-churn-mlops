"""
Unit tests for Pydantic request/response schemas.

Tests that valid data parses correctly, invalid data is rejected,
and enum constraints are enforced. These run without any ML model
or database — pure schema validation logic.

Test coverage:
    CustomerFeatures:
        - Valid input constructs correctly
        - All enum fields accept documented values
        - Invalid enum values raise ValidationError
        - Range constraints (tenure, MonthlyCharges) enforced
        - Missing required fields raise ValidationError
        - Serialization round-trip preserves values

    PredictionResponse:
        - Valid output constructs correctly
        - Probability bounds enforced (0–1)

    BatchPredictionRequest:
        - Accepts 1–100 customers
        - Rejects empty list
        - Rejects >100 customers

    HealthResponse / ModelInfoResponse:
        - Basic construction works
"""

import pytest
from pydantic import ValidationError

from src.serving.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    ErrorDetail,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    RiskTier,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_customer_data() -> dict:
    """One completely valid customer feature dict."""
    return {
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


@pytest.fixture
def high_risk_customer_data() -> dict:
    """A high-risk customer: M2M, Fiber, Electronic check, new, high charges."""
    return {
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.50,
        "TotalCharges": 191.00,
    }


@pytest.fixture
def low_risk_customer_data() -> dict:
    """A low-risk customer: 2yr contract, DSL, auto-pay, long tenure."""
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
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
        "MonthlyCharges": 85.00,
        "TotalCharges": 5100.00,
    }


# ---------------------------------------------------------------------------
# CustomerFeatures — happy path
# ---------------------------------------------------------------------------


class TestCustomerFeaturesValid:
    """Verify that properly formatted customer data parses correctly."""

    def test_valid_data_constructs(self, valid_customer_data):
        """Standard valid input should create a CustomerFeatures instance."""
        customer = CustomerFeatures(**valid_customer_data)
        assert customer.gender.value == "Male"
        assert customer.tenure == 12
        assert customer.MonthlyCharges == 70.35

    def test_high_risk_customer_constructs(self, high_risk_customer_data):
        """High risk profile should parse without errors."""
        customer = CustomerFeatures(**high_risk_customer_data)
        assert customer.SeniorCitizen == 1
        assert customer.Contract.value == "Month-to-month"

    def test_low_risk_customer_constructs(self, low_risk_customer_data):
        """Low risk profile should parse without errors."""
        customer = CustomerFeatures(**low_risk_customer_data)
        assert customer.tenure == 60
        assert customer.Contract.value == "Two year"

    def test_tenure_zero_valid(self, valid_customer_data):
        """New customers with tenure=0 are valid."""
        valid_customer_data["tenure"] = 0
        valid_customer_data["TotalCharges"] = 0.0
        customer = CustomerFeatures(**valid_customer_data)
        assert customer.tenure == 0

    def test_no_internet_services_valid(self, valid_customer_data):
        """Customers with no internet have 'No internet service' for add-ons."""
        valid_customer_data["InternetService"] = "No"
        for field in [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]:
            valid_customer_data[field] = "No internet service"
        customer = CustomerFeatures(**valid_customer_data)
        assert customer.InternetService.value == "No"

    def test_model_dump_returns_dict(self, valid_customer_data):
        """model_dump() should return a dict with all fields."""
        customer = CustomerFeatures(**valid_customer_data)
        dump = customer.model_dump()
        assert isinstance(dump, dict)
        assert len(dump) == 19
        assert "gender" in dump
        assert "TotalCharges" in dump

    def test_all_payment_methods_accepted(self, valid_customer_data):
        """All four payment methods from the Telco dataset should be valid."""
        methods = [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ]
        for method in methods:
            valid_customer_data["PaymentMethod"] = method
            customer = CustomerFeatures(**valid_customer_data)
            assert customer.PaymentMethod.value == method


# ---------------------------------------------------------------------------
# CustomerFeatures — validation errors
# ---------------------------------------------------------------------------


class TestCustomerFeaturesInvalid:
    """Verify that invalid data is properly rejected by Pydantic."""

    def test_invalid_gender_rejected(self, valid_customer_data):
        """Gender must be 'Male' or 'Female'."""
        valid_customer_data["gender"] = "Other"
        with pytest.raises(ValidationError) as exc_info:
            CustomerFeatures(**valid_customer_data)
        assert "gender" in str(exc_info.value)

    def test_senior_citizen_must_be_0_or_1(self, valid_customer_data):
        """SeniorCitizen is binary, not a general integer."""
        valid_customer_data["SeniorCitizen"] = 2
        with pytest.raises(ValidationError):
            CustomerFeatures(**valid_customer_data)

    def test_negative_tenure_rejected(self, valid_customer_data):
        """Tenure cannot be negative."""
        valid_customer_data["tenure"] = -1
        with pytest.raises(ValidationError):
            CustomerFeatures(**valid_customer_data)

    def test_tenure_over_100_rejected(self, valid_customer_data):
        """Tenure > 100 months is implausible for this dataset."""
        valid_customer_data["tenure"] = 101
        with pytest.raises(ValidationError):
            CustomerFeatures(**valid_customer_data)

    def test_zero_monthly_charges_rejected(self, valid_customer_data):
        """MonthlyCharges must be > 0."""
        valid_customer_data["MonthlyCharges"] = 0
        with pytest.raises(ValidationError):
            CustomerFeatures(**valid_customer_data)

    def test_negative_monthly_charges_rejected(self, valid_customer_data):
        """Negative charges make no business sense."""
        valid_customer_data["MonthlyCharges"] = -10.0
        with pytest.raises(ValidationError):
            CustomerFeatures(**valid_customer_data)

    def test_negative_total_charges_rejected(self, valid_customer_data):
        """TotalCharges must be >= 0."""
        valid_customer_data["TotalCharges"] = -1.0
        with pytest.raises(ValidationError):
            CustomerFeatures(**valid_customer_data)

    def test_missing_required_field_rejected(self, valid_customer_data):
        """Every field is required — missing one should raise."""
        del valid_customer_data["tenure"]
        with pytest.raises(ValidationError) as exc_info:
            CustomerFeatures(**valid_customer_data)
        assert "tenure" in str(exc_info.value)

    def test_invalid_contract_type_rejected(self, valid_customer_data):
        """Contract must be one of the three valid types."""
        valid_customer_data["Contract"] = "Three year"
        with pytest.raises(ValidationError):
            CustomerFeatures(**valid_customer_data)

    def test_invalid_internet_service_rejected(self, valid_customer_data):
        """InternetService must be DSL, Fiber optic, or No."""
        valid_customer_data["InternetService"] = "Cable"
        with pytest.raises(ValidationError):
            CustomerFeatures(**valid_customer_data)

    def test_invalid_payment_method_rejected(self, valid_customer_data):
        """PaymentMethod must be one of the four valid options."""
        valid_customer_data["PaymentMethod"] = "Bitcoin"
        with pytest.raises(ValidationError):
            CustomerFeatures(**valid_customer_data)

    def test_wrong_type_tenure_rejected(self, valid_customer_data):
        """Tenure must be an integer, not a string."""
        valid_customer_data["tenure"] = "twelve"
        with pytest.raises(ValidationError):
            CustomerFeatures(**valid_customer_data)


# ---------------------------------------------------------------------------
# PredictionResponse
# ---------------------------------------------------------------------------


class TestPredictionResponse:
    """Verify prediction response schema validation."""

    def test_valid_response_constructs(self):
        """Standard prediction output should construct without error."""
        response = PredictionResponse(
            churn_probability=0.73,
            risk_tier=RiskTier.HIGH,
            will_churn=True,
            threshold_used=0.34,
            request_id="abc-123",
        )
        assert response.churn_probability == 0.73
        assert response.risk_tier == RiskTier.HIGH
        assert response.will_churn is True

    def test_probability_below_zero_rejected(self):
        """Churn probability must be >= 0."""
        with pytest.raises(ValidationError):
            PredictionResponse(
                churn_probability=-0.1,
                risk_tier=RiskTier.LOW,
                will_churn=False,
                threshold_used=0.5,
            )

    def test_probability_above_one_rejected(self):
        """Churn probability must be <= 1."""
        with pytest.raises(ValidationError):
            PredictionResponse(
                churn_probability=1.5,
                risk_tier=RiskTier.HIGH,
                will_churn=True,
                threshold_used=0.5,
            )

    def test_request_id_optional(self):
        """request_id should default to None when not provided."""
        response = PredictionResponse(
            churn_probability=0.5,
            risk_tier=RiskTier.MEDIUM,
            will_churn=True,
            threshold_used=0.34,
        )
        assert response.request_id is None

    def test_all_risk_tiers_valid(self):
        """All three risk tier enum values should be constructable."""
        for tier in RiskTier:
            response = PredictionResponse(
                churn_probability=0.5,
                risk_tier=tier,
                will_churn=True,
                threshold_used=0.5,
            )
            assert response.risk_tier == tier


# ---------------------------------------------------------------------------
# BatchPredictionRequest
# ---------------------------------------------------------------------------


class TestBatchPredictionRequest:
    """Verify batch request constraints."""

    def test_single_customer_batch_valid(self, valid_customer_data):
        """Minimum batch of 1 customer should be accepted."""
        batch = BatchPredictionRequest(
            customers=[CustomerFeatures(**valid_customer_data)]
        )
        assert len(batch.customers) == 1

    def test_multiple_customers_valid(self, valid_customer_data):
        """Batch of 5 customers should work."""
        customers = [CustomerFeatures(**valid_customer_data) for _ in range(5)]
        batch = BatchPredictionRequest(customers=customers)
        assert len(batch.customers) == 5

    def test_empty_batch_rejected(self):
        """Empty customer list should be rejected (min_length=1)."""
        with pytest.raises(ValidationError):
            BatchPredictionRequest(customers=[])

    def test_over_100_customers_rejected(self, valid_customer_data):
        """Batch > 100 customers should be rejected (max_length=100)."""
        customers = [CustomerFeatures(**valid_customer_data) for _ in range(101)]
        with pytest.raises(ValidationError):
            BatchPredictionRequest(customers=customers)

    def test_exactly_100_customers_valid(self, valid_customer_data):
        """100 customers (the limit) should be accepted."""
        customers = [CustomerFeatures(**valid_customer_data) for _ in range(100)]
        batch = BatchPredictionRequest(customers=customers)
        assert len(batch.customers) == 100


# ---------------------------------------------------------------------------
# BatchPredictionResponse
# ---------------------------------------------------------------------------


class TestBatchPredictionResponse:
    """Verify batch response construction."""

    def test_valid_batch_response_constructs(self):
        """Should construct with predictions and summary stats."""
        predictions = [
            PredictionResponse(
                churn_probability=0.8,
                risk_tier=RiskTier.HIGH,
                will_churn=True,
                threshold_used=0.34,
            ),
            PredictionResponse(
                churn_probability=0.2,
                risk_tier=RiskTier.LOW,
                will_churn=False,
                threshold_used=0.34,
            ),
        ]
        response = BatchPredictionResponse(
            predictions=predictions,
            total_customers=2,
            high_risk_count=1,
            medium_risk_count=0,
            low_risk_count=1,
            avg_churn_probability=0.5,
        )
        assert response.total_customers == 2
        assert response.high_risk_count == 1


# ---------------------------------------------------------------------------
# HealthResponse
# ---------------------------------------------------------------------------


class TestHealthResponse:
    """Verify health response schema."""

    def test_healthy_response(self):
        """Should construct a healthy response."""
        response = HealthResponse(
            status="healthy",
            model_loaded=True,
            version="0.1.0",
        )
        assert response.status == "healthy"
        assert response.model_loaded is True

    def test_degraded_response(self):
        """Should construct a degraded response when model not loaded."""
        response = HealthResponse(
            status="degraded",
            model_loaded=False,
            version="0.1.0",
        )
        assert response.status == "degraded"
        assert response.model_loaded is False


# ---------------------------------------------------------------------------
# ModelInfoResponse
# ---------------------------------------------------------------------------


class TestModelInfoResponse:
    """Verify model info response schema."""

    def test_full_response(self):
        """Should construct with all fields populated."""
        response = ModelInfoResponse(
            model_name="customer-churn-lgbm",
            model_version="3",
            optimal_threshold=0.34,
            algorithm="LightGBM",
            feature_count=19,
            mlflow_run_id="abc123",
        )
        assert response.model_name == "customer-churn-lgbm"
        assert response.feature_count == 19

    def test_optional_fields_default_to_none(self):
        """model_version and mlflow_run_id should default to None."""
        response = ModelInfoResponse(
            model_name="test",
            optimal_threshold=0.5,
            algorithm="LightGBM",
            feature_count=19,
        )
        assert response.model_version is None
        assert response.mlflow_run_id is None


# ---------------------------------------------------------------------------
# ErrorDetail
# ---------------------------------------------------------------------------


class TestErrorDetail:
    """Verify error detail schema."""

    def test_error_detail_constructs(self):
        """Should construct a structured error."""
        error = ErrorDetail(
            error="validation_error",
            message="tenure must be >= 0",
            request_id="req-123",
        )
        assert error.error == "validation_error"
        assert "tenure" in error.message

    def test_request_id_optional(self):
        """request_id should default to None."""
        error = ErrorDetail(
            error="internal_error",
            message="Something went wrong.",
        )
        assert error.request_id is None
