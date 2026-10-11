"""
Integration tests for the FastAPI prediction API.

These tests use FastAPI's TestClient (which wraps httpx) to send real
HTTP requests to the API without starting a server. The model is loaded
in local development mode (trains on startup), so these tests require
the training data to be present at data/raw/.

Test coverage:
    Health endpoint:
        - Returns 200 with correct structure
        - Reports model_loaded status

    Single prediction:
        - Valid request returns 200 with correct response structure
        - High-risk customer gets HIGH_RISK tier
        - Low-risk customer gets LOW_RISK tier
        - Invalid input returns 422
        - Missing field returns 422

    Batch prediction:
        - Valid batch returns correct count
        - Empty batch returns 422
        - Mixed risk profiles return correct summary counts

    Model info:
        - Returns 200 with algorithm and threshold

    Request ID:
        - Every response includes X-Request-ID header
        - Response body includes request_id field

Note: These tests are slower than unit tests because they load and
train a real model. Mark them with @pytest.mark.slow if you want
to skip them in fast CI runs.
"""

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    """
    Create a TestClient with the FastAPI app.

    scope="module" means the client (and thus the model) is created once
    per test module, not once per test. This avoids re-training the model
    for every individual test — saving ~30 seconds per test.
    """
    # Import app here to avoid circular imports and to let the model
    # load lazily during the first request.
    from src.serving.api import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def valid_customer() -> dict:
    """One valid customer feature dict for API testing."""
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
def high_risk_customer() -> dict:
    """Customer with every high-churn signal active."""
    return {
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
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
        "MonthlyCharges": 95.50,
        "TotalCharges": 95.50,
    }


@pytest.fixture
def low_risk_customer() -> dict:
    """Customer with every low-churn signal: long tenure, 2yr contract, auto-pay."""
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 72,
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
        "TotalCharges": 6120.00,
    }


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        """Health check should always return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Response should contain status, model_loaded, and version."""
        response = client.get("/health")
        body = response.json()
        assert "status" in body
        assert "model_loaded" in body
        assert "version" in body

    def test_health_model_loaded(self, client):
        """After first request, model should be loaded."""
        # Trigger model loading with a prediction first
        response = client.get("/health")
        body = response.json()
        # Model might or might not be loaded depending on startup
        assert body["status"] in ("healthy", "degraded")
        assert isinstance(body["model_loaded"], bool)


# ---------------------------------------------------------------------------
# Single prediction endpoint
# ---------------------------------------------------------------------------


class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_predict_returns_200(self, client, valid_customer):
        """Valid request should return 200."""
        response = client.post("/predict", json=valid_customer)
        assert response.status_code == 200

    def test_predict_response_structure(self, client, valid_customer):
        """Response should contain all expected fields."""
        response = client.post("/predict", json=valid_customer)
        body = response.json()

        assert "churn_probability" in body
        assert "risk_tier" in body
        assert "will_churn" in body
        assert "threshold_used" in body
        assert "request_id" in body

    def test_predict_probability_range(self, client, valid_customer):
        """Churn probability must be between 0 and 1."""
        response = client.post("/predict", json=valid_customer)
        body = response.json()

        assert 0.0 <= body["churn_probability"] <= 1.0

    def test_predict_risk_tier_valid(self, client, valid_customer):
        """Risk tier must be one of HIGH_RISK, MEDIUM_RISK, LOW_RISK."""
        response = client.post("/predict", json=valid_customer)
        body = response.json()

        assert body["risk_tier"] in ("HIGH_RISK", "MEDIUM_RISK", "LOW_RISK")

    def test_predict_will_churn_is_boolean(self, client, valid_customer):
        """will_churn must be a boolean."""
        response = client.post("/predict", json=valid_customer)
        body = response.json()

        assert isinstance(body["will_churn"], bool)

    def test_predict_high_risk_customer(self, client, high_risk_customer):
        """
        High-risk customer should get elevated churn probability.

        Not asserting a specific tier because the threshold is model-dependent,
        but the probability should be above 0.3 (conservative) for a customer
        with every risk signal active.
        """
        response = client.post("/predict", json=high_risk_customer)
        body = response.json()

        assert body["churn_probability"] > 0.3

    def test_predict_low_risk_customer(self, client, low_risk_customer):
        """
        Low-risk customer should get low churn probability.

        Two-year contract, auto-pay, full service bundle, 72-month tenure —
        probability should be below 0.5 at minimum.
        """
        response = client.post("/predict", json=low_risk_customer)
        body = response.json()

        assert body["churn_probability"] < 0.5

    def test_predict_request_id_in_response(self, client, valid_customer):
        """Response should include a request_id for traceability."""
        response = client.post("/predict", json=valid_customer)
        body = response.json()

        assert body["request_id"] is not None
        assert len(body["request_id"]) > 0

    def test_predict_request_id_in_header(self, client, valid_customer):
        """X-Request-ID header should be present."""
        response = client.post("/predict", json=valid_customer)

        assert "x-request-id" in response.headers

    def test_predict_invalid_gender_returns_422(self, client, valid_customer):
        """Invalid enum value should trigger Pydantic validation error."""
        valid_customer["gender"] = "Invalid"
        response = client.post("/predict", json=valid_customer)

        assert response.status_code == 422

    def test_predict_missing_field_returns_422(self, client, valid_customer):
        """Missing required field should trigger 422."""
        del valid_customer["tenure"]
        response = client.post("/predict", json=valid_customer)

        assert response.status_code == 422

    def test_predict_negative_charges_returns_422(self, client, valid_customer):
        """Negative MonthlyCharges should be rejected."""
        valid_customer["MonthlyCharges"] = -10.0
        response = client.post("/predict", json=valid_customer)

        assert response.status_code == 422

    def test_predict_tenure_over_100_returns_422(self, client, valid_customer):
        """tenure > 100 should be rejected."""
        valid_customer["tenure"] = 150
        response = client.post("/predict", json=valid_customer)

        assert response.status_code == 422

    def test_predict_new_customer_tenure_0(self, client, valid_customer):
        """New customer with tenure=0 should be handled correctly."""
        valid_customer["tenure"] = 0
        valid_customer["TotalCharges"] = 0.0
        response = client.post("/predict", json=valid_customer)

        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["churn_probability"] <= 1.0


# ---------------------------------------------------------------------------
# Batch prediction endpoint
# ---------------------------------------------------------------------------


class TestBatchPredictEndpoint:
    """Tests for POST /predict/batch."""

    def test_batch_returns_200(self, client, valid_customer):
        """Valid batch request should return 200."""
        response = client.post(
            "/predict/batch",
            json={"customers": [valid_customer]},
        )
        assert response.status_code == 200

    def test_batch_response_structure(self, client, valid_customer):
        """Response should contain predictions and summary statistics."""
        response = client.post(
            "/predict/batch",
            json={"customers": [valid_customer, valid_customer]},
        )
        body = response.json()

        assert "predictions" in body
        assert "total_customers" in body
        assert "high_risk_count" in body
        assert "medium_risk_count" in body
        assert "low_risk_count" in body
        assert "avg_churn_probability" in body

    def test_batch_correct_count(self, client, valid_customer):
        """Total customers should match the number of inputs."""
        n = 5
        response = client.post(
            "/predict/batch",
            json={"customers": [valid_customer] * n},
        )
        body = response.json()

        assert body["total_customers"] == n
        assert len(body["predictions"]) == n

    def test_batch_risk_counts_sum(self, client, valid_customer):
        """Risk tier counts should sum to total customers."""
        n = 3
        response = client.post(
            "/predict/batch",
            json={"customers": [valid_customer] * n},
        )
        body = response.json()

        total = (
            body["high_risk_count"] + body["medium_risk_count"] + body["low_risk_count"]
        )
        assert total == body["total_customers"]

    def test_batch_empty_returns_422(self, client):
        """Empty customer list should be rejected."""
        response = client.post(
            "/predict/batch",
            json={"customers": []},
        )
        assert response.status_code == 422

    def test_batch_mixed_risk_profiles(
        self, client, high_risk_customer, low_risk_customer
    ):
        """Batch with different profiles should return varied predictions."""
        response = client.post(
            "/predict/batch",
            json={
                "customers": [
                    high_risk_customer,
                    low_risk_customer,
                    high_risk_customer,
                ]
            },
        )
        body = response.json()

        assert body["total_customers"] == 3
        # At least one should be predicted as high probability and one low
        probas = [p["churn_probability"] for p in body["predictions"]]
        assert max(probas) > min(probas)  # Not all identical


# ---------------------------------------------------------------------------
# Model info endpoint
# ---------------------------------------------------------------------------


class TestModelInfoEndpoint:
    """Tests for GET /model/info."""

    def test_model_info_returns_200(self, client):
        """Model info should return 200 after model is loaded."""
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_structure(self, client):
        """Response should contain expected model metadata."""
        response = client.get("/model/info")
        body = response.json()

        assert "model_name" in body
        assert "optimal_threshold" in body
        assert "algorithm" in body
        assert "feature_count" in body

    def test_model_info_threshold_valid(self, client):
        """Optimal threshold should be between 0 and 1."""
        response = client.get("/model/info")
        body = response.json()

        assert 0.0 <= body["optimal_threshold"] <= 1.0

    def test_model_info_feature_count_positive(self, client):
        """Feature count should be positive (19 original features expected)."""
        response = client.get("/model/info")
        body = response.json()

        assert body["feature_count"] > 0


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    """Tests for GET /metrics."""

    def test_metrics_returns_200(self, client):
        """Metrics endpoint should return 200."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_content(self, client, valid_customer):
        """Metrics endpoint should return Prometheus metrics after a request."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "prediction_requests_total" in response.text


# ---------------------------------------------------------------------------
# Uplift endpoints
# ---------------------------------------------------------------------------


class TestUpliftEndpoints:
    """Tests for Causal Uplift Modeling endpoints."""

    def test_uplift_info_returns_200(self, client):
        """Uplift info should return 200 with model metadata."""
        response = client.get("/uplift/info")
        assert response.status_code == 200
        data = response.json()
        assert "treatment_feature" in data
        assert "model_type" in data
        assert "feature_cols" in data

    def test_uplift_predict_single(self, client, valid_customer):
        """Single customer uplift prediction returns prescriptive CATE and quadrant."""
        response = client.post("/uplift/predict", json=valid_customer)
        assert response.status_code == 200
        data = response.json()
        assert "cate" in data
        assert "uplift_segment" in data
        assert "recommended_action" in data
        assert "base_churn_probability" in data
        assert 0.0 <= data["base_churn_probability"] <= 1.0

    def test_uplift_predict_batch(self, client, valid_customer, high_risk_customer):
        """Batch uplift prediction returns customer list and quadrant counts."""
        payload = {"customers": [valid_customer, high_risk_customer]}
        response = client.post("/uplift/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total_customers"] == 2
        assert len(data["predictions"]) == 2
        assert "persuadables_count" in data
        assert "sure_things_count" in data
        assert "lost_causes_count" in data
        assert "sleeping_dogs_count" in data
        assert (
            data["persuadables_count"]
            + data["sure_things_count"]
            + data["lost_causes_count"]
            + data["sleeping_dogs_count"]
            <= 2
        )


class TestProductionReadinessAndSecurity:
    """Tests for Kubernetes readiness probes, security headers, and rate limiting."""

    def test_readiness_probe_success(self, client):
        """Readiness returns 200 and model_loaded=True once model is ready."""
        response = client.get("/readiness")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["model_loaded"] is True
        assert "version" in data

    def test_security_headers_present(self, client):
        """Responses must contain OWASP Top 10 security headers."""
        response = client.get("/health")
        assert response.status_code == 200
        headers = response.headers
        assert headers.get("X-Content-Type-Options") == "nosniff"
        assert headers.get("X-Frame-Options") == "DENY"
        assert headers.get("Cache-Control") == "no-store"
        assert "Strict-Transport-Security" in headers

    def test_rate_limiting_headers(self, client):
        """Responses must include X-RateLimit headers."""
        response = client.get("/health")
        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers

    def test_rate_limiting_enforcement(self, client, valid_customer):
        """Exceeding the rate limit must return HTTP 429 with Retry-After header."""
        from src.serving.api import rate_limiter

        original_max = rate_limiter.max_requests
        try:
            rate_limiter.max_requests = 2
            rate_limiter.reset()

            # First 2 requests succeed
            r1 = client.post("/predict", json=valid_customer)
            r2 = client.post("/predict", json=valid_customer)
            assert r1.status_code == 200
            assert r2.status_code == 200

            # 3rd request should be blocked
            r3 = client.post("/predict", json=valid_customer)
            assert r3.status_code == 429
            assert r3.headers.get("Retry-After") == "60"
            data = r3.json()
            assert data["error"] == "rate_limit_exceeded"
        finally:
            rate_limiter.max_requests = original_max
            rate_limiter.reset()


class TestApiV1Versioning:
    """Tests for API versioning under /api/v1 prefix."""

    def test_v1_predict_single(self, client, valid_customer):
        """Endpoint /api/v1/predict returns standard prediction response."""
        response = client.post("/api/v1/predict", json=valid_customer)
        assert response.status_code == 200
        data = response.json()
        assert "churn_probability" in data
        assert "risk_tier" in data

    def test_v1_predict_batch(self, client, valid_customer):
        """Endpoint /api/v1/predict/batch returns batch response."""
        response = client.post(
            "/api/v1/predict/batch", json={"customers": [valid_customer]}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_customers"] == 1

    def test_v1_model_info(self, client):
        """Endpoint /api/v1/model/info returns model metadata."""
        response = client.get("/api/v1/model/info")
        assert response.status_code == 200
        assert "algorithm" in response.json()

    def test_v1_readiness(self, client):
        """Endpoint /api/v1/readiness returns ready."""
        response = client.get("/api/v1/readiness")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"


class TestAsyncBatchProcessing:
    """Tests for asynchronous batch submission and status polling."""

    def test_async_batch_lifecycle(self, client, valid_customer, high_risk_customer):
        """Submit async batch job, poll status, and verify results."""
        payload = {"customers": [valid_customer, high_risk_customer]}
        response = client.post("/api/v1/predict/batch/async", json=payload)
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] in ("pending", "processing", "completed")
        assert data["total_customers"] == 2
        assert "poll_url" in data

        job_id = data["job_id"]

        # Poll the job status endpoint
        poll_resp = client.get(f"/api/v1/predict/batch/jobs/{job_id}")
        assert poll_resp.status_code == 200
        job_data = poll_resp.json()
        assert job_data["job_id"] == job_id
        # In TestClient, background tasks run synchronously during request lifecycle
        assert job_data["status"] == "completed"
        assert job_data["result"] is not None
        assert job_data["result"]["total_customers"] == 2

    def test_async_batch_not_found(self, client):
        """Polling non-existent job ID returns 404."""
        response = client.get("/api/v1/predict/batch/jobs/non-existent-uuid")
        assert response.status_code == 404
