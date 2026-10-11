"""
FastAPI application for the Customer Churn Prediction API.

This is the production serving endpoint. It receives customer feature
data, runs it through the full sklearn Pipeline (feature engineering →
preprocessing → model), and returns churn probability with risk tiers.

Endpoint summary:
    POST /predict          Single customer prediction
    POST /predict/batch    Batch prediction (1–100 customers)
    GET  /health           Health check for load balancers / k8s probes
    GET  /model/info       Model metadata for debugging and audit

Architecture notes:
    - The model is loaded lazily on first request (via model_loader.py).
      This means the API starts fast and loads the model in the background.
    - Every request gets a unique request_id (UUID4) for traceability.
      This ID appears in logs, responses, and error messages — essential
      for debugging production issues across distributed systems.
    - CORS is enabled for all origins in development. In production,
      restrict origins to the Streamlit dashboard and internal tools.
    - Pydantic validation runs automatically before endpoint code executes.
      Invalid requests never reach the model — they get a 422 response
      with structured error details.

Usage:
    # Development
    make serve
    # or
    uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

    # Production (Docker)
    uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --workers 4
"""

from __future__ import annotations

import os
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Awaitable, Callable

import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src import __version__
from src.serving.model_loader import (
    get_model,
    get_model_info,
    get_threshold,
    get_uplift_info,
    get_uplift_model,
)
from src.serving.schemas import (
    BatchJobStatus,
    BatchJobStatusResponse,
    BatchJobSubmissionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchUpliftResponse,
    CustomerFeatures,
    ErrorDetail,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    RiskTier,
    UpliftPredictionResponse,
    UpliftSegment,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Prometheus Metrics
# ---------------------------------------------------------------------------

PREDICTION_REQUEST_COUNT = Counter(
    "prediction_requests_total", "Total prediction requests", ["endpoint"]
)
PREDICTION_ERROR_COUNT = Counter(
    "prediction_errors_total", "Total prediction errors", ["endpoint"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Prediction latency", ["endpoint"]
)
CHURN_PROBABILITY_HISTOGRAM = Histogram(
    "churn_prediction_probability",
    "Distribution of churn probabilities predicted",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
UPLIFT_REQUEST_COUNT = Counter(
    "uplift_requests_total", "Total causal uplift prediction requests", ["endpoint"]
)


# ---------------------------------------------------------------------------
# Application lifespan — runs on startup and shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan handler for startup and shutdown events.

    On startup: pre-load the model so the first request doesn't pay
    the loading latency. This is optional — the model loads lazily
    on first request regardless — but pre-loading gives a cleaner UX.

    On shutdown: log a clean shutdown message for audit trails.
    """
    logger.info("Starting Customer Churn Prediction API...")
    try:
        # Pre-load model during startup (optional — fails gracefully)
        get_model()
        logger.info("Model pre-loaded successfully during startup.")
    except Exception as e:
        # Don't crash the API if the model can't load at startup.
        # The /health endpoint will report model_loaded=False, and
        # prediction endpoints will return 503 until the model loads.
        logger.warning("Model pre-loading failed (will retry on first request): %s", e)

    yield  # Application runs here

    logger.info("Shutting down Customer Churn Prediction API.")


# ---------------------------------------------------------------------------
# FastAPI app instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Production REST API for predicting customer churn "
        "using a calibrated LightGBM pipeline with 28 engineered features, "
        "conformal uncertainty quantification, and causal uplift modeling. "
        "Part of the Customer Churn MLOps Platform."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware — allow all origins in development.
# In production, restrict to specific frontends.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Exception handlers — return structured JSON errors
# ---------------------------------------------------------------------------


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return structured error responses for HTTP exceptions."""
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorDetail(
            error="http_error",
            message=exc.detail,
            request_id=request_id,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unhandled exceptions — never expose stack traces."""
    request_id = getattr(request.state, "request_id", None)
    logger.error(
        "Unhandled exception [request_id=%s]: %s",
        request_id,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content=ErrorDetail(
            error="internal_error",
            message=(
                "An unexpected error occurred. " "Contact support with the request_id."
            ),
            request_id=request_id,
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Rate Limiting & API Router Configuration
# ---------------------------------------------------------------------------

RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() in (
    "true",
    "1",
    "yes",
)
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "600"))


class SlidingWindowRateLimiter:
    """
    Thread-safe in-memory sliding window rate limiter.

    Tracks request timestamps within a sliding window to provide smooth
    rate limiting without boundary burst artifacts of fixed-window counters.
    """

    def __init__(self, max_requests: int = 600, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def check(self, client_ip: str) -> tuple[bool, int]:
        now = time.time()
        cutoff = now - self.window_seconds
        valid = [t for t in self._requests.get(client_ip, []) if t > cutoff]
        if len(valid) >= self.max_requests:
            self._requests[client_ip] = valid
            return False, 0
        valid.append(now)
        self._requests[client_ip] = valid
        return True, self.max_requests - len(valid)

    def reset(self) -> None:
        """Clear rate limit history (primarily for test teardowns)."""
        self._requests.clear()


rate_limiter = SlidingWindowRateLimiter(max_requests=RATE_LIMIT_PER_MINUTE)

# API v1 Router for structured versioning
v1_router = APIRouter()


# ---------------------------------------------------------------------------
# Middleware — Request ID, Rate Limiting & OWASP Security Headers
# ---------------------------------------------------------------------------


@app.middleware("http")
async def add_request_id_and_security_headers(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """
    Inject request_id, enforce rate limiting, and attach OWASP security headers.

    The request_id is added to request.state so endpoints and exception
    handlers can include it in responses and logs.

    Rate limiting:
        Enforces RATE_LIMIT_PER_MINUTE requests per minute per IP.
        Bypasses system probes (/health, /readiness, /metrics) and documentation.
        Returns 429 Too Many Requests with Retry-After header if exceeded.

    Security headers:
        Implements OWASP API Security Top 10 hardening headers.
    """
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Check rate limit (bypass system probes and OpenAPI docs)
    path = request.url.path
    is_probe_or_doc = path in (
        "/health",
        "/readiness",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/v1/health",
        "/api/v1/readiness",
    )

    if RATE_LIMIT_ENABLED and not is_probe_or_doc:
        client_ip = request.headers.get(
            "X-Forwarded-For", request.client.host if request.client else "unknown"
        )
        if "," in client_ip:
            client_ip = client_ip.split(",")[0].strip()
        allowed, remaining = rate_limiter.check(client_ip)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content=ErrorDetail(
                    error="rate_limit_exceeded",
                    message=(
                        "Too Many Requests: Rate limit exceeded. " "Please retry later."
                    ),
                    request_id=request_id,
                ).model_dump(),
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(RATE_LIMIT_PER_MINUTE),
                    "X-RateLimit-Remaining": "0",
                    "X-Request-ID": request_id,
                    "X-Content-Type-Options": "nosniff",
                    "X-Frame-Options": "DENY",
                },
            )
    else:
        remaining = RATE_LIMIT_PER_MINUTE

    response = await call_next(request)

    # Traceability & Observability
    response.headers["X-Request-ID"] = request_id
    response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_PER_MINUTE)
    response.headers["X-RateLimit-Remaining"] = str(remaining)

    # Security headers (OWASP API Security Top 10)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )

    return response


# ---------------------------------------------------------------------------
# Fast & Lazy Resources
# ---------------------------------------------------------------------------

_feature_store: Any | None = None


def get_feature_store() -> Any:
    """Lazily load the Feast Feature Store to prevent startup delays."""
    global _feature_store
    if _feature_store is None:
        from pathlib import Path

        from feast import FeatureStore

        # The feast repo is located at src/features/feast_repo
        repo_path = Path(__file__).resolve().parent.parent / "features" / "feast_repo"
        _feature_store = FeatureStore(repo_path=str(repo_path))
    return _feature_store


_conformal_predictor: Any | None = None
_conformal_available: bool | None = None


def get_conformal_predictor() -> Any | None:
    """Lazily load the conformal predictor (returns None if not trained)."""
    global _conformal_predictor, _conformal_available
    if _conformal_available is None:
        from pathlib import Path

        try:
            from src.models.conformal import ConformalChurnPredictor

            artifacts_path = Path("models/conformal")
            if (artifacts_path / "conformal_model.joblib").exists():
                _conformal_predictor = ConformalChurnPredictor.load(artifacts_path)
                _conformal_available = True
                logger.info("Conformal predictor loaded successfully.")
            else:
                _conformal_available = False
                logger.info("No conformal artifacts found — skipping.")
        except Exception as e:
            _conformal_available = False
            logger.warning("Failed to load conformal predictor: %s", e)
    return _conformal_predictor


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _customer_to_dataframe(customer: "CustomerFeatures") -> pd.DataFrame:
    """
    Convert a Pydantic CustomerFeatures model to a pandas DataFrame.

    Uses model_dump(mode='json') to serialise enum values to their
    raw string representation (e.g. "Male" not Gender.MALE).
    The sklearn Pipeline expects plain strings for categorical columns.
    """
    # mode='json' converts enums to their .value strings automatically.
    # This is critical — the OneHotEncoder was fitted on plain strings
    # like "Male", not Enum objects like Gender.MALE.
    data = customer.model_dump(mode="json")

    # Replicate the SeniorCitizen cast from preprocess.py.
    # During training, preprocess() casts SeniorCitizen from int 0/1 to
    # str "0"/"1" so the ColumnTransformer routes it to OneHotEncoder.
    # The Pipeline's feature_engineering step does NOT do this cast —
    # it expects the cast to have already happened. So at inference time
    # we must apply the same transformation here.
    data["SeniorCitizen"] = str(data["SeniorCitizen"])

    return pd.DataFrame([data])


def _explain_predictions(pipeline, df: pd.DataFrame) -> list[dict[str, float] | None]:
    """
    Compute SHAP values for a DataFrame to identify top drivers per row.
    Returns a list of dictionaries (top 3 features and their SHAP contributions).
    """
    try:
        import shap

        base_pipeline = (
            pipeline.estimator if hasattr(pipeline, "estimator") else pipeline
        )
        classifier = base_pipeline.named_steps["classifier"]

        X_engineered = base_pipeline.named_steps["feature_engineering"].transform(df)
        X_preprocessed = base_pipeline.named_steps["preprocessor"].transform(
            X_engineered
        )
        if "feature_selection" in base_pipeline.named_steps:
            X_selected = base_pipeline.named_steps["feature_selection"].transform(
                X_preprocessed
            )
            selected_indices = base_pipeline.named_steps[
                "feature_selection"
            ].get_support(indices=True)
            feature_names = base_pipeline.named_steps[
                "preprocessor"
            ].get_feature_names_out()
            feature_names = [feature_names[i] for i in selected_indices]
        else:
            X_selected = X_preprocessed
            feature_names = base_pipeline.named_steps[
                "preprocessor"
            ].get_feature_names_out()

        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_selected)

        explanations: list[dict[str, float] | None] = []
        # Support various SHAP versions and model outputs
        sv_array = (
            shap_values[1]
            if isinstance(shap_values, list)
            else (shap_values.values if hasattr(shap_values, "values") else shap_values)
        )

        for i in range(len(df)):
            sv = sv_array[i]
            feature_contributions = {
                feat: float(val) for feat, val in zip(feature_names, sv)
            }
            top_3 = dict(
                sorted(
                    feature_contributions.items(),
                    key=lambda item: abs(item[1]),
                    reverse=True,
                )[:3]
            )
            explanations.append(top_3)

        return explanations
    except Exception as e:
        logger.warning("SHAP explanation failed: %s", e)
        return [None] * len(df)


def _predict_single(df: pd.DataFrame, request_id: str) -> PredictionResponse:
    """
    Run prediction on a single-row DataFrame and build the response.

    Steps:
        1. Get the cached model and threshold
        2. predict_proba → churn probability
        3. Apply threshold → binary decision
        4. Map probability to risk tier
        5. Build PredictionResponse with SHAP explanation
    """
    from src.models.threshold import get_risk_tier

    pipeline = get_model()
    threshold = get_threshold()

    # predict_proba returns [[p_retain, p_churn]] — we want p_churn
    proba = float(pipeline.predict_proba(df)[0, 1])
    will_churn = proba >= threshold
    risk_tier = RiskTier(get_risk_tier(proba))

    explanations = _explain_predictions(pipeline, df)
    explainability = explanations[0] if explanations else None

    # Conformal prediction (uncertainty quantification)
    conformal_result = None
    conformal = get_conformal_predictor()
    if conformal is not None:
        try:
            conformal_result = conformal.predict_single(df)
        except Exception as e:
            logger.warning("Conformal prediction failed: %s", e)

    return PredictionResponse(
        churn_probability=round(proba, 4),
        risk_tier=risk_tier,
        will_churn=will_churn,
        threshold_used=round(threshold, 4),
        explainability=explainability,
        conformal_prediction=conformal_result,
        request_id=request_id,
    )


def _process_batch_prediction(
    batch: BatchPredictionRequest, request_id: str
) -> BatchPredictionResponse:
    """
    Process batch prediction vectorised across all submitted customers.
    """
    from src.models.threshold import get_risk_tier

    all_data = []
    for customer in batch.customers:
        row = customer.model_dump(mode="json")
        row["SeniorCitizen"] = str(row["SeniorCitizen"])
        all_data.append(row)
    df = pd.DataFrame(all_data)

    pipeline = get_model()
    threshold = get_threshold()

    probas = pipeline.predict_proba(df)[:, 1]
    explanations = _explain_predictions(pipeline, df)
    conformal = get_conformal_predictor()

    predictions = []
    for i, proba in enumerate(probas):
        proba_float = float(proba)
        risk_tier = RiskTier(get_risk_tier(proba_float))
        explanation = explanations[i] if explanations else None

        conformal_result = None
        if conformal is not None:
            try:
                conformal_result = conformal.predict_single(df.iloc[[i]])
            except Exception as e:
                logger.warning("Conformal prediction failed: %s", e)

        predictions.append(
            PredictionResponse(
                churn_probability=round(proba_float, 4),
                risk_tier=risk_tier,
                will_churn=proba_float >= threshold,
                threshold_used=round(threshold, 4),
                explainability=explanation,
                conformal_prediction=conformal_result,
                request_id=request_id,
            )
        )

    high_risk = sum(1 for p in predictions if p.risk_tier == RiskTier.HIGH)
    medium_risk = sum(1 for p in predictions if p.risk_tier == RiskTier.MEDIUM)
    low_risk = sum(1 for p in predictions if p.risk_tier == RiskTier.LOW)

    for proba in probas:
        CHURN_PROBABILITY_HISTOGRAM.observe(float(proba))

    return BatchPredictionResponse(
        predictions=predictions,
        total_customers=len(predictions),
        high_risk_count=high_risk,
        medium_risk_count=medium_risk,
        low_risk_count=low_risk,
        avg_churn_probability=round(float(np.mean(probas)), 4),
        request_id=request_id,
    )


# In-memory registry for background batch prediction tasks
_BATCH_JOBS: dict[str, dict[str, Any]] = {}


def _execute_async_batch_job(
    job_id: str, batch: BatchPredictionRequest, request_id: str
) -> None:
    """Background task worker for asynchronous batch inference."""
    try:
        _BATCH_JOBS[job_id]["status"] = BatchJobStatus.PROCESSING
        start_time = time.time()
        result = _process_batch_prediction(batch, request_id)
        PREDICTION_LATENCY.labels(endpoint="predict_batch_async").observe(
            time.time() - start_time
        )
        _BATCH_JOBS[job_id]["status"] = BatchJobStatus.COMPLETED
        _BATCH_JOBS[job_id]["result"] = result
        _BATCH_JOBS[job_id]["completed_at"] = time.time()
        logger.info("Async batch job %s completed successfully.", job_id)
    except Exception as e:
        logger.error("Async batch job %s failed: %s", job_id, e, exc_info=True)
        PREDICTION_ERROR_COUNT.labels(endpoint="predict_batch_async").inc()
        _BATCH_JOBS[job_id]["status"] = BatchJobStatus.FAILED
        _BATCH_JOBS[job_id]["error"] = str(e)
        _BATCH_JOBS[job_id]["completed_at"] = time.time()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@v1_router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
    description="Returns service status and model readiness. Used by "
    "load balancers and Kubernetes liveness/readiness probes.",
)
async def health_check() -> HealthResponse:
    """
    Liveness probe endpoint.

    Always returns 200 if the process is alive. Reports model_loaded
    status for observability but does not gate on it — use /readiness
    for Kubernetes readiness probes that should block traffic until
    the model is loaded.
    """
    try:
        get_model()
        model_loaded = True
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version=__version__,
    )


@v1_router.get(
    "/readiness",
    response_model=HealthResponse,
    tags=["System"],
    summary="Readiness probe",
    description="Returns 200 only when the model is loaded and the service "
    "is ready to serve predictions. Returns 503 otherwise. "
    "Use this for Kubernetes readiness probes to prevent routing "
    "traffic before the model is available.",
)
async def readiness_check() -> HealthResponse:
    """
    Readiness probe — returns 503 until the model is fully loaded.

    Kubernetes readiness probes should point here. Unlike /health (liveness),
    this endpoint will cause the pod to be removed from the Service's
    endpoint list until the model is ready, preventing user-facing errors.
    """
    try:
        get_model()
        return HealthResponse(
            status="ready",
            model_loaded=True,
            version=__version__,
        )
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded — service is not ready to serve predictions.",
        )


@v1_router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["System"],
    summary="Model metadata",
    description="Returns model name, version, threshold, and algorithm "
    "details. Useful for debugging and audit trails.",
)
async def model_info() -> ModelInfoResponse:
    """Return metadata about the currently loaded model."""
    try:
        info = get_model_info()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return ModelInfoResponse(
        model_name=info.get("model_name", "unknown"),
        model_version=info.get("model_version"),
        optimal_threshold=info.get("optimal_threshold", 0.5),
        algorithm=info.get("algorithm", "unknown"),
        feature_count=info.get("feature_count", 0),
        mlflow_run_id=info.get("mlflow_run_id"),
    )


@app.get(
    "/metrics",
    tags=["System"],
    summary="Prometheus metrics",
    description="Returns API and model metrics scraped by Prometheus.",
)
async def get_metrics() -> Response:
    """Return Prometheus metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@v1_router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Single customer prediction",
    description=(
        "Predict churn probability for a single customer. "
        "Returns the probability, a business risk tier "
        "(HIGH/MEDIUM/LOW), and the binary churn prediction."
    ),
)
async def predict(
    customer: CustomerFeatures,
    request: Request,
) -> PredictionResponse:
    """
    Predict churn for a single customer.

    The request body must contain all 19 customer features.
    Pydantic validates types, ranges, and enum membership automatically.
    The model then applies 28 engineered features internally before
    making the prediction.
    """
    request_id = request.state.request_id

    logger.info("Prediction request received [request_id=%s]", request_id)
    PREDICTION_REQUEST_COUNT.labels(endpoint="predict").inc()
    start_time = time.time()

    try:
        df = _customer_to_dataframe(customer)
        response = _predict_single(df, request_id)

        PREDICTION_LATENCY.labels(endpoint="predict").observe(time.time() - start_time)
        CHURN_PROBABILITY_HISTOGRAM.observe(response.churn_probability)

        logger.info(
            "Prediction complete [request_id=%s]: probability=%.4f, "
            "risk_tier=%s, will_churn=%s",
            request_id,
            response.churn_probability,
            response.risk_tier.value,
            response.will_churn,
        )

        return response

    except RuntimeError as e:
        PREDICTION_ERROR_COUNT.labels(endpoint="predict").inc()
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        PREDICTION_ERROR_COUNT.labels(endpoint="predict").inc()
        logger.error(
            "Prediction failed [request_id=%s]: %s",
            request_id,
            e,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@v1_router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Predictions"],
    summary="Batch customer prediction (synchronous)",
    description=(
        "Predict churn for 1–100 customers synchronously in a single request. "
        "Returns per-customer predictions and batch summary statistics."
    ),
)
async def predict_batch(
    batch: BatchPredictionRequest,
    request: Request,
) -> BatchPredictionResponse:
    """
    Predict churn for a batch of customers synchronously.

    Vectorised through the full pipeline for optimal inference latency.
    """
    request_id = request.state.request_id

    logger.info(
        "Batch prediction request received [request_id=%s]: %d customers",
        request_id,
        len(batch.customers),
    )
    PREDICTION_REQUEST_COUNT.labels(endpoint="predict_batch").inc()
    start_time = time.time()

    try:
        response = _process_batch_prediction(batch, request_id)
        PREDICTION_LATENCY.labels(endpoint="predict_batch").observe(
            time.time() - start_time
        )
        logger.info(
            "Batch prediction complete [request_id=%s]: "
            "%d customers, avg_proba=%.4f, high=%d, medium=%d, low=%d",
            request_id,
            response.total_customers,
            response.avg_churn_probability,
            response.high_risk_count,
            response.medium_risk_count,
            response.low_risk_count,
        )
        return response

    except RuntimeError as e:
        PREDICTION_ERROR_COUNT.labels(endpoint="predict_batch").inc()
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        PREDICTION_ERROR_COUNT.labels(endpoint="predict_batch").inc()
        logger.error(
            "Batch prediction failed [request_id=%s]: %s",
            request_id,
            e,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}",
        )


@v1_router.post(
    "/predict/batch/async",
    response_model=BatchJobSubmissionResponse,
    status_code=202,
    tags=["Predictions"],
    summary="Batch customer prediction (asynchronous background job)",
    description=(
        "Submit a batch of customers for non-blocking asynchronous processing. "
        "Returns HTTP 202 Accepted with a job_id and poll URL. Designed for "
        "large customer sets and microservice decoupling."
    ),
)
async def submit_batch_async(
    batch: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    request: Request,
) -> BatchJobSubmissionResponse:
    """Submit a batch prediction job for background execution."""
    request_id = request.state.request_id
    job_id = str(uuid.uuid4())
    logger.info(
        "Async batch prediction submitted [job_id=%s, request_id=%s]: %d customers",
        job_id,
        request_id,
        len(batch.customers),
    )
    PREDICTION_REQUEST_COUNT.labels(endpoint="predict_batch_async").inc()

    _BATCH_JOBS[job_id] = {
        "job_id": job_id,
        "status": BatchJobStatus.PENDING,
        "created_at": time.time(),
        "completed_at": None,
        "total_customers": len(batch.customers),
        "result": None,
        "error": None,
    }

    background_tasks.add_task(_execute_async_batch_job, job_id, batch, request_id)

    return BatchJobSubmissionResponse(
        job_id=job_id,
        status=BatchJobStatus.PENDING,
        created_at=_BATCH_JOBS[job_id]["created_at"],
        total_customers=len(batch.customers),
        poll_url=f"/api/v1/predict/batch/jobs/{job_id}",
    )


@v1_router.get(
    "/predict/batch/jobs/{job_id}",
    response_model=BatchJobStatusResponse,
    tags=["Predictions"],
    summary="Poll status of asynchronous batch prediction job",
    description=(
        "Check the execution status and retrieve results of an async batch job."
    ),
)
async def get_batch_job_status(job_id: str) -> BatchJobStatusResponse:
    """Retrieve current status or completed result of a background batch job."""
    job = _BATCH_JOBS.get(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Batch prediction job '{job_id}' not found.",
        )
    return BatchJobStatusResponse(**job)


@v1_router.get(
    "/predict/customer/{customer_id}",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Online Feature Store Prediction (Feast)",
    description=(
        "Fetch a customer's real-time raw features from the Feast online store "
        "and immediately route them to the cached prediction pipeline."
    ),
)
async def predict_customer(
    customer_id: str,
    request: Request,
) -> PredictionResponse:
    request_id = request.state.request_id

    logger.info(
        "Feast prediction request received for %s [request_id=%s]",
        customer_id,
        request_id,
    )
    PREDICTION_REQUEST_COUNT.labels(endpoint="predict_online").inc()
    start_time = time.time()

    try:
        store = get_feature_store()

        # Request these 19 specific features for the customer entity
        feature_vector = store.get_online_features(
            features=[
                "customer_raw_features:gender",
                "customer_raw_features:SeniorCitizen",
                "customer_raw_features:Partner",
                "customer_raw_features:Dependents",
                "customer_raw_features:tenure",
                "customer_raw_features:PhoneService",
                "customer_raw_features:MultipleLines",
                "customer_raw_features:InternetService",
                "customer_raw_features:OnlineSecurity",
                "customer_raw_features:OnlineBackup",
                "customer_raw_features:DeviceProtection",
                "customer_raw_features:TechSupport",
                "customer_raw_features:StreamingTV",
                "customer_raw_features:StreamingMovies",
                "customer_raw_features:Contract",
                "customer_raw_features:PaperlessBilling",
                "customer_raw_features:PaymentMethod",
                "customer_raw_features:MonthlyCharges",
                "customer_raw_features:TotalCharges",
            ],
            entity_rows=[{"customerID": customer_id}],
        ).to_dict()

        # If 'tenure' is None, it means the entity was not found in the online store
        if feature_vector.get("tenure", [None])[0] is None:
            raise HTTPException(
                status_code=404,
                detail=f"Customer '{customer_id}' not found in online feature store.",
            )

        # Build DataFrame directly from the online features dict
        df = pd.DataFrame(feature_vector)

        # Feast injects the entity keys into the response, drop customerID
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])

        # Standardise SeniorCitizen cast precisely like _customer_to_dataframe
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

        # Pass the DataFrame to the pipeline + SHAP logic
        response = _predict_single(df, request_id)

        PREDICTION_LATENCY.labels(endpoint="predict_online").observe(
            time.time() - start_time
        )
        CHURN_PROBABILITY_HISTOGRAM.observe(response.churn_probability)

        logger.info(
            "Feast prediction complete [request_id=%s] for %s: probability=%.4f",
            request_id,
            customer_id,
            response.churn_probability,
        )

        return response

    except HTTPException:
        PREDICTION_ERROR_COUNT.labels(endpoint="predict_online").inc()
        raise
    except RuntimeError as e:
        PREDICTION_ERROR_COUNT.labels(endpoint="predict_online").inc()
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        PREDICTION_ERROR_COUNT.labels(endpoint="predict_online").inc()
        logger.error(
            "Feast prediction failed [request_id=%s]: %s",
            request_id,
            e,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Feast prediction failed: {str(e)}",
        )


# ---------------------------------------------------------------------------
# Causal Uplift (Prescriptive Retention) Endpoints
# ---------------------------------------------------------------------------


def _predict_uplift_df(
    df: pd.DataFrame, request_id: str | None = None
) -> list[UpliftPredictionResponse]:
    """
    Run Causal Uplift inference on a DataFrame of customer records.

    Calculates the Conditional Average Treatment Effect (CATE) using an EconML
    T-Learner and determines the prescriptive retention quadrant.
    """
    from src.features.feature_store import engineer_features

    uplift_model = get_uplift_model()
    if uplift_model is None:
        raise RuntimeError(
            "Causal uplift model is unavailable or failed to initialize."
        )

    info = get_uplift_info()
    feature_cols = info.get("feature_cols", [])

    df_in = df.copy()
    if "TotalCharges" in df_in.columns:
        df_in["TotalCharges"] = pd.to_numeric(
            df_in["TotalCharges"].replace(" ", ""), errors="coerce"
        ).fillna(0.0)

    df_feat = engineer_features(df_in)
    excluded_cols = ["Churn", "TechSupport", "customerID"]
    X_cols = [c for c in df_feat.columns if c not in excluded_cols]
    X = df_feat[X_cols].copy()
    X = pd.get_dummies(X, drop_first=True, dtype=float)

    if feature_cols:
        X_aligned = X.reindex(columns=feature_cols, fill_value=0.0)
    else:
        X_aligned = X

    cate_arr = uplift_model.predict_uplift(X_aligned)
    base_prob_arr = uplift_model.model.models[0].predict_proba(X_aligned.values)[:, 1]

    results: list[UpliftPredictionResponse] = []
    for i in range(len(df)):
        cate_val = round(float(cate_arr[i]), 4)
        base_prob = round(float(base_prob_arr[i]), 4)

        if base_prob >= 0.5 and cate_val <= -0.05:
            segment = UpliftSegment.PERSUADABLE
            action = "Intervene: Target customer. Proactively offer retention bundle."
        elif base_prob < 0.5 and cate_val <= 0.0:
            segment = UpliftSegment.SURE_THING
            action = (
                "No intervention: Low baseline churn risk. Conserve retention budget."
            )
        elif base_prob >= 0.5 and cate_val > -0.05:
            segment = UpliftSegment.LOST_CAUSE
            action = (
                "Do not intervene: Churn probability is not mitigated by intervention."
            )
        elif base_prob < 0.5 and cate_val > 0.0:
            segment = UpliftSegment.SLEEPING_DOG
            action = "DO NOT CONTACT: Outreach triggers negative reaction and churn."
        else:
            segment = UpliftSegment.UNKNOWN
            action = "Evaluate customer manually."

        results.append(
            UpliftPredictionResponse(
                cate=cate_val,
                uplift_segment=segment,
                recommended_action=action,
                base_churn_probability=base_prob,
                request_id=request_id,
            )
        )

    return results


@v1_router.get(
    "/uplift/info",
    tags=["Causal AI"],
    summary="Causal uplift model metadata",
    description=(
        "Returns metadata about the CATE (Conditional Average Treatment Effect) "
        "T-Learner model."
    ),
)
async def uplift_info() -> dict[str, Any]:
    """Return metadata for the causal uplift model."""
    info = get_uplift_info()
    if not info:
        raise HTTPException(
            status_code=503,
            detail="Uplift model metadata is unavailable.",
        )
    return info


@v1_router.post(
    "/uplift/predict",
    response_model=UpliftPredictionResponse,
    tags=["Causal AI"],
    summary="Single customer causal uplift prediction",
    description=(
        "Estimate the Conditional Average Treatment Effect (CATE) of offering "
        "retention support to a customer."
    ),
)
async def predict_uplift(
    customer: CustomerFeatures,
    request: Request,
) -> UpliftPredictionResponse:
    """Predict causal treatment effect and quadrant for a single customer."""
    request_id = request.state.request_id
    UPLIFT_REQUEST_COUNT.labels(endpoint="uplift_predict").inc()

    try:
        df = _customer_to_dataframe(customer)
        results = _predict_uplift_df(df, request_id)
        return results[0]
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(
            "Uplift prediction failed [request_id=%s]: %s",
            request_id,
            e,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Uplift prediction failed: {str(e)}",
        )


@v1_router.post(
    "/uplift/batch",
    response_model=BatchUpliftResponse,
    tags=["Causal AI"],
    summary="Batch causal uplift prediction",
    description=(
        "Run CATE uplift modeling across 1-100 customers with campaign "
        "quadrant distribution."
    ),
)
async def predict_uplift_batch(
    batch: BatchPredictionRequest,
    request: Request,
) -> BatchUpliftResponse:
    """Predict causal treatment effects for a batch of customers."""
    request_id = request.state.request_id
    UPLIFT_REQUEST_COUNT.labels(endpoint="uplift_batch").inc()

    try:
        all_data = []
        for customer in batch.customers:
            row = customer.model_dump(mode="json")
            row["SeniorCitizen"] = str(row["SeniorCitizen"])
            all_data.append(row)
        df = pd.DataFrame(all_data)

        predictions = _predict_uplift_df(df, request_id)
        persuadables = sum(
            1 for p in predictions if p.uplift_segment == UpliftSegment.PERSUADABLE
        )
        sure_things = sum(
            1 for p in predictions if p.uplift_segment == UpliftSegment.SURE_THING
        )
        lost_causes = sum(
            1 for p in predictions if p.uplift_segment == UpliftSegment.LOST_CAUSE
        )
        sleeping_dogs = sum(
            1 for p in predictions if p.uplift_segment == UpliftSegment.SLEEPING_DOG
        )

        return BatchUpliftResponse(
            predictions=predictions,
            total_customers=len(predictions),
            persuadables_count=persuadables,
            sure_things_count=sure_things,
            lost_causes_count=lost_causes,
            sleeping_dogs_count=sleeping_dogs,
            request_id=request_id,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(
            "Batch uplift prediction failed [request_id=%s]: %s",
            request_id,
            e,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Batch uplift prediction failed: {str(e)}",
        )


# ---------------------------------------------------------------------------
# Router Registration — Versioned Prefix & Root Backward Compatibility
# ---------------------------------------------------------------------------

# Mount under /api/v1 for production versioned API clients
app.include_router(v1_router, prefix="/api/v1")
# Mount at root for seamless backward compatibility
app.include_router(v1_router)
