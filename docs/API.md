# API Reference Manual

REST API reference documentation for the **Customer Churn Prediction & Causal AI Platform**.

---

## 1. Overview

The production serving engine is built with [FastAPI](https://fastapi.tiangolo.com/) and provides dual operational capabilities:
1. **Predictive Churn Engine**: Scores 30-to-90 day forward churn probabilities using an isotonic-calibrated LightGBM pipeline with SHAP local explainability and MAPIE conformal prediction uncertainty sets.
2. **Prescriptive Causal Uplift Engine**: Estimates the Conditional Average Treatment Effect (CATE) using an EconML T-Learner to recommend cost-optimal retention interventions across customer behavioral quadrants.
3. **Real-Time Feature Hydration**: Connects to the **Feast Feature Store** to score active subscribers directly by `customer_id`.
4. **Asynchronous Batch Workloads**: High-throughput non-blocking batch scoring via background task workers with job status polling.

**API Versioning Strategy:**
- **Canonical API v1 Routes**: Prefixed with `/api/v1/*` (e.g., `/api/v1/predict`, `/api/v1/predict/batch`, `/api/v1/model/info`).
- **Backward-Compatible Root Aliases**: Top-level routes (e.g., `/predict`) remain active for seamless legacy integration.
- **Base URLs**:
  - Local Development: `http://localhost:8000`
  - Docker Compose: `http://api:8000` (internal network) / `http://localhost:8000` (host port)
  - Interactive Swagger UI: [`http://localhost:8000/docs`](http://localhost:8000/docs)
  - Interactive ReDoc: [`http://localhost:8000/redoc`](http://localhost:8000/redoc)

---

## 2. Authentication, Security & Headers

In local and testing modes, endpoints are open. In production, authenticate requests using an `Authorization: Bearer <JWT>` header via reverse-proxy or API Gateway.

### Request Tracing & Observability
Every API request receives a unique UUID4 trace identifier:
- Returned in HTTP response headers: `X-Request-ID: 7a9e1d84-4b53-48b2-b1e8-782d49e1501b`
- Included in all response JSON payloads: `"request_id": "7a9e1d84-4b53-48b2-b1e8-782d49e1501b"`
- Automatically injected into server logs for distributed cross-system debugging.

### OWASP API Security Top 10 Headers
All responses automatically include defense-in-depth security headers:
- `X-Content-Type-Options: nosniff` (MIME sniffing prevention)
- `X-Frame-Options: DENY` (Clickjacking mitigation)
- `Cache-Control: no-store` (Prevents downstream caching of sensitive predictions)
- `Strict-Transport-Security: max-age=31536000; includeSubDomains` (Enforces HTTPS)

### Traffic Management & Rate Limiting
A thread-safe in-memory sliding-window rate limiter protects endpoints against resource exhaustion and denial-of-service:
- **Default Limit**: `600` requests per minute per client IP (configurable via `RATE_LIMIT_PER_MINUTE`).
- **Headers Returned**:
  - `X-RateLimit-Limit: 600`
  - `X-RateLimit-Remaining: 598`
- **Exceeded Threshold**: Returns HTTP `429 Too Many Requests` with a `Retry-After: 60` header and structured error detail.
- **Probe Bypass**: System probes (`/health`, `/readiness`, `/metrics`) and OpenAPI documentation are exempted from rate limiting.

---

## 3. Serving Endpoints Summary

| Method | Path | Category | Description |
|---|---|---|---|
| `POST` | `/api/v1/predict` | Predictive ML | Single customer prediction with SHAP & Conformal bounds |
| `POST` | `/api/v1/predict/batch` | Predictive ML | Synchronous vectorized batch prediction (1–100 customers) |
| `POST` | `/api/v1/predict/batch/async` | Asynchronous ML | Non-blocking async batch job submission (returns HTTP 202) |
| `GET`  | `/api/v1/predict/batch/jobs/{id}` | Asynchronous ML | Poll status and retrieve completed async batch predictions |
| `GET`  | `/api/v1/predict/customer/{id}` | Feature Store | Real-time prediction fetching raw features from Feast online store |
| `POST` | `/api/v1/uplift/predict` | Causal AI | Single customer CATE uplift & prescriptive intervention recommendation |
| `POST` | `/api/v1/uplift/batch` | Causal AI | Batch prescriptive campaign assignment with quadrant summary |
| `GET`  | `/api/v1/uplift/info` | Causal AI | Causal model metadata (treatment feature, model type, feature count) |
| `GET`  | `/api/v1/model/info` | Diagnostics | Production model metadata (algorithm, version, optimal threshold) |
| `GET`  | `/health` | System Probe | Liveness probe returning 200 if container process is alive |
| `GET`  | `/readiness` | System Probe | Readiness probe (returns 503 until model pipeline is loaded) |
| `GET`  | `/metrics` | Observability | Prometheus metrics scraping endpoint |


---

## 4. Predictive Churn Endpoints

### POST `/predict` — Single Customer Prediction

Evaluates a single customer record through the full 47-feature pipeline (19 raw + 28 engineered domain features), returning calibrated probability, risk tier, local SHAP explanation, and distribution-free conformal prediction bounds.

#### Request Schema (`CustomerFeatures`)

```json
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
  "TotalCharges": 844.20
}
```

#### Field Constraints
- `gender`: `"Male" | "Female"`
- `SeniorCitizen`: `0 | 1`
- `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`: `"Yes" | "No"`
- `MultipleLines`: `"Yes" | "No" | "No phone service"`
- `InternetService`: `"DSL" | "Fiber optic" | "No"`
- `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`: `"Yes" | "No" | "No internet service"`
- `Contract`: `"Month-to-month" | "One year" | "Two year"`
- `PaymentMethod`: `"Electronic check" | "Mailed check" | "Bank transfer (automatic)" | "Credit card (automatic)"`
- `tenure`: integer $\ge 0$
- `MonthlyCharges`: float $> 0$
- `TotalCharges`: float $\ge 0$ (validated: $\text{TotalCharges} \le \text{MonthlyCharges} \times (\text{tenure} + 1) \times 1.5$)

#### cURL Example
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "gender": "Female",
       "SeniorCitizen": 1,
       "Partner": "No",
       "Dependents": "No",
       "tenure": 2,
       "PhoneService": "Yes",
       "MultipleLines": "No",
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
       "TotalCharges": 191.00
     }'
```

#### Response Payload (`PredictionResponse`)
```json
{
  "churn_probability": 0.7421,
  "risk_tier": "HIGH_RISK",
  "will_churn": true,
  "threshold_used": 0.34,
  "explainability": {
    "is_month_to_month": 0.312,
    "fiber_no_security": 0.184,
    "tenure": -0.142
  },
  "conformal_prediction": {
    "prediction_set": ["Churn"],
    "confidence_level": 0.90,
    "is_uncertain": false
  },
  "request_id": "7a9e1d84-4b53-48b2-b1e8-782d49e1501b"
}
```

---

### POST `/predict/batch` — Batch Prediction

Processes 1 to 100 customer records in an efficient, vectorized batch call.

#### cURL Example
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "customers": [
         {
           "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
           "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
           "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
           "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
           "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
           "PaymentMethod": "Electronic check", "MonthlyCharges": 70.35, "TotalCharges": 844.20
         },
         {
           "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
           "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
           "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
           "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
           "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
           "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 55.00, "TotalCharges": 3300.00
         }
       ]
     }'
```

#### Response Payload (`BatchPredictionResponse`)
```json
{
  "predictions": [
    {
      "churn_probability": 0.5843,
      "risk_tier": "HIGH_RISK",
      "will_churn": true,
      "threshold_used": 0.34,
      "explainability": { "is_month_to_month": 0.281, "tenure": -0.112 },
      "conformal_prediction": { "prediction_set": ["Churn"], "confidence_level": 0.90, "is_uncertain": false },
      "request_id": "8b0f2e95-5c64-49c3-c2f9-893e50f2612c"
    },
    {
      "churn_probability": 0.0412,
      "risk_tier": "LOW_RISK",
      "will_churn": false,
      "threshold_used": 0.34,
      "explainability": { "is_committed_customer": -0.345, "tenure": -0.210 },
      "conformal_prediction": { "prediction_set": ["Stay"], "confidence_level": 0.90, "is_uncertain": false },
      "request_id": "8b0f2e95-5c64-49c3-c2f9-893e50f2612c"
    }
  ],
  "total_customers": 2,
  "high_risk_count": 1,
  "medium_risk_count": 0,
  "low_risk_count": 1,
  "avg_churn_probability": 0.3128,
  "request_id": "8b0f2e95-5c64-49c3-c2f9-893e50f2612c"
}
```

---

### POST `/api/v1/predict/batch/async` — Asynchronous Batch Scoring (Non-Blocking)

Submits customer records for asynchronous background evaluation. Returns immediately with HTTP `202 Accepted`, a unique `job_id`, and a status polling URL. Essential for high-throughput batch scoring without HTTP timeout risks.

#### cURL Example
```bash
curl -X POST "http://localhost:8000/api/v1/predict/batch/async" \
     -H "Content-Type: application/json" \
     -d '{
       "customers": [
         {
           "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
           "tenure": 8, "PhoneService": "Yes", "MultipleLines": "No",
           "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
           "DeviceProtection": "Yes", "TechSupport": "No", "StreamingTV": "Yes",
           "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
           "PaymentMethod": "Electronic check", "MonthlyCharges": 85.00, "TotalCharges": 680.00
         }
       ]
     }'
```

#### Response Payload (`202 Accepted`)
```json
{
  "job_id": "6a9e1d84-4b53-48b2-b1e8-782d49e1501b",
  "status": "pending",
  "created_at": 1773289000.12,
  "total_customers": 1,
  "poll_url": "/api/v1/predict/batch/jobs/6a9e1d84-4b53-48b2-b1e8-782d49e1501b"
}
```

---

### GET `/api/v1/predict/batch/jobs/{job_id}` — Poll Asynchronous Batch Job Status

Polls the execution status and retrieves completed predictions for an asynchronous batch job.

#### cURL Example
```bash
curl -X GET "http://localhost:8000/api/v1/predict/batch/jobs/6a9e1d84-4b53-48b2-b1e8-782d49e1501b"
```

#### Response Payload (`BatchJobStatusResponse`)
```json
{
  "job_id": "6a9e1d84-4b53-48b2-b1e8-782d49e1501b",
  "status": "completed",
  "created_at": 1773289000.12,
  "completed_at": 1773289000.45,
  "total_customers": 1,
  "result": {
    "predictions": [
      {
        "churn_probability": 0.6920,
        "risk_tier": "HIGH_RISK",
        "will_churn": true,
        "threshold_used": 0.34,
        "explainability": { "is_month_to_month": 0.295 },
        "conformal_prediction": { "prediction_set": ["Churn"], "confidence_level": 0.90, "is_uncertain": false },
        "request_id": "c1f83a21-9d22-48a1-b844-3329fa014389"
      }
    ],
    "total_customers": 1,
    "high_risk_count": 1,
    "medium_risk_count": 0,
    "low_risk_count": 0,
    "avg_churn_probability": 0.6920,
    "request_id": "c1f83a21-9d22-48a1-b844-3329fa014389"
  },
  "error": null
}
```

---


### GET `/predict/customer/{customer_id}` — Feast Feature Store Real-Time Prediction

Fetches live entity attributes for a customer directly from the **Feast Online Store** (SQLite/Redis), hydrating all 19 raw features and routing through the serving pipeline in $< 10$ milliseconds.

#### cURL Example
```bash
curl -X GET "http://localhost:8000/predict/customer/7590-VHVEG"
```

#### Response Payload
Identical to `POST /predict`. Returns `404 Not Found` if the customer entity has not been materialized to Feast.

---

## 5. Causal Uplift (Prescriptive Retention) Endpoints

### POST `/uplift/predict` — Single Customer Causal Intervention Prediction

Estimates the **Conditional Average Treatment Effect (CATE)** of offering a proactive retention package (TechSupport bundle) to an eligible subscriber. Rather than just asking *"Will this customer leave?"*, it determines *"Will offering this intervention save them, or will it provoke them to leave?"*

#### cURL Example
```bash
curl -X POST "http://localhost:8000/uplift/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "gender": "Female",
       "SeniorCitizen": 0,
       "Partner": "No",
       "Dependents": "No",
       "tenure": 4,
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
       "MonthlyCharges": 74.50,
       "TotalCharges": 298.00
     }'
```

#### Response Payload (`UpliftPredictionResponse`)
```json
{
  "cate": -0.1245,
  "uplift_segment": "Persuadable (Target)",
  "recommended_action": "Intervene: Target customer. Proactively offer retention bundle.",
  "base_churn_probability": 0.6840,
  "request_id": "9c1a3b84-6d75-40e4-d3ba-904f61a3723d"
}
```

#### Prescriptive Quadrants
1. **Persuadables (`uplift_segment = "Persuadable (Target)"`)**:
   - Condition: $p_{\text{base}} \ge 0.50 \land \text{CATE} \le -0.05$
   - Strategy: **Top retention priority**. Intervention significantly reduces churn probability.
2. **Sure Things (`uplift_segment = "Sure Thing (Ignore)"`)**:
   - Condition: $p_{\text{base}} < 0.50 \land \text{CATE} \le 0.00$
   - Strategy: **Do not intervene**. Customer will stay organic; discounts waste budget.
3. **Lost Causes (`uplift_segment = "Lost Cause (Ignore)"`)**:
   - Condition: $p_{\text{base}} \ge 0.50 \land \text{CATE} > -0.05$
   - Strategy: **Do not intervene**. High churn probability is insensitive to standard retention perks.
4. **Sleeping Dogs (`uplift_segment = "Sleeping Dog (Do Not Disturb)"`)**:
   - Condition: $p_{\text{base}} < 0.50 \land \text{CATE} > 0.00$
   - Strategy: **STRICT SUPPRESSION**. Contacting these customers triggers dissatisfaction or plan comparison, inducing churn.

---

### POST `/uplift/batch` — Batch Causal Uplift Allocation

Evaluates 1 to 100 customers and aggregates campaign allocation counts across the four prescriptive quadrants.

#### Response Payload (`BatchUpliftResponse`)
```json
{
  "predictions": [ ...list of UpliftPredictionResponse... ],
  "total_customers": 50,
  "persuadables_count": 14,
  "sure_things_count": 26,
  "lost_causes_count": 7,
  "sleeping_dogs_count": 3,
  "request_id": "a2d4f618-7e86-41f5-e4cb-015a72b4834e"
}
```

---

### GET `/uplift/info` — Causal Model Metadata

Returns architecture and feature alignment specifications for the loaded EconML T-Learner model.

#### Response Payload
```json
{
  "feature_cols": [
    "tenure", "MonthlyCharges", "TotalCharges", "is_month_to_month",
    "service_adoption_count", "has_protection_bundle", ...
  ],
  "treatment_feature": "TechSupport",
  "model_type": "TLearner (LGBMClassifier)",
  "eligible_customers": 3473
}
```

---

## 6. System & Observability Endpoints

### GET `/health` — Operational Liveness Check

Used by Kubernetes liveness probes (`/health`) to confirm that the FastAPI worker process is running and accepting HTTP requests. Returns HTTP `200` with `status: "healthy"` (or `"degraded"` if model loading encountered an error).

```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### GET `/readiness` — Model Readiness Check

Used by Kubernetes readiness probes (`/readiness`) to guarantee that the inference engine and ML model artifacts are fully loaded in memory before the Pod receives network ingress traffic.

- **HTTP `200 OK`**: Service is fully initialized and actively serving predictions.
- **HTTP `503 Service Unavailable`**: Artifacts are still hydrating; Kubernetes removes Pod from Service endpoint pool until ready.

```json
{
  "status": "ready",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### GET `/model/info` — Production Model Audit Metadata

```json
{
  "model_name": "customer-churn-lgbm",
  "model_version": "1.0.0-calibrated",
  "mlflow_run_id": "b3e94a8f102c4819a71db3f05",
  "optimal_threshold": 0.34,
  "algorithm": "LightGBMClassifier",
  "feature_count": 47
}
```

### GET `/metrics` — Prometheus Metrics Scraping

Exposes Prometheus format metrics for real-time observability:
- `prediction_requests_total{endpoint="..."}`: Counter tracking requests per endpoint.
- `prediction_errors_total{endpoint="..."}`: Counter tracking error responses.
- `prediction_latency_seconds{endpoint="..."}`: Histogram of response latencies.
- `churn_prediction_probability`: Histogram distribution of predicted probabilities.
- `uplift_requests_total{endpoint="..."}`: Counter tracking causal inference requests.

---

## 7. Python Client Example (Complete Workflow)

```python
"""
End-to-end Python client demonstration:
1. Churn risk scoring with SHAP & Conformal bounds
2. Prescriptive Causal Uplift decision routing
"""

import requests

BASE_URL = "http://localhost:8000"

customer_payload = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 79.85,
    "TotalCharges": 239.55,
}

# 1. Step 1: Predict Churn Risk
pred_resp = requests.post(f"{BASE_URL}/predict", json=customer_payload)
pred_resp.raise_for_status()
pred = pred_resp.json()

print(f"--- PREDICTION [Request ID: {pred['request_id']}] ---")
print(f"Churn Probability : {pred['churn_probability']:.2%}")
print(f"Risk Tier         : {pred['risk_tier']}")
print(f"Optimal Threshold : {pred['threshold_used']}")
print(f"Top Drivers       : {pred['explainability']}")
print(f"Conformal Set     : {pred['conformal_prediction']['prediction_set']}\n")

# 2. Step 2: Query Causal Uplift for Actionable Retention Policy
uplift_resp = requests.post(f"{BASE_URL}/uplift/predict", json=customer_payload)
uplift_resp.raise_for_status()
uplift = uplift_resp.json()

print(f"--- PRESCRIPTIVE ACTION ---")
print(f"Estimated CATE    : {uplift['cate']:+.4f} (churn reduction effect)")
print(f"Quadrant          : {uplift['uplift_segment']}")
print(f"Action Policy     : {uplift['recommended_action']}")
```
