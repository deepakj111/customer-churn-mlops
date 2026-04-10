# API Documentation

Detailed documentation for the Customer Churn Prediction API.

---

## Overview

The prediction API is built with [FastAPI](https://fastapi.tiangolo.com/) and serves the trained LightGBM churn prediction model. It provides real-time single and batch predictions with full request tracing.

**Base URL:** `http://localhost:8000` (local) or `http://api:8000` (Docker Compose)

**Interactive API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI)

---

## Authentication

The API does not currently require authentication for development use. In production, add JWT or API key authentication via FastAPI middleware.

---

## Endpoints

### POST `/predict` — Single Prediction

Predict churn probability for a single customer.

**Request Body:**

| Field | Type | Required | Constraints | Example |
|-------|------|----------|-------------|---------|
| `gender` | string | ✅ | `Male` or `Female` | `"Male"` |
| `SeniorCitizen` | integer | ✅ | `0` or `1` | `0` |
| `Partner` | string | ✅ | `Yes` or `No` | `"Yes"` |
| `Dependents` | string | ✅ | `Yes` or `No` | `"No"` |
| `tenure` | integer | ✅ | ≥ 0 | `12` |
| `PhoneService` | string | ✅ | `Yes` or `No` | `"Yes"` |
| `MultipleLines` | string | ✅ | `Yes`, `No`, `No phone service` | `"No"` |
| `InternetService` | string | ✅ | `DSL`, `Fiber optic`, `No` | `"Fiber optic"` |
| `OnlineSecurity` | string | ✅ | `Yes`, `No`, `No internet service` | `"No"` |
| `OnlineBackup` | string | ✅ | `Yes`, `No`, `No internet service` | `"Yes"` |
| `DeviceProtection` | string | ✅ | `Yes`, `No`, `No internet service` | `"No"` |
| `TechSupport` | string | ✅ | `Yes`, `No`, `No internet service` | `"No"` |
| `StreamingTV` | string | ✅ | `Yes`, `No`, `No internet service` | `"No"` |
| `StreamingMovies` | string | ✅ | `Yes`, `No`, `No internet service` | `"No"` |
| `Contract` | string | ✅ | `Month-to-month`, `One year`, `Two year` | `"Month-to-month"` |
| `PaperlessBilling` | string | ✅ | `Yes` or `No` | `"Yes"` |
| `PaymentMethod` | string | ✅ | See allowed values below | `"Electronic check"` |
| `MonthlyCharges` | float | ✅ | > 0 | `70.35` |
| `TotalCharges` | float | ✅ | ≥ 0 | `844.20` |

**Allowed `PaymentMethod` values:**
- `Electronic check`
- `Mailed check`
- `Bank transfer (automatic)`
- `Credit card (automatic)`

**Example Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
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
         }'
```

**Response (200 OK):**
```json
{
    "churn_probability": 0.5843,
    "will_churn": true,
    "risk_tier": "HIGH_RISK",
    "threshold_used": 0.34,
    "request_id": "a1b2c3d4-e5f6-7890-abcd-1234567890ab"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `churn_probability` | float | Predicted probability of churn (0.0 to 1.0) |
| `will_churn` | boolean | Binary prediction based on the cost-optimal threshold |
| `risk_tier` | string | `HIGH_RISK`, `MEDIUM_RISK`, or `LOW_RISK` |
| `threshold_used` | float | Decision threshold used for binary classification |
| `request_id` | string | Unique UUID for request tracing in logs |

---

### POST `/predict/batch` — Batch Prediction

Predict churn for multiple customers in a single request.

**Request Body:**
```json
{
    "customers": [
        { ...customer_1_features... },
        { ...customer_2_features... }
    ]
}
```

**Constraints:**
- Maximum 100 customers per batch request
- Each customer object follows the same schema as `/predict`

**Response (200 OK):**
```json
{
    "predictions": [
        {
            "churn_probability": 0.5843,
            "will_churn": true,
            "risk_tier": "HIGH_RISK",
            "threshold_used": 0.34
        },
        {
            "churn_probability": 0.1205,
            "will_churn": false,
            "risk_tier": "LOW_RISK",
            "threshold_used": 0.34
        }
    ],
    "total_customers": 2,
    "high_risk_count": 1,
    "medium_risk_count": 0,
    "low_risk_count": 1,
    "request_id": "b2c3d4e5-f6a7-8901-bcde-2345678901bc"
}
```

---

### GET `/health` — Health Check

Returns the API's operational status. Used by load balancers, Docker health checks, and Kubernetes probes.

**Response (200 OK):**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "version": "0.1.0"
}
```

> **Note:** `model_loaded: false` means the model hasn't been loaded yet (lazy loading). The first `/predict` request triggers model loading. The API is still "healthy" — it just hasn't received its first prediction request yet.

---

### GET `/model/info` — Model Information

Returns metadata about the currently loaded model. Useful for debugging and audit trails.

**Response (200 OK):**
```json
{
    "model_name": "customer-churn-lgbm",
    "model_version": "local-dev",
    "mlflow_run_id": null,
    "optimal_threshold": 0.34,
    "algorithm": "lightgbm",
    "feature_count": 47,
    "source": "local_training"
}
```

---

## Error Handling

All errors return structured JSON responses:

### 422 — Validation Error

Returned when request data fails Pydantic validation.

```json
{
    "detail": "Validation error",
    "request_id": "c3d4e5f6-a7b8-9012-cdef-3456789012cd",
    "errors": [
        {
            "loc": ["body", "MonthlyCharges"],
            "msg": "Input should be greater than 0",
            "type": "greater_than"
        }
    ]
}
```

### 500 — Internal Server Error

Returned when the model fails to load or an unexpected error occurs.

```json
{
    "detail": "Model loading failed. Check the logs for details.",
    "request_id": "d4e5f6a7-b8c9-0123-defa-4567890123de"
}
```

---

## Request Tracing

Every response includes an `X-Request-ID` header containing the unique request UUID. This ID also appears in:
- The response body (`request_id` field)
- Server logs (searchable for debugging)
- Error responses

**Example:**
```
X-Request-ID: a1b2c3d4-e5f6-7890-abcd-1234567890ab
```

Use this ID when reporting issues or debugging production problems.

---

## Rate Limits

No rate limiting is currently implemented. For production deployment, consider adding rate limiting via:
- Nginx reverse proxy (`limit_req_zone`)
- FastAPI middleware (`slowapi`)
- Cloud load balancer settings

---

## Python Client Example

```python
import requests

API_URL = "http://localhost:8000"

# Single prediction
customer = {
    "gender": "Female",
    "SeniorCitizen": 1,
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
    "MonthlyCharges": 95.00,
    "TotalCharges": 95.00,
}

response = requests.post(f"{API_URL}/predict", json=customer)
result = response.json()

print(f"Churn Probability: {result['churn_probability']:.1%}")
print(f"Risk Tier: {result['risk_tier']}")
print(f"Will Churn: {result['will_churn']}")
```
