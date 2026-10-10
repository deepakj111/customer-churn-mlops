# Production Deployment Guide

Operational handbook for deploying the **Customer Churn MLOps & Causal AI Platform** across local, containerized, and cloud Kubernetes environments.

---

## 1. Deployment Architecture Overview

```
                                  INTERNET / CLIENT TRAFFIC
                                              │
                      ┌───────────────────────▼───────────────────────┐
                      │    Reverse Proxy / Ingress Controller (Nginx) │
                      │       SSL Termination │ JWT Auth │ Rate-Limit │
                      └───────────────┬───────────────────────────────┘
                                      │
               ┌──────────────────────┴──────────────────────┐
               │                                             │
      ┌────────▼────────┐                           ┌────────▼────────┐
      │  FastAPI Serving │                           │ Streamlit UI    │
      │   (Port 8000)   │                           │  (Port 8501)    │
      │  Workers: 4 CPU │                           │                 │
      └────────┬────────┘                           └─────────────────┘
               │
    ┌──────────┴───────────────────────────┬───────────────────────────┐
    │                                      │                           │
┌───▼──────────────────────┐   ┌───────────▼──────────────┐  ┌─────────▼───────────────┐
│  Feast Online Store      │   │  Prometheus (Port 9090)  │  │  MLflow Registry (5000) │
│  (Sub-10ms Redis/SQLite) │   │  Metrics Scraper         │  │  Model Artifacts & Runs │
└──────────────────────────┘   └───────────┬──────────────┘  └─────────────────────────┘
                                           │
                               ┌───────────▼──────────────┐
                               │  Grafana Dashboard (3000)│
                               │  Real-time Traffic & Lat │
                               └──────────────────────────┘
```

---

## 2. Quick Local Deployment (CLI & Makefile)

### Prerequisites
- Python 3.11+
- Poetry 2.x
- Git & Make

### Step-by-Step Local Deployment

```bash
# 1. Clone repository
git clone https://github.com/deepakj111/customer-churn-mlops.git
cd customer-churn-mlops

# 2. Install dependencies (creates virtualenv)
poetry install

# 3. Configure environment
cp .env.example .env

# 4. Pull dataset via DVC (or place WA_Fn-UseC_-Telco-Customer-Churn.csv in data/raw/)
poetry run dvc pull

# 5. Run the automated production training pipeline
# (Ingests data -> Validates with Pandera -> Trains LightGBM -> Calibrates -> Audits Fairness -> Fits Conformal)
poetry run python main.py train

# 6. Train the Causal Uplift (CATE) Prescriptive model
poetry run python main.py uplift

# 7. Materialize Feast Feature Store
make feast-apply
make feast-materialize

# 8. Launch the FastAPI serving server
poetry run python main.py serve --port 8000
# Accessible at: http://localhost:8000 (Docs: http://localhost:8000/docs)

# 9. Launch the Streamlit dashboard in a second terminal
make dashboard
# Accessible at: http://localhost:8501
```

---

## 3. Containerized Deployment (Docker & Docker Compose)

The repository provides production-hardened multi-stage Dockerfiles utilizing minimal non-root Debian-slim images.

### Services Stack in `docker/docker-compose.yml`

| Service | Internal Address | Host Port | Purpose | Health Check |
|---|---|---|---|---|
| **`api`** | `http://api:8000` | `8000` | FastAPI Serving (Predictive + Causal) | `GET /health` |
| **`dashboard`** | `http://dashboard:8501` | `8501` | Streamlit User Interface | `GET /_stcore/health` |
| **`mlflow`** | `http://mlflow:5000` | `5000` | MLflow Experiment Tracker & Registry | TCP port 5000 |
| **`prometheus`** | `http://prometheus:9090` | `9090` | Time-series Metrics Collector | `GET /-/healthy` |
| **`grafana`** | `http://grafana:3000` | `3000` | Operational Visualization Dashboards | `GET /api/health` |

### One-Command Deployment

```bash
# Build and launch all 5 services in detached mode
docker compose -f docker/docker-compose.yml up --build -d

# Verify container statuses
docker compose -f docker/docker-compose.yml ps

# View live service logs
docker compose -f docker/docker-compose.yml logs -f api

# Stop all services and network bridges
docker compose -f docker/docker-compose.yml down
```

---

## 4. Production Kubernetes Deployment Patterns

### Recommended Resource Sizing

| Container | CPU Request / Limit | Memory Request / Limit | Concurrency Model |
|---|---|---|---|
| **API Container** | `500m / 2000m` | `1Gi / 2Gi` | Uvicorn (4 async workers) |
| **Dashboard** | `250m / 1000m` | `512Mi / 1Gi` | Streamlit WebSockets |
| **Feast Redis Cache** | `250m / 500m` | `512Mi / 1Gi` | In-memory key-value |

### Kubernetes Pod Manifest (`api-deployment.yaml`)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn-prediction-api
  labels:
    app: churn-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: churn-api
  template:
    metadata:
      labels:
        app: churn-api
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
        - name: api
          image: churn-api:latest
          command: ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
          ports:
            - containerPort: 8000
          resources:
            requests:
              memory: "1Gi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "2000m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 15
            periodSeconds: 10
            timeoutSeconds: 3
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
            timeoutSeconds: 2
            failureThreshold: 2
          envFrom:
            - configMapRef:
                name: churn-api-config
            - secretRef:
                name: churn-api-secrets
```

---

## 5. Automated Production Retraining Pipelines (Prefect)

The system includes production orchestration flows in `pipelines/`:

### Automated Daily Monitoring & Retraining Flow
Execute via cron, Kubernetes CronJob, or Prefect Cloud:
```bash
# Evaluates PSI, Chi-sq, and Wasserstein drift:
# If drift exceeds thresholds, automatically trains challenger,
# performs Nadeau-Bengio corrected t-test, and conditionally promotes.
poetry run python main.py retrain
```

### Forcing Retraining
```bash
# Force challenger training regardless of drift triggers
poetry run python main.py retrain --force
```

---

## 6. Environment Variables Reference

| Variable | Default | Required | Purpose |
|---|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | No | MLflow server URL (local or remote DagsHub) |
| `MLFLOW_TRACKING_USERNAME` | - | No | MLflow / DagsHub username |
| `MLFLOW_TRACKING_PASSWORD` | - | No | MLflow / DagsHub password token |
| `API_HOST` | `0.0.0.0` | No | FastAPI bind address |
| `API_PORT` | `8000` | No | FastAPI bind port |
| `API_URL` | `http://localhost:8000` | No | Streamlit → API connection endpoint |
| `LOG_LEVEL` | `INFO` | No | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `DVC_REMOTE_URL` | - | No | Remote bucket storage (S3 / GCS) for raw data versioning |

---

## 7. Operational Runbook & Troubleshooting

### Problem: API returns `503 Service Unavailable` on startup
- **Cause**: Champion model artifact is missing from disk and MLflow is offline.
- **Remedy**:
  ```bash
  # Execute the production training pipeline to generate models/champion_pipeline.joblib
  poetry run python main.py train
  ```

### Problem: `/uplift/predict` returns `503 Service Unavailable`
- **Cause**: Causal uplift artifact has not yet been fitted.
- **Remedy**:
  ```bash
  # Execute causal uplift training to generate models/uplift/uplift_model.joblib
  poetry run python main.py uplift
  ```

### Problem: `/predict/customer/{customer_id}` returns `404 Not Found`
- **Cause**: Customer entity has not been materialized to Feast online store.
- **Remedy**:
  ```bash
  cd src/features/feast_repo
  poetry run feast apply
  poetry run feast materialize-incremental $(date -Iseconds)
  ```
