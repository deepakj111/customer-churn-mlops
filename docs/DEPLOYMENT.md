# Deployment Guide

Instructions for deploying the Customer Churn Prediction system in various environments.

---

## Table of Contents

- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Docker Compose (Full Stack)](#docker-compose-full-stack)
- [Production Checklist](#production-checklist)
- [Environment Variables Reference](#environment-variables-reference)

---

## Local Development

### Prerequisites

- Python 3.11+
- Poetry 2.x

### Setup

```bash
# Install dependencies
poetry install

# Set up environment
cp .env.example .env
# Edit .env with your credentials

# Pull data
poetry run dvc pull

# Start the API server (with hot-reload)
make serve
# → http://localhost:8000/docs

# Start the dashboard (in another terminal)
make dashboard
# → http://localhost:8501
```

### Running the API in Debug Mode

```bash
# With verbose logging
LOG_LEVEL=DEBUG poetry run uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
```

---

## Docker Deployment

### Building Images

```bash
# Build the API image
docker build -f docker/Dockerfile -t churn-api:latest .

# Build the dashboard image
docker build -f docker/Dockerfile.streamlit -t churn-dashboard:latest .
```

### Running Individual Containers

```bash
# API only
docker run -d \
    --name churn-api \
    -p 8000:8000 \
    --env-file .env \
    churn-api:latest

# Dashboard only (needs API running)
docker run -d \
    --name churn-dashboard \
    -p 8501:8501 \
    -e API_URL=http://host.docker.internal:8000 \
    churn-dashboard:latest
```

### Image Details

| Image | Base | Size (approx) | Non-root | Health Check |
|-------|------|---------------|----------|-------------|
| `churn-api` | `python:3.11-slim` | ~500MB | ✅ `appuser` | ✅ `/health` |
| `churn-dashboard` | `python:3.11-slim` | ~500MB | ✅ `appuser` | ✅ `/_stcore/health` |

---

## Docker Compose (Full Stack)

The full stack includes three services: API, Dashboard, and MLflow.

### Starting the Stack

```bash
# Build and start (foreground — see logs)
docker compose -f docker/docker-compose.yml up --build

# Start in background
docker compose -f docker/docker-compose.yml up --build -d

# Check status
docker compose -f docker/docker-compose.yml ps

# View logs
docker compose -f docker/docker-compose.yml logs -f api
docker compose -f docker/docker-compose.yml logs -f dashboard
docker compose -f docker/docker-compose.yml logs -f mlflow

# Stop everything
docker compose -f docker/docker-compose.yml down

# Stop and remove volumes (fresh start)
docker compose -f docker/docker-compose.yml down -v
```

### Service Map

| Service | Internal URL | External URL | Description |
|---------|-------------|-------------|-------------|
| `api` | `http://api:8000` | `http://localhost:8000` | FastAPI prediction API |
| `dashboard` | `http://dashboard:8501` | `http://localhost:8501` | Streamlit dashboard |
| `mlflow` | `http://mlflow:5000` | `http://localhost:5000` | MLflow tracking server |

### Internal Networking

Services communicate via Docker's internal DNS:
- The **dashboard** calls the API at `http://api:8000` (not `localhost`)
- The **API** connects to MLflow at `http://mlflow:5000`
- External clients access services via `localhost` port mappings

### Persistent Storage

| Volume | Mapped To | Purpose |
|--------|----------|---------|
| `mlflow-data` | `/mlflow` | MLflow experiments, artifacts, and SQLite database |
| `../data/raw` (bind mount) | `/app/data/raw:ro` | Read-only access to training data for local model fallback |
| `../reports` (bind mount) | `/app/reports:ro` | Report images displayed in the dashboard |

---

## Production Checklist

Before deploying to production, review and address these items:

### Security

- [ ] **Remove `.env` from git history** — If `.env` was ever committed with real credentials, rotate all tokens
  ```bash
  # Remove from git tracking (file stays on disk)
  git rm --cached .env
  # Add to .gitignore (already present)
  # Rotate all tokens in DagsHub/Prefect/etc.
  ```
- [ ] **Restrict CORS origins** — Update `api.py` to only allow specific origins:
  ```python
  # Replace allow_origins=["*"] with:
  allow_origins=["https://your-dashboard-domain.com"]
  ```
- [ ] **Add API authentication** — Implement JWT or API key middleware
- [ ] **Use Docker secrets** — Don't pass sensitive env vars via `docker run -e`; use Docker secrets or a vault

### Performance

- [ ] **Use multiple Uvicorn workers** — Scale with `--workers 4` (1 worker per CPU core)
- [ ] **Add a reverse proxy** — Put Nginx or Traefik in front for SSL termination and rate limiting
- [ ] **Enable response caching** — Cache predictions for identical inputs (Redis or in-memory LRU)
- [ ] **Set resource limits** — Add `mem_limit` and `cpus` to docker-compose services

### Reliability

- [ ] **Set up health check monitoring** — Alert when `/health` returns non-200
- [ ] **Configure container restart policies** — `restart: unless-stopped` (already set in docker-compose)
- [ ] **Add logging aggregation** — Ship container logs to CloudWatch, Datadog, or ELK
- [ ] **Schedule drift monitoring** — Run `generate_drift_report.py` daily via cron or orchestrator

### Data

- [ ] **Automate reference data updates** — Run `save_reference_data.py` after each model retraining
- [ ] **Set up DVC remote storage** — Configure S3/GCS bucket for production data versioning
- [ ] **Implement data retention policies** — Archive old prediction logs and drift reports

---

## Environment Variables Reference

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `MLFLOW_TRACKING_URI` | - | No | MLflow server URL. If not set, uses local tracking |
| `MLFLOW_TRACKING_USERNAME` | - | No | DagsHub/MLflow username |
| `MLFLOW_TRACKING_PASSWORD` | - | No | DagsHub/MLflow token |
| `DVC_REMOTE_URL` | - | No | DVC remote storage URL |
| `MODEL_NAME` | `churn-predictor` | No | Registered model name in MLflow |
| `CHAMPION_STAGE` | `Production` | No | MLflow model stage to load |
| `API_HOST` | `0.0.0.0` | No | FastAPI bind address |
| `API_PORT` | `8000` | No | FastAPI bind port |
| `API_URL` | `http://localhost:8000` | No | Dashboard → API connection URL |
| `LOG_LEVEL` | `INFO` | No | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `PREFECT_API_URL` | - | No | Prefect Cloud workspace URL |
| `PREFECT_API_KEY` | - | No | Prefect Cloud API key |

---

## Troubleshooting

### API returns "Model loading failed"

The API couldn't load the model from MLflow or train one locally.

```bash
# Check if the raw data file exists
ls data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv

# If missing, pull via DVC
poetry run dvc pull
```

### Docker build fails at Poetry export

```bash
# Ensure poetry.lock is up to date
poetry lock
```

### Dashboard shows "API unreachable"

```bash
# Check if the API container is running
docker compose -f docker/docker-compose.yml ps

# Check API logs for errors
docker compose -f docker/docker-compose.yml logs api

# Verify the API_URL environment variable in the dashboard container
docker exec churn-dashboard printenv API_URL
# Should show http://api:8000 (not localhost)
```

### Tests fail with "DVC file not found"

```bash
# Pull the DVC-tracked data
poetry run dvc pull

# Or, if you don't have DVC remote configured,
# download the dataset manually from Kaggle and place it at:
# data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```
