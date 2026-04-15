# 🔭 Customer Churn Predictor — End-to-End MLOps Pipeline

[![CI Pipeline](https://github.com/deepakj111/customer-churn-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/deepakj111/customer-churn-mlops/actions/workflows/ci.yml)
[![CD](https://github.com/deepakj111/customer-churn-mlops/actions/workflows/cd.yml/badge.svg)](https://github.com/deepakj111/customer-churn-mlops/actions/workflows/cd.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/Tests-281%20Passed-brightgreen.svg?logo=pytest)](https://docs.pytest.org/)
[![Coverage](https://img.shields.io/badge/Coverage-75%25-yellowgreen.svg)](htmlcov/index.html)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-3.10+-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-13ADC7.svg?logo=dvc&logoColor=white)](https://dvc.org/)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

---

A **production-grade Machine Learning Operations pipeline** that predicts customer churn using the [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) dataset. This project demonstrates the **complete ML lifecycle** — from exploratory analysis to containerized deployment — following industry best practices in MLOps, software engineering, and DevOps.

> **Why this project matters:** Reducing customer churn by even 5% can increase profits by 25–95% ([Harvard Business Review](https://hbr.org/2014/10/the-value-of-keeping-the-right-customers)). This pipeline doesn't just predict churn — it optimizes for **real dollar savings** using asymmetric cost-matrix threshold tuning, making predictions directly actionable for business teams.

---

## 📋 Table of Contents

- [Key Features](#-key-features)
- [Architecture Overview](#-architecture-overview)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [API Reference](#-api-reference)
- [Docker Deployment](#-docker-deployment)
- [Testing](#-testing)
- [Configuration](#-configuration)
- [MLOps Practices](#-mlops-practices)
- [License](#-license)

---

## ✨ Key Features

### 🧠 Machine Learning
- **28 engineered features** across 6 groups (demographic, billing, service depth, contract, interaction, composite)
- **5-stage consensus feature selection** (Variance → Correlation → Mutual Information → Random Forest → Permutation Importance)
- **LightGBM** with Optuna-tuned hyperparameters and class-imbalance handling via `scale_pos_weight`
- **Cost-matrix threshold optimization** — minimizes total business cost ($500/missed churner vs $20/false alarm)
- **3-way stratified data split** (Train 70% / Validation 10% / Test 20%) to prevent data leakage

### 🚀 Production Serving
- **FastAPI REST API** with single and batch prediction endpoints, Pydantic v2 validation, and OpenAPI docs
- **Thread-safe lazy model loading** with automatic fallback chain (MLflow Registry → Local Training)
- **Request tracing** via `X-Request-ID` UUID header on every response
- **Risk tier classification** (High / Medium / Low) with actionable business recommendations

### 📊 Monitoring & Observability
- **Real-time API Metrics**: Prometheus instrumentation tracking request volumes, error rates, latencies, and prediction distributions over time. Actively scraped from `/metrics`.
- **Grafana Dashboards**: Live, out-of-the-box provisioned dashboard visualizing incoming prediction requests.
- **Data drift detection** using PSI (numerical) and Chi-squared (categorical) statistical tests
- **Prediction drift tracking** via relative mean shift against training baseline
- **Config-driven thresholds** — all monitoring parameters tunable from YAML without code changes

### 🐳 DevOps & Infrastructure
- **Multi-stage Docker builds** — optimized for size (slim Python) and security (non-root user)
- **Docker Compose** — one-command local stack (API + Dashboard + MLflow + Prometheus + Grafana)
- **Continuous Integration (CI)** — automated linting, testing, and Docker image validation on every PR
- **Continuous Deployment (CD)** — automated Docker build and push to GitHub Container Registry (GHCR) upon merges to `main`
- **DVC** for dataset versioning, **Poetry** for dependency management

### 🧪 Quality Assurance
- **281 tests** (unit + integration) with **75% code coverage**
- **Pandera schema validation** at ingestion and inference boundaries
- **Pre-commit hooks** (Black, isort, Flake8) enforce consistent code style
- **Type-safe configuration** — YAML configs loaded into Python dataclasses

---

## 🏗️ Architecture Overview

```
                         ┌─────────────────────────────────────────┐
                         │            GitHub Actions CI/CD         │
                         │   Lint → Test (285) → Docker Build &    │
                         │   Push Image to Container Registry      │
                         └───────────────┬─────────────────────────┘
                                         │
    ┌────────────┐    ┌──────────┐    ┌──┴───────┐    ┌──────────────┐
    │  Raw Data  │───▶│ Validate │───▶│ Feature  │───▶│   Train +    │
    │  (DVC)     │    │ (Pandera)│    │  Store   │    │  Optimize    │
    └────────────┘    └──────────┘    │(28 feats)│    │  (MLflow)    │
                                      └──────────┘    └──────┬───────┘
                                                             │
                                                    ┌────────┴────────┐
                                                    │  sklearn.Pipeline│
                                                    │ (FeatEng→Prep→  │
                                                    │  Model)         │
                                                    └────────┬────────┘
                                                             │
         ┌─────────────┐    ┌──────────────┐    ┌────────────┴────────┐
         │  Streamlit   │◀──│  FastAPI      │◀──│  Model Loader       │
         │  Dashboard   │   │  /predict     │   │  (MLflow→Local      │
         │  (port 8501) │   │  (port 8000)  │   │   fallback)         │
         └─────────────┘    └──────┬───────┘    └─────────────────────┘
                                   │
                    ┌──────────────┴───────────────┐
                    │          Prometheus          │
                    │        (Scrapes /metrics)    │
                    └──────────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │           Grafana            │
                    │    (Live Traffic Dashboards) │
                    └──────────────────────────────┘
```

**Key Design Decisions:**
1. **Training-Serving Parity** — The `sklearn.Pipeline` encapsulates feature engineering, preprocessing, and model inference into a single serializable object. Training-serving skew is structurally impossible.
2. **Stateless Feature Engineering** — All 28 features are computed via pure functions, guaranteeing identical behavior in training and inference.
3. **Cost-Sensitive, Not Accuracy-Driven** — The decision threshold is optimized using a business cost matrix, not the default 0.5 cutoff.

---

## 📂 Project Structure

```
customer-churn-mlops/
│
├── .github/workflows/         # CI/CD pipeline (lint, test, Docker build)
│   └── ci.yml
│
├── configs/                   # All configuration (YAML → Python dataclasses)
│   ├── feature_config.yaml    #   Feature names, target column, engineered features
│   ├── model_config.yaml      #   Algorithm, hyperparameters, cost matrix, performance gates
│   ├── monitoring_config.yaml #   Drift thresholds (PSI, chi-squared, prediction drift)
│   └── training_config.yaml   #   Split ratios, CV folds, experiment name
│
├── dashboards/                # Streamlit monitoring & prediction dashboard
│   └── streamlit_app.py
│
├── data/
│   ├── raw/                   # DVC-tracked raw dataset (Telco CSV)
│   ├── processed/             # Intermediate artifacts (tuning results, etc.)
│   └── reference/             # Drift monitoring reference snapshots (Parquet)
│
├── docker/                    # Containerization
│   ├── Dockerfile             #   Multi-stage build for FastAPI API
│   ├── Dockerfile.streamlit   #   Dashboard container
│   ├── docker-compose.yml     #   Full local stack (API + Dashboard + MLflow)
│   └── .dockerignore
│
├── notebooks/                 # Exploratory analysis & experiments
│   ├── 01_eda_and_business_analysis.py
│   ├── 02_feature_engineering_experiments.py
│   ├── 03a_algorithm_scan.py
│   ├── 03b_hyperparameter_tuning.py
│   └── 03c_champion_evaluation.py
│
├── reports/                   # Auto-generated plots (EDA, feature importance, etc.)
│
├── scripts/                   # CLI utilities
│   ├── save_reference_data.py #   Save training data for drift monitoring
│   └── generate_drift_report.py #  Generate drift analysis report
│
├── src/                       # Production source code
│   ├── data/                  #   Ingestion, validation (Pandera), preprocessing
│   │   ├── ingest.py
│   │   ├── validate.py
│   │   └── preprocess.py
│   ├── features/              #   Feature engineering & multi-stage selection
│   │   ├── feature_store.py
│   │   └── feature_selector.py
│   ├── models/                #   Training, evaluation, pipeline factory, threshold tuning
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── pipeline.py
│   │   └── threshold.py
│   ├── monitoring/            #   Data drift detection & reference management
│   │   ├── drift_detector.py
│   │   └── reference_builder.py
│   ├── serving/               #   FastAPI prediction API
│   │   ├── api.py
│   │   ├── schemas.py
│   │   └── model_loader.py
│   └── utils/                 #   Configuration loader & structured logging
│       ├── config_loader.py
│       └── logging.py
│
├── tests/                     # 281 tests (unit + integration + data validation)
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── data_tests/
│
├── main.py                    # Training pipeline entry point
├── pyproject.toml             # Poetry deps, pytest, Black, isort, Flake8, mypy config
├── Makefile                   # Developer workflow automation
├── .pre-commit-config.yaml    # Git hooks for code quality
└── .env.example               # Environment variable template
```

---

## 🚀 Quick Start

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11+ | Runtime |
| Poetry | 2.x | Dependency management |
| Docker (optional) | 24+ | Container deployment |
| Make (optional) | Any | Workflow automation |

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/deepakjangra/customer-churn-mlops.git
cd customer-churn-mlops

# 2. Install all dependencies (production + dev)
poetry install

# 3. Set up environment variables
cp .env.example .env
# Edit .env with your MLflow/DagsHub credentials (optional)

# 4. Pull the DVC-tracked dataset + install pre-commit hooks
make setup

# 5. Verify everything works
make test
```

### 30-Second Demo

```bash
# Train the model
make train

# Start the API
make serve

# In another terminal — predict churn
curl -s http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Male","SeniorCitizen":0,"Partner":"Yes","Dependents":"No",
       "tenure":2,"PhoneService":"Yes","MultipleLines":"No",
       "InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No",
       "DeviceProtection":"No","TechSupport":"No","StreamingTV":"No",
       "StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes",
       "PaymentMethod":"Electronic check","MonthlyCharges":70.35,"TotalCharges":140.70}' | python -m json.tool
```

**Expected Output:**
```json
{
    "churn_probability": 0.8214,
    "will_churn": true,
    "risk_tier": "HIGH_RISK",
    "threshold_used": 0.34,
    "request_id": "a1b2c3d4-e5f6-7890-abcd-1234567890ab"
}
```

---

## 💻 Usage Guide

All commands are available via the `Makefile`. Run `make help` to see all options.

| Command | Description |
|---------|-------------|
| `make setup` | First-time setup (install deps, pre-commit hooks, DVC pull) |
| `make train` | Run the full training pipeline with MLflow logging |
| `make serve` | Start FastAPI prediction server (port 8000) |
| `make dashboard` | Launch Streamlit monitoring dashboard (port 8501) |
| `make test` | Run all 281 tests with coverage report |
| `make test-unit` | Run unit tests only |
| `make test-integration` | Run integration tests only |
| `make format` | Auto-format code (Black + isort) |
| `make lint` | Run linters (Flake8 + mypy) |
| `make drift-report` | Generate a data drift monitoring report |
| `make docker-up` | Start full Docker Compose stack |
| `make docker-down` | Stop all Docker containers |
| `make clean` | Remove Python cache files |

### Training Pipeline

The training pipeline executes the following sequence:

```
Raw CSV → Validate (Pandera) → Preprocess → Engineer Features (28)
       → 3-Way Split → Build sklearn.Pipeline → Train LightGBM
       → Optimize Threshold (cost matrix) → Evaluate → Log to MLflow
```

```bash
# Run training with MLflow tracking
make train

# Results are logged to:
# - Terminal: metrics summary
# - MLflow: full experiment (metrics, params, model artifact)
# - reports/: evaluation plots (confusion matrix, feature importance, etc.)
```

### Prediction API

```bash
# Start with hot-reload (development)
make serve

# Access interactive docs
open http://localhost:8000/docs
```

### Streamlit Dashboard

```bash
make dashboard
# Opens at http://localhost:8501
```

The dashboard provides 4 tabs:
1. **🎯 Predict** — Interactive single-customer churn prediction form
2. **📋 Batch Analysis** — Upload CSV for bulk predictions (up to 100 customers)
3. **📈 Model Performance** — Training metrics, confusion matrices, feature importance
4. **💰 Business Impact** — ROI calculator showing dollar savings vs. no-model baseline

---

## 📡 API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Single customer churn prediction |
| `POST` | `/predict/batch` | Batch prediction (1–100 customers) |
| `GET` | `/health` | Health check (for load balancers / k8s) |
| `GET` | `/model/info` | Model metadata (version, features, threshold) |
| `GET` | `/docs` | Interactive OpenAPI documentation |

### POST `/predict` — Request Body

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

### POST `/predict` — Response

```json
{
    "churn_probability": 0.5843,
    "will_churn": true,
    "risk_tier": "HIGH_RISK",
    "threshold_used": 0.34,
    "request_id": "a1b2c3d4-e5f6-7890-abcd-1234567890ab"
}
```

### Risk Tier Definitions

| Tier | Threshold | Recommended Action |
|------|-----------|-------------------|
| 🔴 `HIGH_RISK` | probability ≥ 0.60 | Immediate personal outreach, retention offer |
| 🟡 `MEDIUM_RISK` | 0.35 ≤ probability < 0.60 | Automated retention campaign, usage monitoring |
| 🟢 `LOW_RISK` | probability < 0.35 | Standard engagement, cross-sell opportunities |

---

## 🐳 Docker Deployment

### Quick Start with Docker Compose

```bash
# Build and start all services
docker compose -f docker/docker-compose.yml up --build -d

# Services:
#   API:        http://localhost:8000  (FastAPI + Swagger docs + /metrics)
#   Dashboard:  http://localhost:8501  (Streamlit)
#   MLflow:     http://localhost:5000  (Experiment tracker)
#   Grafana:    http://localhost:3000  (Real-time dashboards)

# Stop all services
docker compose -f docker/docker-compose.yml down
```

### Individual Images

```bash
# Build API image only
docker build -f docker/Dockerfile -t churn-api:latest .

# Run API container
docker run -p 8000:8000 --env-file .env churn-api:latest
```

### Docker Architecture

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `api` | `churn-api` | 8000 | FastAPI prediction service |
| `dashboard` | `churn-dashboard` | 8501 | Streamlit monitoring UI |
| `mlflow` | `ghcr.io/mlflow/mlflow` | 5000 | Experiment tracking server |
| `prometheus` | `prom/prometheus` | 9090 | Metrics scraper for the API |
| `grafana` | `grafana/grafana`| 3000 | Real-time observability dashboards |

**Design choices:**
- Multi-stage builds minimize image size (builder exports deps → runtime uses slim Python)
- Non-root `appuser` for security best practices
- Built-in health checks for container orchestrators (Docker Swarm, k8s, ECS)
- Internal Docker DNS for service-to-service communication (dashboard → API at `http://api:8000`)

---

## 🧪 Testing

### Test Suite Overview

```
tests/
├── conftest.py                    # Shared fixtures, config singleton reset
├── unit/                          # Component-level tests
│   ├── test_ingest.py             # Data loading
│   ├── test_preprocess.py         # Preprocessing logic
│   ├── test_validate.py           # Pandera schema validation
│   ├── test_feature_store.py      # Feature engineering (28 features)
│   ├── test_pipeline.py           # sklearn Pipeline factory
│   ├── test_evaluate.py           # ML + business metrics
│   ├── test_threshold.py          # Cost-optimal threshold, risk tiers
│   ├── test_schemas.py            # Pydantic request/response schemas
│   ├── test_config_loader.py      # Configuration loading + validation
│   ├── test_logging.py            # Structured logging format
│   ├── test_drift_detector.py     # PSI, chi-squared, prediction drift
│   └── test_reference_builder.py  # Reference data Parquet round-trip
├── integration/
│   └── test_api.py                # FastAPI endpoint integration tests
└── data_tests/
    └── test_data_validation.py    # Raw data quality checks
```

### Running Tests

```bash
# Full suite with coverage
make test
# → 281 passed, 75% coverage

# Unit tests only
make test-unit

# Integration tests only
make test-integration

# Specific test file
poetry run pytest tests/unit/test_drift_detector.py -v

# With detailed coverage report
poetry run pytest --cov=src --cov-report=html
open htmlcov/index.html
```

### Test Philosophy

| Principle | Implementation |
|-----------|---------------|
| **Isolation** | Config singleton reset between tests (`conftest.py`) |
| **No side effects** | Tests use fixtures and `tmp_path`, never touch real data |
| **Fast feedback** | Full suite completes in ~12 seconds |
| **Realistic fixtures** | Test data mirrors actual Telco dataset schema and distributions |

---

## ⚙️ Configuration

All configuration is managed through YAML files in `configs/` and loaded into typed Python dataclasses via `src/utils/config_loader.py`.

### Config Files

| File | Purpose | Key Settings |
|------|---------|-------------|
| `model_config.yaml` | Model algorithm, hyperparameters, cost matrix | `false_negative_cost: 500`, `false_positive_cost: 20` |
| `feature_config.yaml` | Feature names, target column, engineered feature list | 28 engineered features across 6 groups |
| `training_config.yaml` | Data splits, cross-validation, experiment name | `test_size: 0.2`, `val_size: 0.1`, `cv_folds: 5` |
| `monitoring_config.yaml` | Drift detection thresholds | `psi_threshold: 0.25`, `chi_squared_alpha: 0.05` |

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# MLflow / DagsHub (optional — local tracking used if not set)
MLFLOW_TRACKING_URI=https://dagshub.com/username/repo.mlflow
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_token

# DVC remote storage
DVC_REMOTE_URL=https://dagshub.com/username/repo.dvc

# API configuration
API_HOST=0.0.0.0
API_PORT=8000
```

---

## 🔄 MLOps Practices

This project demonstrates the following MLOps maturity practices:

### Data Management
| Practice | Implementation |
|----------|---------------|
| Data versioning | DVC tracks raw datasets independently from Git |
| Schema validation | Pandera enforces column types, ranges, and allowed values |
| Training-serving parity | `sklearn.Pipeline` encapsulates the full transformation chain |
| Reference snapshots | Parquet-based training data snapshots for drift comparison |

### Model Lifecycle
| Practice | Implementation |
|----------|---------------|
| Experiment tracking | MLflow logs all params, metrics, and model artifacts |
| Model registry | MLflow Model Registry with staging/production stages |
| Performance gates | Minimum ROC-AUC (0.82), PR-AUC (0.65), Recall (0.70) required |
| Reproducibility | Fixed random seeds, locked dependencies (poetry.lock), DVC |

### Deployment & Monitoring
| Practice | Implementation |
|----------|---------------|
| Containerization | Multi-stage Docker builds with persistent data volumes |
| CI/CD | GitHub Actions: Push to GHCR on main branch with parallel test/lint gates |
| Data drift monitoring | PSI (numerical), Chi-squared (categorical), prediction drift |
| Observability | Prometheus `/metrics` scraping, Grafana live Dashboards, Request ID tracing |

### Code Quality
| Practice | Implementation |
|----------|---------------|
| Testing | 281 tests, 75% coverage, unit + integration + data validation |
| Linting | Black (formatting), isort (imports), Flake8 (style), mypy (types) |
| Pre-commit hooks | Automated checks before every commit |
| Modular architecture | Clean separation: data → features → models → serving → monitoring |



## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn) — IBM/Kaggle
- [LightGBM](https://lightgbm.readthedocs.io/) — Microsoft
- [MLflow](https://mlflow.org/) — Databricks
- [FastAPI](https://fastapi.tiangolo.com/) — Sebastián Ramírez
- [Scikit-learn](https://scikit-learn.org/) — Community
- [Pandera](https://pandera.readthedocs.io/) — Niels Bantilan

---

<p align="center">
<em>Building data science that aligns tightly with engineering rigor and business impact.</em>
</p>
