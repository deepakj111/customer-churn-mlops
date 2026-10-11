# 🔭 Customer Churn Predictor — Production MLOps & Causal AI Platform

[![CI Pipeline](https://github.com/deepakj111/customer-churn-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/deepakj111/customer-churn-mlops/actions/workflows/ci.yml)
[![CD](https://github.com/deepakj111/customer-churn-mlops/actions/workflows/cd.yml/badge.svg)](https://github.com/deepakj111/customer-churn-mlops/actions/workflows/cd.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/Tests-303%20Passed-brightgreen.svg?logo=pytest)](https://docs.pytest.org/)
[![Type Checked: Mypy](https://img.shields.io/badge/Type%20Check-Mypy%20Strict-blue.svg)](https://mypy-lang.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-3.10+-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Kustomize%20Ready-326CE5.svg?logo=kubernetes&logoColor=white)](deploy/kubernetes/)
[![Release](https://img.shields.io/badge/Release-v1.0.0-blue.svg)](CHANGELOG.md)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-13ADC7.svg?logo=dvc&logoColor=white)](https://dvc.org/)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

---

A **Machine Learning Operations (MLOps) & Causal AI system** that predicts subscriber churn and prescribes cost-optimal retention interventions on the Telco Customer Churn dataset.

> **System Highlights:** Rather than evaluating models solely on symmetric accuracy with default 0.50 thresholds, this system focuses on **economic utility**:
> - **Asymmetric Cost Matrix Optimization ($p^* = 0.34$)** yielding **+$172,400 net savings** per 10k customer cohort.
> - **Causal Uplift Modeling (EconML T-Learner)** targeting *Persuadables* (28.2%) while strictly suppressing outreach to *Sleeping Dogs* (6.0%).
> - **Distribution-Free Uncertainty Quantification** via MAPIE Conformal Prediction at 90% confidence.
> - **Automated Retraining Pipelines** outside notebooks with **Nadeau-Bengio corrected hypothesis tests** to eliminate false-positive model promotions.

---

## 📑 Core Reports & Summaries

Key reports and architectural summaries:

| Document | Focus & Highlights | Direct Link |
|---|---|---|
| **Production Model Card** | Mitchell et al. format: feature schemas, isotonic calibration, MAPIE conformal uncertainty, EEOC 4/5ths fairness audit. | [**`results/MODEL_CARD.md`**](results/MODEL_CARD.md) |
| **Business Case Study** | Unit economics (CAC vs LTV), cost matrix optimization ($C_{FN}=\$500$ vs $C_{FP}=\$50$), prescriptive CATE uplift ROI. | [**`results/BUSINESS_CASE_STUDY.md`**](results/BUSINESS_CASE_STUDY.md) |
| **Statistical Benchmark Report** | 10 algorithms (LightGBM vs TabPFN vs FT-Transformer) with Nadeau-Bengio corrected CV t-tests and latency profiles. | [**`results/BENCHMARK_REPORT.md`**](results/BENCHMARK_REPORT.md) |
| **Drift Monitoring Report** | Population Stability Index (PSI), Chi-squared categorical shift tests, and prediction drift monitoring. | [**`results/DRIFT_MONITORING_REPORT.md`**](results/DRIFT_MONITORING_REPORT.md) |
| **Data Governance Report** | Pandera validation contracts, 28-feature lineage, Feast feature store topology, and privacy compliance. | [**`results/DATA_GOVERNANCE_REPORT.md`**](results/DATA_GOVERNANCE_REPORT.md) |
| **Research Notebooks Index** | 5 fully rendered `.ipynb` notebooks with rich visualizations, tables, and narrative business analysis. | [**`notebooks/README.md`**](notebooks/README.md) |

---

## 📚 Detailed System Documentation

Comprehensive architecture, operational manuals, and governance handbooks:

- 📡 [**API Reference Manual (`docs/API.md`)**](docs/API.md) — Complete REST endpoints (v1 routes, async batch scoring, readiness probes), cURL examples, Pydantic schemas, and Causal Uplift specifications.
- 🏗️ [**System Architecture Guide (`docs/ARCHITECTURE.md`)**](docs/ARCHITECTURE.md) — Layered design, dependency graphs, Prefect pipeline workflows, and deep engineering design decisions.
- 🚀 [**Production Deployment Guide (`docs/DEPLOYMENT.md`)**](docs/DEPLOYMENT.md) — Docker Compose stack, Kubernetes Kustomize manifests, resource governance, and operational runbooks.
- ☸️ [**Kubernetes Deployment Bundle (`deploy/kubernetes/README.md`)**](deploy/kubernetes/README.md) — Production Kustomize manifests, HPA autoscaling (2–10 pods), and zero-cost local validation (`make k8s-validate`).
- 📊 [**Continuous Monitoring Guide (`docs/MONITORING.md`)**](docs/MONITORING.md) — Population Stability Index (PSI), Chi-squared drift, Wasserstein distance, Prometheus metrics, and automated retraining flows.
- 🛡️ [**Security Policy (`SECURITY.md`)**](SECURITY.md) — Vulnerability disclosure process, OWASP API Top 10 headers, sliding-window rate limiting, and 3-tier input validation.
- 🤝 [**Contributing Guidelines (`CONTRIBUTING.md`)**](CONTRIBUTING.md) — Developer setup, branching strategy, conventional commit conventions, and pull request checklist.
- 📜 [**Changelog (`CHANGELOG.md`)**](CHANGELOG.md) — Semantic Versioning release notes across v0.9.0 and v1.0.0.

---

## 📋 Table of Contents

- [End-to-End Execution Guide: What to Run to Get What](#-end-to-end-execution-guide-what-to-run-to-get-what)
- [Key Features](#-key-features)
- [Architecture Overview](#-architecture-overview)
- [Project Structure](#-project-structure)
- [Operational Usage & CLI](#-operational-usage--cli)
- [API Reference](#-api-reference)
- [Docker Deployment](#-docker-deployment)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Configuration Architecture](#-configuration-architecture)
- [MLOps Practices](#-mlops-practices)
- [License](#-license)

---

## ⚡ End-to-End Execution Guide: What to Run to Get What

If someone clones this repository from scratch, here is the exact step-by-step sequence to run every component and what artifacts and outputs to expect:

| Step | Command | What It Does | Key Output / Artifact |
|:---:|---|---|---|
| 1 | `poetry install` | Installs all dependencies in virtualenv | `.venv` environment |
| 2 | `poetry run dvc pull` | Pulls dataset from remote storage | `data/raw/*.csv` |
| 3 | `make feast-apply` | Applies Feast Feature Store schema | Feast feature registry |
| 4 | `make feast-materialize` | Materializes SQLite online store | `src/features/feast_repo/data/online_store.db` |
| 5 | `poetry run python main.py train` | Runs automated training pipeline | `models/champion_pipeline.joblib`, `results/metrics/` |
| 6 | `poetry run python main.py uplift` | Trains EconML Causal Uplift T-Learner | `models/uplift/*`, `results/metrics/uplift_results.json` |
| 7 | `poetry run python main.py benchmark` | Runs 10-model CV + Nadeau-Bengio t-test | `results/metrics/benchmark_results.json` |
| 8 | `poetry run python main.py monitor` | Calculates decile PSI & Chi-sq drift | `results/metrics/drift_results.json`, `results/DRIFT_MONITORING_REPORT.md` |
| 9 | `poetry run python main.py retrain` | Evaluates drift & runs challenger flow | Promoted model in MLflow / disk |
| 10 | `poetry run python main.py serve` | Starts FastAPI server on port 8000 | Live REST API (`http://localhost:8000`) |
| 11 | `make dashboard` | Launches Streamlit dashboard (8501) | Interactive UI (`http://localhost:8501`) |
| 12 | `make docker-up` | Runs 5-container Compose stack | Complete containerized platform |
| 13 | `make test` | Runs all 303 pytest tests | 100% test pass rate |
| 14 | `make lint` | Checks flake8 and mypy strict typing | 0 lint / type errors |
| 15 | `make k8s-validate` | Validates K8s manifests offline | Zero-cost Kustomize validation |

### Stage 1: Installation & Data Pull
```bash
# 1. Install all dependencies via Poetry
poetry install

# 2. Configure environment variables
cp .env.example .env

# 3. Pull dataset tracked by DVC (or ensure WA_Fn-UseC_-Telco-Customer-Churn.csv is in data/raw/)
poetry run dvc pull
```

### Stage 2: Feature Store Initialization (Feast)
```bash
# Initialize and apply Feast feature store definition
make feast-apply

# Materialize batch features into online store for low-latency scoring
make feast-materialize
```
*Output:* Creates `data/online_store.db` allowing sub-10ms customer lookup via `GET /predict/customer/{id}`.

### Stage 3: Research Notebooks Exploration
```bash
# Research notebooks are pre-executed with cached charts in notebooks/*.ipynb
# To re-execute all notebooks and update charts:
make notebooks-exec
```
*Output:* Re-generates fully executed `.ipynb` notebooks in `notebooks/`.

### Stage 4: Production Training Pipeline (Prefect Flow)
```bash
# Executes the production training pipeline outside notebooks:
# Ingest -> Validate (Pandera) -> Train LightGBM -> Isotonic Calibration -> Fairness Audit -> Conformal Fit
poetry run python main.py train
# or: make pipeline-train
```
*Output Artifacts:*
- `models/champion_pipeline.joblib`: Serialized calibrated Scikit-Learn pipeline.
- `models/conformal/`: Fitted MAPIE conformal prediction artifacts.
- MLflow experiment run with logged parameters, metrics, and registered model.

### Stage 5: Causal Uplift (CATE) Prescriptive Modeling
```bash
# Trains an EconML T-Learner to isolate treatment effect of retention support
poetry run python main.py uplift
# or: make uplift-train
```
*Output Artifacts:*
- `models/uplift/uplift_model.joblib`: Fitted CATE meta-learner.
- `models/uplift/uplift_meta.json`: Feature alignments and schema metadata.
- `results/metrics/uplift_results.json`: Customer quadrant breakdown (Persuadables, Sleeping Dogs).

### Stage 6: Statistical Model Benchmarking Suite
```bash
# Runs 5-fold CV across 10 tabular models (LightGBM, CatBoost, XGBoost, TabPFN, FT-Transformer)
# and evaluates pairwise differences using the Nadeau-Bengio corrected t-test
poetry run python main.py benchmark
# or: make benchmark
```
*Output Artifacts:*
- `results/metrics/benchmark_results.json`: Complete fold scores, p-values, Cohen's $d$ effect sizes, and latency metrics.

### Stage 7: Continuous Monitoring & Drift Detection
```bash
# Save training reference baseline snapshot (if not already saved)
make save-reference

# Generate data and prediction drift report
poetry run python main.py monitor
# or: make drift-report
```
*Output Artifacts:*
- `results/metrics/drift_results.json`: Population Stability Index (PSI), Chi-sq p-values, and Wasserstein distance.
- `results/DRIFT_MONITORING_REPORT.md`: Comprehensive executive drift documentation.

### Stage 8: Automated Retraining & Champion-Challenger Promotion
```bash
# Evaluates drift triggers; if drift exceeds thresholds, trains challenger,
# executes Nadeau-Bengio corrected CV t-test, and conditionally promotes
poetry run python main.py retrain

# To force retraining regardless of drift triggers:
poetry run python main.py retrain --force
```

### Stage 9: Launch Serving API & Query Endpoints
```bash
# Start FastAPI server on port 8000
poetry run python main.py serve --port 8000
# or: make serve
```

```bash
# In a second terminal — query single customer churn prediction:
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Male","SeniorCitizen":0,"Partner":"Yes","Dependents":"No",
       "tenure":2,"PhoneService":"Yes","MultipleLines":"No",
       "InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No",
       "DeviceProtection":"No","TechSupport":"No","StreamingTV":"No",
       "StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes",
       "PaymentMethod":"Electronic check","MonthlyCharges":70.35,"TotalCharges":140.70}' | python -m json.tool
```

```bash
# Query prescriptive Causal Uplift intervention:
curl -s -X POST http://localhost:8000/uplift/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"No","Dependents":"No",
       "tenure":4,"PhoneService":"Yes","MultipleLines":"No",
       "InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No",
       "DeviceProtection":"No","TechSupport":"No","StreamingTV":"No",
       "StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes",
       "PaymentMethod":"Electronic check","MonthlyCharges":74.50,"TotalCharges":298.00}' | python -m json.tool
```

### Stage 10: Interactive Streamlit Dashboard
```bash
# Start the interactive UI
make dashboard
# Opens at: http://localhost:8501
```

### Stage 11: One-Command Docker Compose Stack
```bash
# Start API + Dashboard + MLflow + Prometheus + Grafana
make docker-up

# Verify services:
# API:        http://localhost:8000/docs
# Dashboard:  http://localhost:8501
# MLflow:     http://localhost:5000
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000
```

### Stage 12: Run Test Suite & Linting
```bash
# Run all 293+ unit and integration tests with coverage
make test

# Run style linting and strict static type checking
make lint
```

---

## ✨ Key Features

### 🧠 Machine Learning & Causal AI
- **28 engineered domain features** across 6 groups (contract dynamics, service adoption depth, billing friction, family isolation, composite risk).
- **5-stage consensus feature selection** (Variance → Correlation → Mutual Information → Random Forest → Permutation Importance).
- **LightGBM Champion** with Optuna Bayesian optimization and class-imbalance weighting.
- **Cost-matrix threshold optimization** ($p^* = 0.34$) minimizing business losses ($C_{FN}=\$500$ vs $C_{FP}=\$50$).
- **EconML T-Learner Causal Uplift Modeling** estimating Conditional Average Treatment Effects (CATE).
- **MAPIE Split-Conformal Prediction** delivering 90% confidence uncertainty sets.
- **EEOC 4/5ths Rule Fairness Auditing** verifying algorithmic parity across protected groups.
- **Non-Parametric Isotonic Calibration** using Scikit-Learn 1.6+ `FrozenEstimator`.

### 🚀 Production Serving & Feature Store
- **FastAPI REST API** with single, vectorized batch, and Feast real-time feature hydration endpoints.
- **Sub-millisecond model loading** with disk fallback chain for cold-start resilience.
- **Local SHAP Feature Attributions** identifying top churn drivers on every prediction.
- **Feast Feature Store** supporting online entity lookup via `customer_id`.
- **Request Tracing** via `X-Request-ID` UUID headers on all calls.

### 📊 Monitoring & Observability
- **Prometheus Telemetry** scraping request counts, latencies, error rates, and probability histograms at `/metrics`.
- **Pre-provisioned Grafana Dashboards** for real-time traffic and churn drift visualization.
- **Population Stability Index (PSI)** and **Chi-squared tests** detecting data drift across numerical and categorical features.
- **Wasserstein Distance** detecting model prediction drift.
- **Automated Retraining Flows** gating challenger promotion with the **Nadeau-Bengio corrected t-test**.

---

## 🏗️ Architecture Overview

```
                         ┌─────────────────────────────────────────┐
                         │            GitHub Actions CI/CD         │
                         │   Lint → Test (293) → Mypy → Build      │
                         └───────────────┬─────────────────────────┘
                                         │
    ┌────────────┐    ┌──────────┐    ┌──┴───────┐    ┌──────────────┐
    │  Raw Data  │───▶│ Validate │───▶│ Feature  │───▶│  Prefect     │
    │  (DVC)     │    │ (Pandera)│    │  Store   │    │  Pipeline    │
    └────────────┘    └──────────┘    │ (Feast)  │    │  (MLflow)    │
                                      └──────────┘    └──────┬───────┘
                                                             │
                                                    ┌────────┴────────┐
                                                    │ sklearn.Pipeline│
                                                    │ (Calibrated)    │
                                                    └────────┬────────┘
                                                             │
         ┌─────────────┐    ┌──────────────┐    ┌────────────┴────────┐
         │  Streamlit   │◀──│  FastAPI      │◀──│  Model Loader       │
         │  Dashboard   │   │  /predict     │   │  (Instant Disk      │
         │  (port 8501) │   │  /uplift      │   │   Fallback)         │
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

---

## 📂 Project Structure

```
customer-churn-mlops/
│
├── .github/workflows/         # CI/CD pipelines (lint, mypy, test, build)
│   ├── ci.yml
│   └── cd.yml
│
├── configs/                   # Type-safe configuration (YAML -> Dataclasses)
│   ├── feature_config.yaml    #   Raw & engineered feature specifications
│   ├── model_config.yaml      #   Hyperparameters, cost matrix ($500/$50), thresholds
│   ├── monitoring_config.yaml #   Drift alert thresholds (PSI, Chi-sq, Wasserstein)
│   └── training_config.yaml   #   Split ratios, CV folds, random seeds
│
├── dashboards/                # Streamlit monitoring & prediction UI
│   └── streamlit_app.py
│
├── data/                      # 4-tier data taxonomy (data/README.md)
│   ├── raw/                   #   Immutable DVC-tracked raw dataset (Telco CSV)
│   ├── processed/             #   Canonical preprocessed datasets & train/val/test splits (Parquet)
│   ├── features/              #   Feast feature store offline storage (Parquet)
│   └── reference/             #   Production drift reference snapshots (Parquet)
│
├── docker/                    # Containerization & local observability stack
│   ├── Dockerfile             #   Multi-stage API container
│   ├── Dockerfile.streamlit   #   Streamlit dashboard container
│   ├── docker-compose.yml     #   5-service stack (API, UI, MLflow, Prometheus, Grafana)
│   ├── prometheus.yml         #   Prometheus scrape config
│   └── grafana/provisioning/  #   Pre-provisioned dashboards & data sources
│
├── docs/                      # Architectural manuals & operational guides
│   ├── API.md                 #   REST API specifications & schemas
│   ├── ARCHITECTURE.md        #   Layered system design & design decisions
│   ├── DEPLOYMENT.md          #   Local, Docker Compose, and Kubernetes deployment
│   └── MONITORING.md          #   Drift monitoring & champion-challenger retraining
│
├── models/                    # Serialized production model artifacts
│   ├── champion_pipeline.joblib # Calibrated LightGBM champion pipeline
│   ├── model_metadata.json    # Champion version, threshold & metrics metadata
│   ├── conformal/             # MAPIE conformal prediction intervals & bounds
│   ├── ft_transformer/        # Deep tabular FT-Transformer model checkpoints
│   └── uplift/                # EconML CATE causal uplift meta-learner
│
├── notebooks/                 # Experimental research notebooks (.ipynb)
│   ├── 01_eda_and_business_analysis.ipynb
│   ├── 02_feature_engineering_experiments.ipynb
│   ├── 03a_algorithm_scan.ipynb
│   ├── 03b_hyperparameter_tuning.ipynb
│   ├── 03c_champion_evaluation.ipynb
│   └── README.md
│
├── pipelines/                 # Production pipeline orchestration outside notebooks
│   ├── training_pipeline.py   #   Full training, calibration, fairness & conformal flow
│   └── retraining_pipeline.py #   Closed-loop drift check & champion-challenger promotion
│
├── results/                   # Unified Project Results & Reports (results/README.md)
│   ├── MODEL_CARD.md          #   Production model card (Mitchell et al.)
│   ├── BUSINESS_CASE_STUDY.md #   Unit economics & causal uplift ROI analysis
│   ├── BENCHMARK_REPORT.md    #   10-algorithm benchmark with Nadeau-Bengio t-tests
│   ├── DRIFT_MONITORING_REPORT.md # Drift detection, PSI & population stability
│   ├── DATA_GOVERNANCE_REPORT.md  # Pandera contracts, feature lineage & privacy
│   ├── metrics/               #   JSON evaluation results (champion, benchmark, conformal, uplift, drift)
│   ├── tuning/                #   Optuna trials & algorithm sweep checkpoints (CSV/PKL)
│   └── figures/               #   Hierarchically categorized visualization assets
│       ├── eda/               #     Exploratory data analysis charts (01 to 09)
│       ├── features/          #     Mutual info, correlation & selection curves
│       ├── benchmarks/        #     Optuna sweeps & tuning history
│       └── evaluation/        #     PR curves, confusion matrices, business impact
│
├── scripts/                   # CLI utilities & standalone jobs
│   ├── prepare_feast_data.py  #   Generate Feast offline Parquet feature views
│   ├── save_reference_data.py #   Save training baseline snapshot for drift monitoring
│   ├── generate_drift_report.py # Compute PSI, Chi-sq & Wasserstein drift
│   ├── train_uplift.py        #   Train Causal Uplift (CATE) T-Learner
│   ├── train_conformal.py     #   Fit MAPIE conformal prediction wrapper
│   └── run_benchmarks.py      #   Run 5-fold CV algorithm benchmarking
│
├── src/                       # Production source code
│   ├── data/                  #   Ingestion, validation (Pandera), preprocessing
│   │   ├── ingest.py
│   │   ├── validate.py
│   │   └── preprocess.py
│   ├── features/              #   Feature store & 5-stage consensus selection
│   │   ├── feature_store.py
│   │   ├── feature_selector.py
│   │   └── feast_repo/        #   Feast Feature Store definition
│   ├── models/                #   Training, evaluation, calibration, conformal, uplift
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── pipeline.py
│   │   ├── threshold.py
│   │   ├── conformal.py
│   │   ├── uplift.py
│   │   ├── fairness_audit.py
│   │   ├── statistical_comparison.py
│   │   ├── ft_transformer_wrapper.py
│   │   ├── tabpfn_wrapper.py
│   │   ├── hpo.py
│   │   └── learning_curves.py
│   ├── monitoring/            #   Data drift detection & reference management
│   │   ├── drift_detector.py
│   │   └── reference_builder.py
│   ├── serving/               #   FastAPI prediction & causal uplift API
│   │   ├── api.py
│   │   ├── schemas.py
│   │   └── model_loader.py
│   └── utils/                 #   Configuration loader & structured logging
│       ├── config_loader.py
│       └── logging.py
│
├── tests/                     # 293 tests (unit + integration + data validation)
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── data_tests/
│
├── main.py                    # Unified CLI (train, retrain, monitor, uplift, benchmark, serve)
├── pyproject.toml             # Poetry dependencies, pytest, Black, isort, Flake8, mypy config
├── Makefile                   # Developer workflow automation
└── .pre-commit-config.yaml    # Git hooks for code quality
```

---

## 💻 Operational Usage & CLI

The platform provides a unified CLI via [`main.py`](main.py) and developer workflows via [`Makefile`](Makefile).

### Unified Operational CLI

```bash
# Production Training Flow (Prefect + Feast + MLflow + Calibration + Conformal)
poetry run python main.py train

# Drift-Triggered Automated Retraining (Champion vs Challenger with Nadeau-Bengio t-test)
poetry run python main.py retrain

# Population Drift & Feature Stability Monitoring
poetry run python main.py monitor

# Causal Uplift (CATE) Intervention Modeling
poetry run python main.py uplift

# Tabular ML Benchmarks (LightGBM vs CatBoost vs XGBoost vs TabPFN vs FT-Transformer)
poetry run python main.py benchmark

# Production FastAPI Serving
poetry run python main.py serve --port 8000
```

### Makefile Reference

| Command | Description |
|---------|-------------|
| `make setup` | First-time setup (install deps, pre-commit hooks, DVC pull) |
| `make pipeline-train` | Run automated Prefect production training pipeline |
| `make pipeline-retrain` | Run drift-triggered champion-challenger retraining flow |
| `make uplift-train` | Run Causal Uplift (CATE) prescriptive experiment |
| `make benchmark` | Run cross-validation benchmark and Nadeau-Bengio statistical tests |
| `make notebooks-exec` | Re-execute all research notebooks and update rendered outputs |
| `make serve` | Start FastAPI prediction & uplift server (port 8000) |
| `make dashboard` | Launch Streamlit monitoring dashboard (port 8501) |
| `make test` | Run all 293+ tests with coverage report |
| `make test-unit` | Run unit tests only |
| `make test-integration` | Run integration tests only |
| `make format` | Auto-format code (Black + isort) |
| `make lint` | Run linters (Flake8 + mypy) |
| `make drift-report` | Generate a data drift monitoring report |
| `make docker-up` | Start full Docker Compose stack (API, Streamlit, MLflow, Prometheus, Grafana) |
| `make docker-down` | Stop all Docker containers |

---

## 📡 API Reference

The serving engine exposes versioned routes (`/api/v1/*`) alongside root aliases for backward compatibility. All responses include OWASP API Security Top 10 hardening headers (`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Cache-Control: no-store`, `Strict-Transport-Security`) and unique `X-Request-ID` tracing headers. A thread-safe sliding-window rate limiter enforces `600` req/min with standard RFC 6585 `429 Too Many Requests` responses.

### Serving Endpoints Summary

| Method | Path | Category | Description |
|---|---|---|---|
| `POST` | `/api/v1/predict` | Predictive ML | Single customer churn prediction (SHAP + Conformal sets) |
| `POST` | `/api/v1/predict/batch` | Predictive ML | Vectorized synchronous batch prediction (1–100 customers) |
| `POST` | `/api/v1/predict/batch/async` | Asynchronous ML | Non-blocking async batch job submission (HTTP 202 Accepted) |
| `GET`  | `/api/v1/predict/batch/jobs/{id}` | Asynchronous ML | Poll status and retrieve completed async batch predictions |
| `GET`  | `/api/v1/predict/customer/{id}` | Feature Store | Real-time prediction fetching raw features from Feast online store |
| `POST` | `/api/v1/uplift/predict` | Causal AI | Causal CATE treatment effect and prescriptive quadrant |
| `POST` | `/api/v1/uplift/batch` | Causal AI | Batch prescriptive campaign assignment (Persuadables, Sleeping Dogs) |
| `GET`  | `/api/v1/uplift/info` | Causal AI | Causal model metadata (treatment feature, model type) |
| `GET`  | `/api/v1/model/info` | Diagnostics | Production model metadata (algorithm, version, optimal threshold) |
| `GET`  | `/health` | System Probe | Liveness probe returning 200 if container process is alive |
| `GET`  | `/readiness` | System Probe | Kubernetes readiness probe (returns 503 until model loads) |
| `GET`  | `/metrics` | Observability | Prometheus time-series metrics scrape target |
| `GET`  | `/docs` | Documentation | Interactive OpenAPI / Swagger UI documentation |

*For complete curl commands, request bodies, and JSON schemas, see [**`docs/API.md`**](docs/API.md).*

---

## 🐳 Docker Deployment

Deploy the entire microservice ecosystem with Docker Compose, featuring explicit CPU and memory resource limits (`limits` & `reservations`):

```bash
# Build and start all 5 services
docker compose -f docker/docker-compose.yml up --build -d

# Verify running services & health statuses
docker compose -f docker/docker-compose.yml ps

# Services available:
#   API:        http://localhost:8000/docs
#   Dashboard:  http://localhost:8501
#   MLflow:     http://localhost:5000
#   Prometheus: http://localhost:9090
#   Grafana:    http://localhost:3000

# Stop services
docker compose -f docker/docker-compose.yml down
```

---

## ☸️ Kubernetes Deployment (Kustomize)

The repository provides production-grade declarative Kubernetes manifests under [`deploy/kubernetes/`](deploy/kubernetes/) bundled via **Kustomize**:

- **Rolling Updates**: Zero-downtime rolling update strategy (`maxSurge: 1`, `maxUnavailable: 0`).
- **Decoupled Probes**: Liveness probe (`/health`) decoupled from Readiness probe (`/readiness`), preventing premature traffic routing.
- **Horizontal Pod Autoscaling (HPA)**: Automatically scales between **2 and 10 pods** based on CPU (75%) and Memory (80%) utilization.
- **Security & Governance**: Container runs with `runAsNonRoot: true`, user UID 1000, dropped Linux capabilities, and resource boundaries (500m/512Mi request, 2000m/2048Mi limit).

### 1. Offline Validation (No Cloud Cluster Required — 100% Free)
Validate and compile the complete Kubernetes manifest bundle locally without provisioning any live infrastructure:

```bash
# Validate manifests via Makefile
make k8s-validate

# Or inspect the compiled YAML output through Kustomize
kubectl kustomize deploy/kubernetes
```

### 2. Local Execution via `kind` (Kubernetes in Docker)
Run a real ephemeral Kubernetes cluster locally inside Docker:

```bash
# 1. Create a local cluster inside Docker
kind create cluster --name churn-cluster

# 2. Deploy the Kustomize bundle
kubectl apply -k deploy/kubernetes

# 3. Port-forward the service to your workstation
kubectl port-forward -n churn-serving svc/churn-api-service 8000:8000
```

*For complete manifest specifications, see [**`deploy/kubernetes/README.md`**](deploy/kubernetes/README.md) and [**`docs/DEPLOYMENT.md`**](docs/DEPLOYMENT.md).*

---

## 🧪 Testing & Quality Assurance

The project enforces strict software engineering standards with **303 tests**:

```bash
# Run complete test suite with coverage
make test

# Run strict static type checking across all source files
poetry run mypy src/

# Run flake8 style checks
poetry run flake8 src/ tests/ scripts/
```

### Test Coverage Highlights
- **Schema Contracts**: Pandera input schema validation and boundary checks.
- **Unit Tests**: Pure function assertions for all 28 engineered features, cost-matrix optimization, calibration, and conformal prediction sets.
- **Integration Tests**: FastAPI TestClient end-to-end verification of `/predict`, `/predict/batch`, `/predict/batch/async`, `/predict/customer/{id}`, `/readiness`, `/health`, rate limiting 429 enforcement, and OWASP security headers.
- **Reproducibility**: Parameterized fixtures, registered pytest markers (`slow`, `integration`, `unit`, `smoke`), and deterministic random seeds.

---

## ⚙️ Configuration Architecture

All operational settings are decoupled from code into YAML files in `configs/`:
- [`configs/feature_config.yaml`](configs/feature_config.yaml): Feature types, target column, engineered feature definitions.
- [`configs/model_config.yaml`](configs/model_config.yaml): Hyperparameters, cost matrix ($C_{FN}=500, C_{FP}=50$), performance gates.
- [`configs/training_config.yaml`](configs/training_config.yaml): Split ratios (70/10/20), cross-validation folds ($K=5$).
- [`configs/monitoring_config.yaml`](configs/monitoring_config.yaml): Drift alert limits (PSI = 0.25, Chi-sq $\alpha = 0.05$, prediction drift = 0.15).

---

## 🔄 MLOps Practices

| Dimension | Engineering Implementation |
|---|---|
| **Data Versioning** | DVC tracks raw datasets with remote hash verification |
| **Data Validation** | Pandera enforces column types, missingness, and range constraints at ingestion |
| **Feature Store** | Feast materializes features to online store for real-time inference |
| **Pipeline Parity** | `sklearn.Pipeline` encapsulates feature engineering, scaling, and classification |
| **Experiment Tracking** | MLflow logs parameters, metrics, confusion matrices, and model artifacts |
| **Model Registry** | MLflow Model Registry manages champion-challenger transitions |
| **Continuous Integration** | GitHub Actions runs lint, mypy type checks, and 293 pytest tests on every PR |
| **Continuous Deployment** | Automated multi-stage Docker image builds and verification |
| **Drift Detection** | Decile PSI, Chi-squared goodness-of-fit, and Wasserstein distance tracking |
| **Telemetry & Observability** | Prometheus time-series metrics scraped into provisioned Grafana dashboards |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
