# 🔭 Customer Churn Predictor: End-to-End MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10+-blue.svg?logo=mlflow)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-13c2c2.svg?logo=data-version-control)](https://dvc.org/)
[![Pandera](https://img.shields.io/badge/Pandera-Validation-ff69b4.svg)](https://pandera.readthedocs.io/)
[![pytest](https://img.shields.io/badge/pytest-Passing-success.svg?logo=pytest)](https://docs.pytest.org/en/latest/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-grade, end-to-end Machine Learning Operations (MLOps) pipeline built to predict customer churn using the Telco Customer Churn dataset.

This project goes beyond standard machine learning objectives by explicitly optimizing for **real business value**. By integrating a **cost-matrix threshold optimization** mechanism, it balances the financial trade-off between the cost of false positives (unnecessary retention costs) and false negatives (lost customer revenue). The architecture is designed with **production reliability** at its core, ensuring seamless transitions from experimentation to deployment while preventing training-serving skew.

---

## 🏗️ Architecture & Core Features

### 1. Robust Data Pipeline & Feature Store
- **Data Validation (`pandera`)**: Strict schema enforcement guarantees data integrity at both the raw ingestion step and API inference boundaries.
- **Stateless Engineering (`sklearn.Pipeline`)**: 28 richly engineered behavioral features compiled into a stateless `FunctionTransformer`. This enforces identical transformations between training and inference environments, completely eliminating training-serving skew.
- **5-Stage Feature Selection**: A rigorous consensus pipeline (Variance, Correlation, Mutual Information, Random Forest Importance, and Permutation Importance) extracts only statistically significant features.

### 2. Business-Optimized Model Training
- **Algorithm**: `LightGBM` wrapped tightly within an `sklearn.Pipeline`.
- **3-Way Data Split**: Rigorous `Train/Validation/Test` stratifications to prevent data leakage during hyperparameter tuning and threshold optimization.
- **Asymmetric Cost Matrix**: Instead of defaulting to a 0.5 decision threshold, the pipeline simulates financial impact (e.g. $500 cost per missed churn vs $20 cost per false alarm intervention) to dynamically locate the **Cost-Optimal Threshold**.

### 3. Production Serving Layer
- **Real-Time API (`FastAPI`)**: Exposes `/predict` and `/predict/batch` endpoints, complete with Pydantic v2 response schemas and interactive OpenAPI `/docs`.
- **Dual-Mode Thread-Safe Loader**: The API operates smoothly via MLflow Model Registry fallback chains or self-trained local caching during local development.
- **Observability**: Implements `X-Request-ID` middleware to track single and batch prediction requests through logs.

### 4. MLOps Integrations
- **MLflow Tracking & Registry**: Centrally logs metrics, parameters, and serialized pipeline artifacts for strict reproducibility.
- **DVC (Data Version Control)**: Decouples dataset versioning from source control.
- **Config-Driven Architecture**: YAML to Python `dataclass` singletons ensure type-safe environment adjustments without touching application code.
- **Extensive Testing**: 240+ passing unit and integration tests enforcing schemas, structural constraints, and API flow using HTTPX `TestClient`.

---

## 📂 Project Structure

```text
customer-churn-mlops/
├── configs/                   # Type-safe YAML configurations (model, features, training)
├── data/
│   ├── raw/                   # DVC-tracked raw datasets
│   └── processed/             # Interim/cleaned datasets (ignored from git)
├── notebooks/                 # Exploratory Data Analysis & Tuning experiments
├── src/
│   ├── data/                  # Ingestion, Pandera validation, and leakage-free splitting
│   ├── features/              # Feature engineering functions & 5-stage selector
│   ├── models/                # Training, Pipeline wrappers, Evaluation, Threshold tuning
│   ├── serving/               # FastAPI layer: api.py, schemas.py, model_loader.py
│   └── utils/                 # Singleton config loader and deterministic logging
├── tests/
│   ├── unit/                  # 200+ unit tests covering validation, components, thresholds
│   └── integration/           # Integration tests targeting the FastAPI endpoints
├── pyproject.toml             # Poetry dependencies, pytest, formatting, and lint configs
└── Makefile                   # Command aliases for testing, linting, and serving
```

---

## 🚀 Setup & Installation

**Prerequisites:** Python 3.11+, Poetry, and Make.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/customer-churn-mlops.tgit
   cd customer-churn-mlops
   ```

2. **Install dependencies:**
   ```bash
   poetry install
   ```

3. **Environment Setup (Optional MLflow configuration):**
   Copy `.env.example` to `.env` and configure your credentials if connecting to a remote MLflow tracking server (e.g., DagsHub). Otherwise, default local tracking will be used.
   ```bash
   cp .env.example .env
   ```

---

## 💻 Usage

We use a `Makefile` for automated task runners. Ensure you are running commands inside the active Poetry shell (`poetry shell`) or prepend with `poetry run`.

### 1. Data Replication & Pipeline setup
Pull the version-controlled datasets to ensure you are experimenting on identical baseline distributions:
```bash
make setup
```

### 2. Run the Full Training Pipeline
You can trigger the entire end-to-end training pipeline. This sequence ingests data, validates it using Pandera schemas, engineers 28 custom features, tunes the LightGBM hyperparameters, optimizes the business threshold, and natively logs all coefficients, metrics, and models to MLflow.
```bash
make train
```
**Results & Tracking:**
- **Metrics**: The pipeline outputs complex evaluation metrics onto the terminal and logs ROC AUC, PR AUC, and dynamic financial insights directly.
- **Model Artifacts**: The strictly compiled `sklearn.Pipeline` object (preprocessor + model) is stored via MLflow and becomes ready for inference.

### 3. Start the Inference API
Serve your trained model dynamically utilizing our ASGI FastAPI service. *If the model wasn't registered to an MLflow tracking server, the API gracefully falls back to train and cache it purely in-memory in ~15 seconds.*
```bash
make serve
# Available locally at: http://localhost:8000/docs
```

### 4. Querying the Model
Once the server is running on `localhost:8000`, you can interact with the endpoints.

**Single Prediction (cURL)**:
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
**Expected Response**:
```json
{
  "churn_probability": 0.5843,
  "risk_tier": "HIGH_RISK",
  "will_churn": true,
  "threshold_used": 0.34,
  "request_id": "a1b2c3d4-e5f6-7890-abcd-1234567890ab"
}
```

### 5. Running the Tests & Linters
Validate that the data interfaces, business constraints, schemas, and endpoint responses remain perfectly healthy.
```bash
make test          # Runs 240+ unit & integration tests
make format        # Auto-formats source code with Black and Isort
make lint          # Validates types and style protocols with MyPy & Flake8
```

---

## 📈 Future Roadmap

The training and API environments are fully functional, resilient, and 100% tested. The upcoming operational components include:

- [ ] **Dockerization**: Containerizing the FastAPI application.
- [ ] **Evidently AI / Monitoring**: Data drift observability and rolling performance metric thresholds in `src/monitoring/`.
- [ ] **CI/CD Pipelines**: GitHub Actions specifically targeting tests, linting, and automated pipeline triggers upon Pull Requests.
- [ ] **Streamlit / Dashboard**: Translating API results into an interactive business intelligence visualizer.

---
_Building data science that aligns tightly with engineering rigor and business impacts._
