# Architecture Guide

Deep dive into the architecture, design decisions, and data flow of the Customer Churn MLOps pipeline.

---

## Table of Contents

- [High-Level Architecture](#high-level-architecture)
- [Module Dependency Graph](#module-dependency-graph)
- [Data Flow](#data-flow)
- [Key Design Decisions](#key-design-decisions)
- [Configuration Architecture](#configuration-architecture)
- [Model Serving Architecture](#model-serving-architecture)
- [Monitoring Architecture](#monitoring-architecture)

---

## High-Level Architecture

The system follows a **layered architecture** where each layer has a single responsibility and dependencies only flow downward:

```
┌──────────────────────────────────────────────────────────────────┐
│                         Presentation                             │
│ Streamlit (port 8501) │ FastAPI Docs (port 8000) │ Grafana (3000)│
├──────────────────────────────────────────────────────────────────┤
│                         Serving Layer                            │
│   api.py  │  schemas.py (Pydantic v2)  │  model_loader.py       │
├──────────────────────────────────────────────────────────────────┤
│                         Model Layer                              │
│   train.py  │  pipeline.py  │  evaluate.py  │  threshold.py     │
├──────────────────────────────────────────────────────────────────┤
│                         Feature Layer                            │
│   feature_store.py (28 features)  │  feature_selector.py        │
├──────────────────────────────────────────────────────────────────┤
│                         Data Layer                               │
│   ingest.py  │  validate.py (Pandera)  │  preprocess.py         │
├──────────────────────────────────────────────────────────────────┤
│                         Infrastructure                           │
│   config_loader.py  │  logging.py  │  drift_detector.py         │
├──────────────────────────────────────────────────────────────────┤
│                         External Services                        │
│  MLflow │ DVC │ Docker │ GitHub Actions │ Prometheus (port 9090) │
└──────────────────────────────────────────────────────────────────┘
```

---

## Module Dependency Graph

```
src/utils/config_loader.py    ←── Everything depends on this
src/utils/logging.py           ←── Everything depends on this
        │
        ▼
src/data/ingest.py            ← Loads raw CSV, minimal cleanup
src/data/validate.py          ← Pandera schema enforcement
src/data/preprocess.py        ← Target encoding, column cleanup
        │
        ▼
src/features/feature_store.py  ← 28 stateless features, routing lists
src/features/feature_selector.py ← 5-stage selection (optional)
        │
        ▼
src/models/pipeline.py        ← sklearn.Pipeline factory
src/models/train.py           ← Training orchestration
src/models/evaluate.py        ← ML + business metrics
src/models/threshold.py       ← Cost-optimal threshold tuning
        │
        ▼
src/serving/schemas.py        ← Pydantic v2 request/response models
src/serving/model_loader.py   ← Thread-safe lazy loading
src/serving/api.py            ← FastAPI endpoints
        │
        ▼
src/monitoring/drift_detector.py  ← PSI, chi-squared, prediction drift
src/monitoring/reference_builder.py ← Parquet snapshots
```

---

## Data Flow

### Training Pipeline

```
Raw CSV (data/raw/)
    │
    ▼ ingest.load_for_training()
Loaded DataFrame (7043 rows, 21 cols)
    │
    ▼ validate.validate_raw_data()
    │   ├── Fix TotalCharges blank strings → MonthlyCharges
    │   └── Pandera schema: types, ranges, allowed values
Validated DataFrame
    │
    ▼ preprocess.run_preprocessing()
    │   ├── Encode target: "Yes"/"No" → 1/0
    │   ├── Cast SeniorCitizen: int → str ("0"/"1")
    │   └── Drop customerID column
(X: features DataFrame, y: target Series)
    │
    ▼ train_test_split (3-way: 70/10/20)
(X_train, X_val, X_test, y_train, y_val, y_test)
    │
    ▼ pipeline.build_pipeline()
    │   ├── Step 1: FunctionTransformer(engineer_features)  → 28 new columns
    │   ├── Step 2: ColumnTransformer
    │   │     ├── OneHotEncoder for categorical features
    │   │     ├── StandardScaler for numerical features
    │   │     └── passthrough for binary/ordinal features
    │   └── Step 3: LGBMClassifier(hyperparams from config)
sklearn.Pipeline (single serializable object)
    │
    ▼ pipeline.fit(X_train, y_train)
    │
    ▼ threshold.find_cost_optimal_threshold(y_val, y_val_proba)
    │   └── Minimizes: FN_cost × FN_count + FP_cost × FP_count
Optimal threshold (e.g., 0.34)
    │
    ▼ evaluate.evaluate(y_test, y_test_proba, threshold)
    │   ├── ML metrics: ROC-AUC, PR-AUC, F1, Precision, Recall
    │   └── Business metrics: total_cost, cost_saved, ROI
Final metrics dict → MLflow logging
```

### Inference Pipeline

```
JSON request body (19 fields)
    │
    ▼ Pydantic v2 validation (schemas.CustomerFeatures)
    │   └── Type checks, value constraints, enum validation
Validated CustomerFeatures object
    │
    ▼ Convert to 1-row DataFrame
    │
    ▼ model_loader.get_model() (lazy, thread-safe)
    │   ├── Try 1: MLflow Model Registry
    │   ├── Try 2: Local training (in-memory, ~15s)
    │   └── Cache: subsequent calls return instantly
(pipeline, threshold)
    │
    ▼ pipeline.predict_proba(df)[:, 1]
churn_probability (float)
    │
    ▼ Apply threshold + risk tier classification
    │   ├── probability ≥ 0.60  → HIGH_RISK
    │   ├── 0.35 ≤ prob < 0.60  → MEDIUM_RISK
    │   └── probability < 0.35  → LOW_RISK
    │
    ▼ PredictionResponse JSON
{churn_probability, will_churn, risk_tier, threshold_used, request_id}
```

---

## Key Design Decisions

### 1. Single sklearn.Pipeline Object

**Decision:** Encapsulate feature engineering, preprocessing, and model inference into a single `sklearn.Pipeline`.

**Why:** This is the most critical design decision. It makes training-serving skew *structurally impossible*. The exact same transformations that ran during training run during inference — there's no separate "feature engineering service" or manual preprocessing step that could diverge.

```python
# The Pipeline contains 3 steps:
# Step 1: FunctionTransformer(engineer_features)  → adds 28 columns
# Step 2: ColumnTransformer                       → OHE + Scale + passthrough
# Step 3: LGBMClassifier                          → prediction
pipeline = Pipeline([
    ("feature_engineering", FunctionTransformer(engineer_features)),
    ("preprocessor", column_transformer),
    ("classifier", LGBMClassifier(**hyperparams)),
])
```

### 2. Stateless Feature Engineering

**Decision:** All 28 engineered features are computed via pure, stateless functions.

**Why:** Stateful features (e.g., running averages, lookups against training data) create subtle bugs when applied to single production rows. Stateless features guarantee identical behavior regardless of batch size or execution context.

```python
# ✅ Stateless — works on any single row
df["charge_to_tenure_ratio"] = df["MonthlyCharges"] / (df["tenure"] + 1)

# ❌ Stateful — requires training set context
df["charge_percentile"] = df["MonthlyCharges"].rank(pct=True)  # WRONG in production
```

### 3. Cost-Matrix Threshold Optimization

**Decision:** Use a business cost matrix ($500/FN, $20/FP) instead of the default 0.5 threshold.

**Why:** In churn prediction, missing a churner (false negative) is 25x more expensive than a false alarm (false positive). The default 0.5 threshold optimizes for accuracy, which is misleading with imbalanced data (73% non-churners). Our cost-optimal threshold (~0.34) catches more churners at the cost of more false alarms — a profitable trade-off.

### 4. Lazy Model Loading

**Decision:** Load the model on the first `/predict` request, not at API startup.

**Why:**
- Faster API startup time (important for container orchestrators that need quick health check responses)
- Graceful handling of MLflow unavailability (falls back to local training)
- Thread-safe via `threading.Lock` — concurrent first-requests don't trigger duplicate loading

### 5. 3-Way Data Split

**Decision:** Train (70%) / Validation (10%) / Test (20%) instead of the common Train/Test split.

**Why:** The validation set is used exclusively for threshold optimization. If we optimized the threshold on the test set, we'd be leaking test information into the model's decision boundary — the reported test metrics would be overly optimistic.

### 6. Config-Driven Architecture

**Decision:** All tunable parameters live in YAML files, loaded into typed Python dataclasses.

**Why:**
- Business users can adjust cost matrix without touching code
- New deployments can use different thresholds without rebuilding containers
- Type safety catches misconfiguration at load time, not at runtime

---

## Configuration Architecture

```
configs/
├── feature_config.yaml      # What features exist
├── model_config.yaml         # How the model is built
├── training_config.yaml      # How training is orchestrated
└── monitoring_config.yaml    # When to trigger drift alerts
         │
         ▼
src/utils/config_loader.py
         │
    get_config() → singleton
         │
         ▼
    @dataclass FeatureConfig
    @dataclass ModelConfig
    @dataclass TrainingConfig
    @dataclass MonitoringConfig
    @dataclass ProjectConfig (wraps all above)
```

The `get_config()` function implements the singleton pattern — config is loaded once and cached. Tests reset the singleton via `reset_config()` in `conftest.py`.

---

## Model Serving Architecture

```
FastAPI Application (api.py)
    │
    ├── Middleware: CORS + Request ID injection
    │
    ├── POST /predict
    │   ├── Pydantic validation (automatic, before handler)
    │   ├── model_loader.get_model() → (pipeline, threshold)
    │   ├── pipeline.predict_proba(df)
    │   └── Apply threshold → response
    │
    ├── POST /predict/batch
    │   ├── Loop: validate + predict per customer
    │   └── Aggregate risk tier counts
    │
    ├── GET /health
    │   └── Returns status + model_loaded flag
    │
    ├── GET /model/info
    │   └── Returns model metadata from model_loader
    │
    └── GET /metrics
        └── Exposes Prometheus counters and histograms

Model Loader (model_loader.py)
    │
    ├── Lazy loading (first request triggers load)
    ├── Thread-safe (threading.Lock)
    │
    ├── Strategy 1: MLflow Model Registry
    │   └── Fetch by registered_model_name + stage
    │
    └── Strategy 2: Local Training Fallback
        ├── Load raw data → validate → preprocess
        ├── 3-way split (70/10/20)
        ├── Build pipeline → fit → threshold optimize
        └── Cache in module globals (memory only)
```

---

## Monitoring Architecture

The system implements a dual-mode monitoring architecture combining real-time API observability with periodic batch data drift detection.

### 1. Real-Time Observability (Prometheus & Grafana)

```
                            Production Requests
                                     │
    ┌────────────────────────────────▼─────────────────────────────────┐
    │                       FastAPI /predict                           │
    │  Records Counter (reqs, errors) & Histogram (latency, probs)     │
    └────────────────────────────────┬─────────────────────────────────┘
                                     │ Exposes /metrics
    ┌────────────────────────────────▼─────────────────────────────────┐
    │                       Prometheus (9090)                          │
    │      Iteratively scrapes and stores time-series metrics          │
    └────────────────────────────────┬─────────────────────────────────┘
                                     │ Queried via PromQL
    ┌────────────────────────────────▼─────────────────────────────────┐
    │                        Grafana (3000)                            │
    │         Visualizes request load, churn drift, and errors         │
    └──────────────────────────────────────────────────────────────────┘
```

### 2. Batch Data Drift Detection (PSI & Chi-Squared)

```
Training Data (saved as reference)
    │
    ▼ reference_builder.build_reference()
    │   ├── reference_features.parquet    (feature distributions)
    │   ├── reference_probabilities.npy   (model predictions)
    │   └── reference_metadata.json       (timestamps, model version)
    │
    │                    Production Requests (over time)
    │                              │
    └──────────┐                   │
               ▼                   ▼
        DriftDetector(reference_df)
               │
    ┌──────────┼──────────────────────┐
    │          │                      │
    ▼          ▼                      ▼
  PSI       Chi-squared         Prediction Drift
(numerical) (categorical)     (mean probability shift)
    │          │                      │
    └──────────┴──────────────────────┘
               │
               ▼
         DriftReport
    ├── per-feature drift scores
    ├── overall_drift_detected (bool)
    ├── drifted_features (list)
    └── summary message

    Thresholds (from monitoring_config.yaml):
    ├── PSI > 0.25         → significant numerical drift
    ├── chi-squared p < 0.05 → significant categorical drift
    └── prediction drift > 15% → model output has shifted
```
