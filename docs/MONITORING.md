# Monitoring & Drift Detection Guide

This document explains how the data drift monitoring system works, how to use it, and how to interpret the results.

---

## Table of Contents

- [Why Monitor for Drift?](#why-monitor-for-drift)
- [Types of Drift Detected](#types-of-drift-detected)
- [Quick Start](#quick-start)
- [Understanding the Results](#understanding-the-results)
- [Configuration](#configuration)
- [Integration with Retraining](#integration-with-retraining)

---

## Why Monitor for Drift?

A machine learning model is a **snapshot of patterns observed in historical data**. When the production data distribution shifts away from that snapshot, the model's predictions become unreliable — even if the model code hasn't changed.

Common causes of data drift in churn prediction:
- **New pricing plans** change `MonthlyCharges` distribution
- **Market expansion** shifts `Contract` type ratios
- **Seasonal effects** alter customer behavior patterns
- **Upstream system changes** modify how features like `TotalCharges` are calculated

> **Key Insight:** Data drift is a *leading indicator* of model degradation. By the time model performance metrics drop, you've already been making bad predictions for weeks. Drift detection gives you an early warning.

---

## Types of Drift Detected

### 1. Real-Time API Metrics (Prometheus & Grafana)

**What it measures:** Continuous tracking of API requests, prediction latency overhead, error rates, and the raw distribution of churn probabilities streaming into the endpoint.

**Method:** Time-series metric `Counters` and `Histograms` scraped concurrently by **Prometheus** at the `/metrics` endpoint.

**Interpretation:**
Visualized dynamically inside **Grafana** using the pre-provisioned "Churn Prediction API" dashboard. This isn't "drift" per se, but instead provides a second-by-second operational view to ensure the model service isn't crashing and predictions are being served in a timely manner.

### 2. Numerical Feature Drift — PSI (Population Stability Index)

**What it measures:** Whether the distribution shape of a numerical feature has changed.

**Method:** Quantile-based binning of both reference and current data, then computing:

```
PSI = Σ (P_current - P_reference) × ln(P_current / P_reference)
```

**Interpretation:**

| PSI Value | Meaning | Action |
|-----------|---------|--------|
| < 0.10 | No significant shift | No action needed |
| 0.10 – 0.25 | Moderate shift | Monitor closely |
| > 0.25 | Significant drift | Investigate + retrain |

**Example:** If `MonthlyCharges` has PSI = 0.32, it means the distribution of monthly charges in production data is significantly different from training data — perhaps due to a pricing change.

### 3. Categorical Feature Drift — Chi-Squared Test

**What it measures:** Whether the category proportions of a categorical feature have changed.

**Method:** Chi-squared goodness-of-fit test comparing observed (current) frequencies against expected (reference) frequencies.

**Interpretation:**

| p-value | Meaning | Action |
|---------|---------|--------|
| > 0.05 | No significant change | No action needed |
| ≤ 0.05 | Statistically significant change | Investigate + potential retrain |

**Example:** If `Contract` type distribution changes from (55% month-to-month, 25% one-year, 20% two-year) to (30% month-to-month, 40% one-year, 30% two-year), the chi-squared test will flag this as drift.

### 4. Prediction Drift — Mean Probability Shift

**What it measures:** Whether the model's average predicted churn probability has changed.

**Method:** Relative shift between reference and current mean predictions:

```
drift = |mean(current_proba) - mean(reference_proba)| / mean(reference_proba)
```

**Interpretation:**

| Relative Shift | Meaning | Action |
|---------------|---------|--------|
| < 0.10 (10%) | Normal variation | No action needed |
| 0.10 – 0.15 | Moderate shift | Monitor closely |
| > 0.15 (15%) | Significant output drift | Investigate root cause |

---

## Quick Start

### Step 1: Save Reference Data

After training your model, save the training data as the drift reference:

```bash
poetry run python scripts/save_reference_data.py
```

This creates three files in `data/reference/`:
- `reference_features.parquet` — Feature distributions
- `reference_probabilities.npy` — Model predictions (if provided)
- `reference_metadata.json` — Metadata (timestamp, model version, feature names)

### Step 2: Generate a Drift Report

Compare current data against the reference:

```bash
# Self-comparison (baseline — should show no drift)
poetry run python scripts/generate_drift_report.py

# Compare against new production data
poetry run python scripts/generate_drift_report.py --current data/raw/new_data.csv

# Save report to a specific location
poetry run python scripts/generate_drift_report.py --output reports/drift_report.json
```

### Step 3: Interpret the Report

```
=======================================================
  DRIFT MONITORING REPORT
=======================================================
  Reference shape     : (7043, 47)
  Current shape       : (1500, 47)
  Overall drift       : YES ⚠️
  Drifted features    : 2
    - MonthlyCharges
    - Contract
  Prediction drift    : 0.1823

  Report saved to: /path/to/reports/drift_report.json
=======================================================
```

### Programmatic Usage

```python
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.reference_builder import load_reference

# Load reference data
reference_df, reference_proba = load_reference()

# Initialize detector
detector = DriftDetector(reference_df)

# Generate report
report = detector.generate_report(
    current_df=production_data_df,
    reference_proba=reference_proba,
    current_proba=model.predict_proba(production_data_df)[:, 1],
)

# Check results
if report.overall_drift_detected:
    print(f"ALERT: {len(report.drifted_features)} features drifted!")
    print(f"Drifted: {report.drifted_features}")
else:
    print("No drift detected ✅")

# Export for logging
report_dict = report.to_dict()  # JSON-serializable dict
```

---

## Configuration

All thresholds are configurable in `configs/monitoring_config.yaml`:

```yaml
# Numerical feature drift threshold (PSI)
psi_threshold: 0.25

# Categorical feature drift significance level
chi_squared_alpha: 0.05

# Prediction drift — relative mean shift threshold
prediction_drift_threshold: 0.15

# Performance drift — F1 score drop vs. baseline
performance_drop_threshold: 0.05

# Reference window — days of training data to use as baseline
reference_window_days: 90

# Monitoring window — days of production data to compare
monitoring_window_days: 7

# Label delay — days before ground truth labels are available
label_delay_days: 45
```

### Tuning Guidelines

- **More sensitive detection:** Lower `psi_threshold` (e.g., 0.15) and raise `chi_squared_alpha` (e.g., 0.10)
- **Fewer false alarms:** Raise `psi_threshold` (e.g., 0.30) and lower `chi_squared_alpha` (e.g., 0.01)
- **Churn-specific:** Customer churn labels typically take 30–60 days to materialize. The `label_delay_days` parameter accounts for this when scheduling performance evaluation.

---

## Understanding the Results

### DriftReport Fields

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | string | ISO timestamp of report generation |
| `reference_shape` | tuple | (rows, cols) of reference data |
| `current_shape` | tuple | (rows, cols) of current data |
| `numerical_drift` | list | PSI results per numerical feature |
| `categorical_drift` | list | Chi-squared results per categorical feature |
| `prediction_drift_score` | float | Relative mean probability shift |
| `prediction_is_drifted` | bool | Whether prediction drift exceeds threshold |
| `overall_drift_detected` | bool | True if ANY drift was detected |
| `drifted_features` | list | Names of features that exceeded thresholds |
| `summary` | string | Human-readable summary message |

### Per-Feature Result Fields

| Field | Type | Description |
|-------|------|-------------|
| `feature_name` | string | Column name |
| `drift_score` | float | PSI value or chi-squared p-value |
| `is_drifted` | bool | Whether threshold was exceeded |
| `method` | string | `"psi"` or `"chi_squared"` |
| `threshold` | float | Threshold used for this feature |
| `details` | dict | Additional stats (mean, std, chi2 statistic) |

---

## Integration with Retraining

The drift monitoring system is designed to be the trigger for automated retraining. Here's the recommended workflow:

```
Daily Production Data
        │
        ▼
  Drift Detection Job (scheduled)
        │
        ├── No drift detected → Log metrics, continue
        │
        └── Drift detected ────▶ Alert Team
                                      │
                                      ▼
                               Investigate Root Cause
                                      │
                   ┌──────────────────┼──────────────┐
                   │                  │              │
              Data Issue       Distribution    Model Decay
              (fix upstream)    Shift (real)   (retrain)
                                      │              │
                                      ▼              ▼
                                 Retrain Model ◀─────┘
                                      │
                                      ▼
                               Evaluate on Test Set
                                      │
                                      ▼
                               Update Reference Data
                                      │
                                      ▼
                               Deploy New Model
```

### Recommended Schedule

| Check | Frequency | Automation |
|-------|-----------|-----------|
| Data drift (PSI + chi-squared) | Daily | Cron job / Prefect flow |
| Prediction drift | Daily | Cron job / Prefect flow |
| Performance evaluation | After label delay (45 days) | Scheduled pipeline |
| Reference data update | After each retraining | Post-training script |
