# Production Drift & Population Stability Monitoring Report

**Generated:** 2026-10-12T03:55:21.565151+00:00
**Evaluation Scope:** Continuous Distribution Drift Detection against Baseline Snapshot

---

## 1. Executive Summary

| Monitoring Dimension | Value | Operational Status |
|---|---|---|
| **Overall Drift Status** | False | ✅ **NO DRIFT DETECTED (STABLE)** |
| **Reference Snapshot Shape** | `(7043, 47)` | Baseline reference |
| **Current Production Shape** | `(7043, 47)` | Production stream |
| **Drifted Features** | `0` | Alert threshold $\ge 3$ |
| **Prediction Drift Score (Wasserstein)** | `0.0000` | Threshold: 0.10 |

---

## 2. Drift Detection Methodology & Thresholds

- **Numerical Features (Population Stability Index - PSI)**:
  - $\text{PSI} < 0.10$: Minimal shift, distribution stable.
  - $0.10 \le \text{PSI} < 0.20$: Moderate shift, warning trigger.
  - $\text{PSI} \ge 0.20$: Significant shift, auto-retrain trigger.
- **Categorical Features**:
  - Two-sample Chi-Square goodness-of-fit test (Significance level $\alpha = 0.05$).
- **Prediction Drift**:
  - Wasserstein distance metric on calibrated model output probabilities.

---

## 3. Top Feature Stability Analysis

| Feature Name | PSI Score | Status |
|---|---|---|
| `tenure` | 0.0000 | 🟢 Stable |
| `MonthlyCharges` | 0.0000 | 🟢 Stable |
| `TotalCharges` | 0.0000 | 🟢 Stable |
| `is_month_to_month` | 0.0000 | 🟢 Stable |
| `service_adoption_count` | 0.0000 | 🟢 Stable |
| `has_protection_bundle` | 0.0000 | 🟢 Stable |
| `charge_to_tenure_ratio` | 0.0000 | 🟢 Stable |
| `is_fiber_optic` | 0.0000 | 🟢 Stable |
| `avg_charge_per_service` | 0.0000 | 🟢 Stable |
| `has_family` | 0.0000 | 🟢 Stable |

---

## 4. Drifted Features List

_None. All feature distributions align with baseline._

---

## 5. Automated Operational Recommendations

1. **Retraining Action**: If overall drift is flagged, execute
   `poetry run python main.py retrain` to initiate champion-challenger tournament.
2. **Monitoring Dashboards**: Real-time feature PSI decile distribution is available
   in the Streamlit monitoring tab (`make dashboard`).
