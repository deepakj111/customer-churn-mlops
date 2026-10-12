# Unified Project Results Catalog

This directory is the single source of truth for **all project results**: executive research whitepapers, machine-readable evaluation metrics, hyperparameter optimization checkpoints, and structured visual figures.

---

## 1. Executive & Technical Reports (Markdown)

| Report Document | Audience / Persona | Key Insights & Content |
|---|---|---|
| [**`MODEL_CARD.md`**](MODEL_CARD.md) | Regulators, ML Engineers, Auditors | Production Model Card (Mitchell et al. format). Documents LightGBM pipeline, Isotonic probability calibration, MAPIE conformal prediction intervals, and EEOC 4/5ths fairness audit. |
| [**`BUSINESS_CASE_STUDY.md`**](BUSINESS_CASE_STUDY.md) | C-Suite, VP of Product, Finance | Unit economics ($C_{FN}=\$500$ vs $C_{FP}=\$50$), cost-optimal decision threshold ($p^*=0.34$), and prescriptive CATE uplift ROI resulting in **$170,140 in net savings**. |
| [**`BENCHMARK_REPORT.md`**](BENCHMARK_REPORT.md) | Data Scientists, Machine Learning Engineers | Rigorous 10-algorithm benchmark comparing LightGBM vs. CatBoost vs. XGBoost vs. FT-Transformer vs. TabPFN with **Nadeau-Bengio corrected repeated t-tests** and latency profiling. |
| [**`DRIFT_MONITORING_REPORT.md`**](DRIFT_MONITORING_REPORT.md) | MLOps Engineers, SREs | Production drift detection report evaluating Population Stability Index (PSI), Chi-square categorical distribution tests, and Wasserstein prediction drift. |
| [**`DATA_GOVERNANCE_REPORT.md`**](DATA_GOVERNANCE_REPORT.md) | Data Engineers, Governance Teams | Data contracts (Pandera schemas), 28-feature engineering lineage, Feast feature store topology, consensus feature selection, and GDPR/CCPA privacy compliance. |

---

## 2. Machine-Readable Metrics (`results/metrics/`)

Structured JSON evaluation results ingested by CI/CD performance gates, monitoring alerts, and automated retraining pipelines:

| File Name | Producing Component | Key Metrics Recorded |
|---|---|---|
| `production_pipeline_results.json` | `pipelines/training_pipeline.py` | Test ROC-AUC (0.8225), PR-AUC (0.6071), Recall (0.9920), Cost-optimal threshold (0.030), Net financial savings ($170,140). |
| `benchmark_results.json` | `scripts/run_benchmarks.py` | 5-Fold CV metrics across LightGBM, CatBoost, XGBoost, FT-Transformer, TabPFN with Nadeau-Bengio corrected t-test p-values. |
| `conformal_results.json` | `scripts/train_conformal.py` | MAPIE coverage (89.28% @ 90% target, 94.46% @ 95% target) and singleton set efficiency rates (70.83%). |
| `uplift_results.json` | `scripts/train_uplift.py` | Causal CATE customer quadrant breakdown: Persuadables (978), Sleeping Dogs (210), Sure Things (1,777), Lost Causes (508). |
| `drift_results.json` | `scripts/generate_drift_report.py` | Feature-level PSI decile metrics, Chi-squared categorical shift values, and Wasserstein prediction drift scores. |
| `retraining_audit.json` | `pipelines/retraining_pipeline.py` | Challenger promotion/rejection decision, Nadeau-Bengio corrected t-test p-value, and statistical rationale. |

---

## 3. Experimental Sweeps & Tuning Checkpoints (`results/tuning/`)

Optimization artifacts produced during algorithmic scanning and Optuna hyperparameter searches:

- `phase1_scan_results.csv`: 10-algorithm baseline comparison across ROC-AUC, F1, Recall, Precision, and Brier score.
- `tuned_best_params.csv`: Optimal hyperparameters discovered by Optuna for LightGBM, CatBoost, XGBoost, and Random Forest.
- `tuning_results.pkl`: Serialized Python dictionary containing complete trial histories, parameter trees, and fold scores.

---

## 4. Visual Figures & Analytics (`results/figures/`)

All generated figures are organized into four dedicated subdirectories:

```
results/figures/
├── eda/                         # Exploratory Data Analysis
│   ├── eda_01_churn_distribution.png
│   ├── eda_02_churn_by_contract.png
│   ├── eda_03_tenure_analysis.png
│   ├── eda_04_monthly_charges.png
│   ├── eda_05_internet_service_churn.png
│   ├── eda_06_service_adoption_churn.png
│   ├── eda_07_contract_internet_heatmap.png
│   ├── eda_08_categorical_churn_rates.png
│   └── eda_09_correlation_matrix.png
├── features/                    # Feature Engineering & Selection Analysis
│   ├── feature_eng_01_mutual_information.png
│   ├── feature_eng_02_rf_importance.png
│   ├── feature_eng_03_raw_vs_engineered.png
│   ├── feature_eng_04_engineered_correlation.png
│   ├── feature_eng_05_feature_lift_analysis.png
│   ├── feat_eng_06_new_binary_lift.png
│   ├── feat_eng_07_numeric_distributions.png
│   ├── feat_eng_08_mi_all_28.png
│   ├── feat_eng_09_selection_results.png
│   ├── feat_eng_10_selection_cv_comparison.png
│   └── feat_eng_11_4way_cv_comparison.png
├── benchmarks/                  # Algorithm Sweeps & Deep Learning Comparisons
│   ├── phase1_algorithm_scan.png
│   ├── phase2_optuna_history.png
│   ├── phase2_tuning_lift.png
│   └── tuned_models_curves.png
└── evaluation/                  # Champion Model Performance & Business Impact
    ├── business_impact.png
    ├── confusion_matrices.png
    ├── cv_vs_test_pr_auc.png
    ├── feature_importance.png
    ├── metrics_heatmap.png
    ├── precision_recall_scatter.png
    └── threshold_sensitivity.png
```

---

## 5. How to Reproduce All Results

```bash
# 1. Run full production training pipeline & persist champion metrics
poetry run python main.py train

# 2. Run multi-algorithm benchmarking & Nadeau-Bengio statistical tests
poetry run python main.py benchmark

# 3. Train prescriptive causal uplift (CATE) model
poetry run python main.py uplift

# 4. Generate conformal prediction uncertainty metrics
poetry run python scripts/train_conformal.py

# 5. Evaluate production population drift against reference baseline
poetry run python main.py monitor

# 6. Launch interactive Streamlit analytics dashboard
poetry run streamlit run dashboards/streamlit_app.py
```
