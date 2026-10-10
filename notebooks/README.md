# Experimental Research Notebooks

This directory contains research, exploratory data analysis, feature engineering experiments, and algorithm scans in standard Jupyter Notebook (`.ipynb`) format.

> [!IMPORTANT]
> **Production Decoupling Note**:  
> **Production training and automated retraining run strictly outside notebooks**. Production flows are fully orchestrated in [`pipelines/training_pipeline.py`](../pipelines/training_pipeline.py) and [`pipelines/retraining_pipeline.py`](../pipelines/retraining_pipeline.py) via Prefect, Feast, and MLflow.
> 
> These notebooks serve as reproducible research artifacts, featuring fully rendered inline visualizations, tables, and narrative commentary for technical review.

---

## Notebook Directory Guide

| Notebook | Focus & Findings | Key Visualizations |
|---|---|---|
| [**`01_eda_and_business_analysis.ipynb`**](01_eda_and_business_analysis.ipynb) | Exploratory Data Analysis & Business Framing | Churn distribution by contract, tenure survival curves, billing friction heatmaps, bivariate categorical lift |
| [**`02_feature_engineering_experiments.ipynb`**](02_feature_engineering_experiments.ipynb) | 28 Domain Features & Consensus Selection | Mutual information ranking, RF Gini importance, VIF collinearity checks, 5-stage selection voting |
| [**`03a_algorithm_scan.ipynb`**](03a_algorithm_scan.ipynb) | 10-Algorithm 5-Fold Cross-Validation Scan | Linear vs Tree vs Ensemble comparison, ROC/PR curves across folds, baseline ranking |
| [**`03b_hyperparameter_tuning.ipynb`**](03b_hyperparameter_tuning.ipynb) | Bayesian Optimization (Optuna) | Hyperparameter search histories, slice plots, tuning lift across LightGBM, CatBoost, XGBoost |
| [**`03c_champion_evaluation.ipynb`**](03c_champion_evaluation.ipynb) | Champion Model Deep Evaluation & Diagnostics | Threshold sensitivity curve, asymmetric cost optimization, SHAP global/local summary, test set verification |

---

## Running and Executing Notebooks

To launch an interactive Jupyter Lab or Notebook server:
```bash
# Launch Jupyter Lab
poetry run jupyter lab notebooks/

# Or launch Jupyter Notebook
poetry run jupyter notebook notebooks/
```

To re-execute any notebook headless from the command line:
```bash
# Execute a notebook inplace with nbconvert
poetry run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_and_business_analysis.ipynb

# Or execute all notebooks via Makefile:
make notebooks-exec
```
