
# %% [markdown]
# # 03b — Hyperparameter Tuning (Phase 2)
#
# **Goal:** Use Optuna's TPE sampler to find the best hyperparameters for each
# top-N candidate identified in 03a. Each study maximises **CV PR-AUC on the
# training set** — the test set is never touched in this notebook.
#
# **Why Optuna?**
# - TPE sampler (Tree-structured Parzen Estimator) is ~3× more sample-efficient
#   than random search and does not require a grid.
# - Each trial calls `build_baseline_pipeline()` — identical to production.
# - `MedianPruner` terminates unpromising trials early, saving compute.
#
# **Pipeline position:**
# ```
# 03a_algorithm_scan → [THIS NOTEBOOK] → 03c_champion_evaluation
# ```
#
# **Inputs:**  `data/processed/phase1_scan_results.csv` (from 03a)
# **Outputs:** `data/processed/tuned_best_params.csv` + pickled pipelines

# %% [markdown]
# ## 0. Setup & Imports

# %%
import os
import sys
import warnings
import pickle
from pathlib import Path
from time import perf_counter

warnings.filterwarnings("ignore")

# ── Project root resolution ───────────────────────────────────────────────────
def find_project_root(marker: str = "pyproject.toml") -> Path:
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not locate project root from {current}")

PROJECT_ROOT = find_project_root()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
print(f"Project root : {PROJECT_ROOT}")

# ── Third-party ───────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# ── Project modules ────────────────────────────────────────────────────────────
from src.data.ingest import load_for_training
from src.data.validate import validate_raw_data
from src.data.preprocess import run_preprocessing
from src.models.pipeline import build_baseline_pipeline
from src.utils.logging import get_logger
from src.utils.config_loader import get_config

logger = get_logger(__name__)
cfg    = get_config()

# ── Global constants ──────────────────────────────────────────────────────────
RANDOM_STATE = cfg.training.random_state
CV_FOLDS     = cfg.training.cv_folds
TEST_SIZE    = cfg.training.test_size

# Number of top Phase-1 models to carry into Optuna tuning
TOP_N_FOR_TUNING = 5

# Optuna trials per model (increase for more exhaustive search)
OPTUNA_N_TRIALS = 100

# ── Plotting ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 120})

print(f"Config  → random_state={RANDOM_STATE}, cv_folds={CV_FOLDS}")
print(f"Tuning  → top_n={TOP_N_FOR_TUNING}, trials_per_model={OPTUNA_N_TRIALS}")

# %% [markdown]
# ## 1. Load Data & Phase 1 Results

# %%
# Load and prepare data identically to 03a
raw_df    = load_for_training()
validated = validate_raw_data(raw_df)
X, y      = run_preprocessing(validated)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"Train : {X_train.shape[0]} rows  |  churn {y_train.mean()*100:.1f}%")
print(f"Test  : {X_test.shape[0]} rows   |  (held out — not used in this notebook)")

# %%
# Load Phase 1 scan results
phase1_path = Path("data/processed/phase1_scan_results.csv")
if not phase1_path.exists():
    raise FileNotFoundError(
        f"Phase 1 results not found at {phase1_path}. "
        "Run 03a_algorithm_scan.py first."
    )

phase1_df = pd.read_csv(phase1_path)
print(f"\nPhase 1 results loaded — {len(phase1_df)} algorithms scored.")
print(phase1_df[["model", "cv_pr_auc_mean"]].to_string(index=False))

# %% [markdown]
# ## 2. Select Top-N Candidates

# %%
tuning_candidates = (
    phase1_df[phase1_df["model"] != "Dummy"]
    .head(TOP_N_FOR_TUNING)["model"]
    .tolist()
)

print(f"\nModels selected for Optuna tuning (top {TOP_N_FOR_TUNING} by CV PR-AUC):")
for rank, name in enumerate(tuning_candidates, 1):
    pr = phase1_df.loc[phase1_df["model"] == name, "cv_pr_auc_mean"].values[0]
    print(f"  {rank}. {name:<26}  CV PR-AUC = {pr:.4f}")

# %% [markdown]
# ## 3. Define Optuna Objective Functions
#
# Each factory returns a closure that Optuna calls once per trial.
# The closure builds a pipeline with trial-suggested params, runs CV,
# and returns mean PR-AUC. All pipelines use `build_baseline_pipeline()`
# for consistency with the production `sklearn.Pipeline`.

# %%
def _cv_pr_auc(pipeline, X, y, cv) -> float:
    """Helper: return mean CV PR-AUC for a fitted pipeline."""
    return cross_val_score(
        pipeline, X, y, cv=cv, scoring="average_precision", n_jobs=1
    ).mean()


def make_objective_lgbm(X_tr, y_tr, cv):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators"    : trial.suggest_int("n_estimators", 200, 1200, step=100),
            "learning_rate"   : trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "num_leaves"      : trial.suggest_int("num_leaves", 20, 255),
            "max_depth"       : trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample"       : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.5, 5.0),
            "random_state"    : RANDOM_STATE,
            "verbose"         : -1,
            "n_jobs"          : -1,
        }
        pipeline = build_baseline_pipeline(LGBMClassifier(**params))
        return _cv_pr_auc(pipeline, X_tr, y_tr, cv)
    return objective


def make_objective_xgb(X_tr, y_tr, cv):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators"    : trial.suggest_int("n_estimators", 200, 1200, step=100),
            "learning_rate"   : trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "max_depth"       : trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample"       : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "gamma"           : trial.suggest_float("gamma", 0.0, 5.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.5, 5.0),
            "eval_metric"     : "logloss",
            "verbosity"       : 0,
            "random_state"    : RANDOM_STATE,
            "n_jobs"          : -1,
        }
        pipeline = build_baseline_pipeline(XGBClassifier(**params))
        return _cv_pr_auc(pipeline, X_tr, y_tr, cv)
    return objective


def make_objective_catboost(X_tr, y_tr, cv):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations"     : trial.suggest_int("iterations", 200, 1200, step=100),
            "learning_rate"  : trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "depth"          : trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg"    : trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "border_count"   : trial.suggest_int("border_count", 32, 255),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
            "random_strength": trial.suggest_float("random_strength", 0.1, 5.0),
            "auto_class_weights": "Balanced",
            "random_seed"    : RANDOM_STATE,
            "verbose"        : 0,
        }
        pipeline = build_baseline_pipeline(CatBoostClassifier(**params))
        return _cv_pr_auc(pipeline, X_tr, y_tr, cv)
    return objective


def make_objective_rf(X_tr, y_tr, cv):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators"  : trial.suggest_int("n_estimators", 100, 800, step=100),
            "max_depth"     : trial.suggest_int("max_depth", 3, 25),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "max_features"  : trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
            "class_weight"  : "balanced_subsample",
            "random_state"  : RANDOM_STATE,
            "n_jobs"        : -1,
        }
        pipeline = build_baseline_pipeline(RandomForestClassifier(**params))
        return _cv_pr_auc(pipeline, X_tr, y_tr, cv)
    return objective


def make_objective_gbm(X_tr, y_tr, cv):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators"  : trial.suggest_int("n_estimators", 100, 800, step=100),
            "learning_rate" : trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "max_depth"     : trial.suggest_int("max_depth", 2, 8),
            "subsample"     : trial.suggest_float("subsample", 0.5, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 80),
            "max_features"  : trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 1.0]),
            "random_state"  : RANDOM_STATE,
        }
        pipeline = build_baseline_pipeline(GradientBoostingClassifier(**params))
        return _cv_pr_auc(pipeline, X_tr, y_tr, cv)
    return objective


def make_objective_lr(X_tr, y_tr, cv):
    def objective(trial: optuna.Trial) -> float:
        penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
        solver  = "saga"  # supports all penalties
        params = {
            "C"            : trial.suggest_float("C", 1e-3, 20.0, log=True),
            "penalty"      : penalty,
            "solver"       : solver,
            "max_iter"     : 3000,
            "class_weight" : "balanced",
            "random_state" : RANDOM_STATE,
        }
        if penalty == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
        pipeline = build_baseline_pipeline(LogisticRegression(**params))
        return _cv_pr_auc(pipeline, X_tr, y_tr, cv)
    return objective


def make_objective_et(X_tr, y_tr, cv):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators"  : trial.suggest_int("n_estimators", 100, 800, step=100),
            "max_depth"     : trial.suggest_int("max_depth", 3, 25),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 40),
            "max_features"  : trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
            "class_weight"  : "balanced_subsample",
            "random_state"  : RANDOM_STATE,
            "n_jobs"        : -1,
        }
        pipeline = build_baseline_pipeline(ExtraTreesClassifier(**params))
        return _cv_pr_auc(pipeline, X_tr, y_tr, cv)
    return objective


def make_objective_svc(X_tr, y_tr, cv):
    def objective(trial: optuna.Trial) -> float:
        kernel = trial.suggest_categorical("kernel", ["rbf", "poly"])
        params = {
            "kernel"       : kernel,
            "C"            : trial.suggest_float("C", 0.01, 100.0, log=True),
            "gamma"        : trial.suggest_categorical("gamma", ["scale", "auto"]),
            "class_weight" : "balanced",
            "random_state" : RANDOM_STATE,
        }
        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 4)
        est = CalibratedClassifierCV(
            SVC(**params), method="isotonic", cv=3
        )
        pipeline = build_baseline_pipeline(est)
        return _cv_pr_auc(pipeline, X_tr, y_tr, cv)
    return objective


# Map model name → objective factory
OBJECTIVE_FACTORIES = {
    "LightGBM"           : make_objective_lgbm,
    "XGBoost"            : make_objective_xgb,
    "CatBoost"           : make_objective_catboost,
    "Random Forest"      : make_objective_rf,
    "Extra Trees"        : make_objective_et,
    "Gradient Boosting"  : make_objective_gbm,
    "Logistic Regression": make_objective_lr,
    "SVC (calibrated)"   : make_objective_svc,
}

# %% [markdown]
# ## 4. Run Optuna Studies

# %%
cv_splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
tuning_results: dict[str, dict] = {}

for name in tuning_candidates:
    if name not in OBJECTIVE_FACTORIES:
        print(f"  ⚠️  No Optuna objective defined for '{name}' — skipping.")
        continue

    print(f"\n  {'─'*55}")
    print(f"  Tuning: {name}  ({OPTUNA_N_TRIALS} trials)")
    print(f"  {'─'*55}")

    sampler = TPESampler(seed=RANDOM_STATE, n_startup_trials=10)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study   = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"churn_{name.replace(' ', '_').lower()}",
    )

    objective_fn = OBJECTIVE_FACTORIES[name](X_train, y_train, cv_splitter)
    t0 = perf_counter()
    study.optimize(objective_fn, n_trials=OPTUNA_N_TRIALS, show_progress_bar=True)
    elapsed = perf_counter() - t0

    best_val  = study.best_value
    best_params = study.best_params

    tuning_results[name] = {
        "best_cv_pr_auc" : round(best_val, 4),
        "best_params"    : best_params,
        "study"          : study,
        "tuning_time_s"  : round(elapsed, 1),
        "n_trials"       : len(study.trials),
    }

    # Scan PR-AUC for comparison
    scan_pr = phase1_df.loc[phase1_df["model"] == name, "cv_pr_auc_mean"].values[0]
    lift    = best_val - scan_pr

    print(f"  ✅  Best CV PR-AUC : {best_val:.4f}  "
          f"(scan={scan_pr:.4f}, lift={lift:+.4f}, {elapsed:.0f}s)")
    print(f"  Best params: {best_params}")

# %% [markdown]
# ## 5. Tuning Lift Summary
#
# Compare the PR-AUC improvement from Optuna tuning vs Phase 1 defaults.

# %%
lift_rows = []
for name in tuning_candidates:
    if name not in tuning_results:
        continue
    scan_pr = phase1_df.loc[phase1_df["model"] == name, "cv_pr_auc_mean"].values[0]
    tuned   = tuning_results[name]["best_cv_pr_auc"]
    lift_rows.append({
        "model"          : name,
        "scan_cv_pr_auc" : scan_pr,
        "tuned_cv_pr_auc": tuned,
        "lift"           : round(tuned - scan_pr, 4),
        "tuning_time_s"  : tuning_results[name]["tuning_time_s"],
    })

lift_df = pd.DataFrame(lift_rows).sort_values("tuned_cv_pr_auc", ascending=False).reset_index(drop=True)
lift_df.index += 1

print("\n── Optuna Tuning Lift Summary ──────────────────────────────────────")
print(lift_df.to_string(float_format="{:.4f}".format))

# %%
fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(lift_df))
w = 0.35
bars1 = ax.bar(x - w/2, lift_df["scan_cv_pr_auc"],  width=w, label="Phase 1 Scan",     alpha=0.75, color="#5b8db8")
bars2 = ax.bar(x + w/2, lift_df["tuned_cv_pr_auc"], width=w, label="Optuna Tuned",      alpha=0.90, color="#e05c5c")

for bar, lift in zip(bars2, lift_df["lift"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{lift:+.3f}", ha="center", va="bottom", fontsize=8, color="#d62728", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(lift_df["model"], rotation=18, ha="right")
ax.set_ylabel("CV PR-AUC")
ax.set_ylim(max(0, lift_df["scan_cv_pr_auc"].min() - 0.05),
            min(1.0, lift_df["tuned_cv_pr_auc"].max() + 0.04))
ax.legend()
ax.set_title("Hyperparameter Tuning Lift — Scan vs. Optuna Best  (CV PR-AUC)", fontweight="bold")
plt.tight_layout()
plt.savefig("reports/phase2_tuning_lift.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Optimization History (Top-3 Candidates)

# %%
top3_tuned = lift_df["model"].head(3).tolist()

fig, axes = plt.subplots(1, len(top3_tuned), figsize=(5 * len(top3_tuned), 4))
if len(top3_tuned) == 1:
    axes = [axes]

for ax, name in zip(axes, top3_tuned):
    study  = tuning_results[name]["study"]
    trials = study.trials_dataframe()
    ax.scatter(trials.index, trials["value"], alpha=0.4, s=12, color="#5b8db8")
    running_best = trials["value"].cummax()
    ax.plot(trials.index, running_best, color="#e05c5c", lw=2, label="Running best")
    ax.axhline(study.best_value, color="green", linestyle="--", lw=1.2,
               label=f"Best={study.best_value:.4f}")
    ax.set_title(f"{name}", fontweight="bold", fontsize=10)
    ax.set_xlabel("Trial")
    ax.set_ylabel("CV PR-AUC")
    ax.legend(fontsize=8)

plt.suptitle("Optuna Optimization History — Top 3 Candidates", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("reports/phase2_optuna_history.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Save Tuning Artifacts for 03c
#
# Save two things:
# 1. **Best params CSV** — lightweight, human-readable summary
# 2. **Fitted pipelines pickle** — so 03c can skip refitting if desired

# %%
# 7a. Save best params as CSV
params_rows = []
for name in tuning_candidates:
    if name not in tuning_results:
        continue
    params_rows.append({
        "model"          : name,
        "best_cv_pr_auc" : tuning_results[name]["best_cv_pr_auc"],
        "tuning_time_s"  : tuning_results[name]["tuning_time_s"],
        "n_trials"       : tuning_results[name]["n_trials"],
        "best_params"    : str(tuning_results[name]["best_params"]),
    })

params_df = pd.DataFrame(params_rows).sort_values("best_cv_pr_auc", ascending=False)
params_path = Path("data/processed/tuned_best_params.csv")
params_df.to_csv(params_path, index=False)
print(f"✅  Best params saved to {params_path}")

# %%
# 7b. Save tuning results (params only, not studies) as pickle
tuning_export = {}
for name, result in tuning_results.items():
    tuning_export[name] = {
        "best_cv_pr_auc": result["best_cv_pr_auc"],
        "best_params"   : result["best_params"],
        "tuning_time_s" : result["tuning_time_s"],
        "n_trials"      : result["n_trials"],
    }

pickle_path = Path("data/processed/tuning_results.pkl")
with open(pickle_path, "wb") as f:
    pickle.dump(tuning_export, f)
print(f"✅  Tuning results pickle saved to {pickle_path}")

# %%
print(f"\n{'='*60}")
print(f"  Phase 2 complete — {len(tuning_results)} models tuned")
print(f"  Top model: {lift_df.iloc[0]['model']}  "
      f"(CV PR-AUC = {lift_df.iloc[0]['tuned_cv_pr_auc']:.4f})")
print(f"{'='*60}")
print(f"\n  → Next: run 03c_champion_evaluation.py")
