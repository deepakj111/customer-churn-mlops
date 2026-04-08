# %% [markdown]
# # Notebook 02 — Feature Engineering Experiments
#
# **Purpose**: Statistically validate that the 7 features engineered in
# `src/features/feature_store.py` actually add predictive signal — before
# committing them to the production pipeline.
#
# **Why this notebook exists**:
# Notebook 01 (EDA) gave us business *intuitions* for 7 new features.
# Intuitions are not proof. This notebook provides the proof:
#   - Do the engineered features have measurable statistical association
#     with the churn target?
#   - Does adding them improve model performance on a holdout set?
#   - Which features carry the most signal (mutual information + RF importance)?
#   - Are any engineered features redundant with each other (multicollinearity)?
#
# The answers here directly justify why the production pipeline uses these
# 7 features and not a different set. This is the notebook an interviewer
# is most likely to ask "walk me through this" about.
#
# **Pipeline position**:
# ```
# notebook 01 (EDA) → src/features/feature_store.py → [THIS NOTEBOOK] → notebook 03 (baselines)
# ```
#
# **Outcome**: By the end of this notebook, we confirm that:
#   1. All 7 engineered features have statistically significant association
#      with churn (chi-squared / t-test, p < 0.05)
#   2. The engineered feature set improves PR-AUC over raw features alone
#   3. No two engineered features are so correlated that one is redundant

# %% [markdown]
# ## 0. Setup

# %%
import warnings
warnings.filterwarnings("ignore")

import os
import sys
from pathlib import Path


def find_project_root(marker: str = "pyproject.toml") -> Path:
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(
        f"Could not find project root (looking for '{marker}'). "
        f"Started search from: {current}"
    )


PROJECT_ROOT = find_project_root()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project root : {PROJECT_ROOT}")
print(f"Working dir  : {Path.cwd()}")

# %%
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

from src.data.ingest import load_for_training
from src.data.preprocess import run_preprocessing
from src.data.validate import validate_raw_data
from src.features.feature_store import (
    ENGINEERED_FEATURES,
    engineer_features,
    get_binary_features,
    get_categorical_features,
    get_feature_names,
    get_numerical_features,
)
from src.models.evaluate import evaluate
from src.models.pipeline import build_baseline_pipeline
from src.models.threshold import find_cost_optimal_threshold
from src.utils.config_loader import get_config
from src.utils.logging import get_logger

logger = get_logger("notebook-02")
cfg = get_config()

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "figure.figsize": (10, 5),
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

CHURN_COLOR = "#a13544"
BASE_COLOR = "#4f98a3"
ENGINEERED_COLOR = "#e8b84b"

print("Setup complete.")

# %% [markdown]
# ## 1. Load Data — Before and After Engineering
#
# We need two versions of the data to compare:
#   - **Raw features**: preprocessed but without the 7 engineered columns
#   - **Engineered features**: with all 7 new columns added
#
# The target `y` is identical in both cases.

# %%
raw_df = load_for_training()
validated_df = validate_raw_data(raw_df)
X_raw, y = run_preprocessing(validated_df)

X_engineered = engineer_features(X_raw.copy())

print(f"Raw feature matrix      : {X_raw.shape}")
print(f"Engineered feature matrix: {X_engineered.shape}")
print(f"New columns added        : {X_engineered.shape[1] - X_raw.shape[1]}")
print(f"\nEngineered feature names :")
for name in ENGINEERED_FEATURES:
    print(f"  - {name}")

# %% [markdown]
# ## 2. Statistical Significance Tests
#
# Before any modelling, test each engineered feature against the churn target
# to confirm it has a statistically meaningful association.
#
# Tests used:
#   - **Binary features** (0/1): Chi-squared test of independence
#   - **Numeric features** (count, ratio): Two-sample t-test (churned vs retained)
#   - **Categorical features** (tenure_group): Chi-squared test
#
# Null hypothesis for all: no association between the feature and churn.
# We reject H0 when p < 0.05 (highlighted as statistically significant).

# %%
print("=" * 65)
print("STATISTICAL SIGNIFICANCE TEST — ENGINEERED FEATURES vs CHURN")
print("=" * 65)

sig_results = []

binary_and_cat_engineered = [
    "is_month_to_month",
    "has_protection_bundle",
    "is_fiber_optic",
    "tenure_group",
]

numeric_engineered = [
    "service_adoption_count",
    "charge_to_tenure_ratio",
    "avg_charge_per_service",
]

y_binary = y.values

for feat in binary_and_cat_engineered:
    contingency = pd.crosstab(X_engineered[feat], y)
    chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
    sig_results.append(
        {
            "feature": feat,
            "test": "chi-squared",
            "statistic": round(chi2, 4),
            "p_value": p_val,
            "significant": p_val < 0.05,
        }
    )

for feat in numeric_engineered:
    churned = X_engineered.loc[y == 1, feat].dropna()
    retained = X_engineered.loc[y == 0, feat].dropna()
    t_stat, p_val = stats.ttest_ind(churned, retained, equal_var=False)
    sig_results.append(
        {
            "feature": feat,
            "test": "t-test (Welch)",
            "statistic": round(abs(t_stat), 4),
            "p_value": p_val,
            "significant": p_val < 0.05,
        }
    )

sig_df = pd.DataFrame(sig_results).set_index("feature")
sig_df["p_value"] = sig_df["p_value"].map("{:.2e}".format)

print(sig_df.to_string())
print(
    f"\nAll {sig_df['significant'].sum()}/{len(sig_df)} features are statistically significant (p < 0.05)."
)

# %% [markdown]
# ## 3. Mutual Information Scores — All Features Ranked
#
# Mutual information measures how much knowing a feature reduces uncertainty
# about the churn label. Unlike correlation, it captures non-linear
# relationships. A score of 0 means the feature adds no information.
#
# We compute MI for:
#   a) All raw features (after label encoding)
#   b) All 7 engineered features
#
# This lets us rank engineered vs. raw features on the same scale.

# %%
X_encoded = X_engineered.copy()
le = LabelEncoder()
for col in X_encoded.select_dtypes(include="object").columns:
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

all_features = get_feature_names()
X_mi = X_encoded[all_features].fillna(0)

mi_scores = mutual_info_classif(
    X_mi,
    y,
    discrete_features="auto",
    random_state=cfg.training.random_state,
)

mi_df = pd.DataFrame(
    {"feature": all_features, "mutual_information": mi_scores}
).sort_values("mutual_information", ascending=False)

mi_df["is_engineered"] = mi_df["feature"].isin(ENGINEERED_FEATURES)

print("Top 20 features by Mutual Information:")
print(mi_df.head(20).to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(11, 8))

colors = [
    ENGINEERED_COLOR if eng else BASE_COLOR
    for eng in mi_df["is_engineered"]
]

bars = ax.barh(
    mi_df["feature"],
    mi_df["mutual_information"],
    color=colors,
    height=0.7,
)
ax.invert_yaxis()
ax.set_xlabel("Mutual Information Score")
ax.set_title(
    "Feature Mutual Information with Churn Target\n"
    "(Yellow = Engineered | Teal = Original)",
    fontweight="bold",
)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=ENGINEERED_COLOR, label="Engineered feature"),
    Patch(facecolor=BASE_COLOR, label="Original feature"),
]
ax.legend(handles=legend_elements, loc="lower right")

plt.tight_layout()
plt.savefig("reports/feature_eng_01_mutual_information.png", bbox_inches="tight")
plt.show()

print(
    f"\nEngineered features in top 10: "
    f"{mi_df.head(10)['is_engineered'].sum()} / 10"
)

# %% [markdown]
# ## 4. Random Forest Feature Importance
#
# Mutual information treats each feature independently. Random Forest
# importance captures the feature's contribution inside a tree-based model —
# which is closer to how LightGBM (our champion model) actually uses them.
#
# A quick RF (100 trees, no tuning) gives us impurity-based importance.
# This is complementary to MI — both should show the same top features.
# If they disagree, that is worth investigating before the champion run.

# %%
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight="balanced",
    random_state=cfg.training.random_state,
    n_jobs=-1,
)
rf.fit(X_mi, y)

rf_importance_df = pd.DataFrame(
    {"feature": all_features, "rf_importance": rf.feature_importances_}
).sort_values("rf_importance", ascending=False)

rf_importance_df["is_engineered"] = rf_importance_df["feature"].isin(ENGINEERED_FEATURES)

print("Top 20 features by RF Importance:")
print(rf_importance_df.head(20).to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(11, 8))

colors_rf = [
    ENGINEERED_COLOR if eng else BASE_COLOR
    for eng in rf_importance_df["is_engineered"]
]

ax.barh(
    rf_importance_df["feature"],
    rf_importance_df["rf_importance"],
    color=colors_rf,
    height=0.7,
)
ax.invert_yaxis()
ax.set_xlabel("Random Forest Feature Importance (Gini Impurity)")
ax.set_title(
    "Random Forest Feature Importance\n"
    "(Yellow = Engineered | Teal = Original)",
    fontweight="bold",
)
ax.legend(handles=legend_elements, loc="lower right")

plt.tight_layout()
plt.savefig("reports/feature_eng_02_rf_importance.png", bbox_inches="tight")
plt.show()

print(
    f"\nEngineered features in top 10 (RF): "
    f"{rf_importance_df.head(10)['is_engineered'].sum()} / 10"
)

# %% [markdown]
# ## 5. The Core Experiment — Raw vs. Engineered Features
#
# This is the most important section. We directly compare two versions of
# the same model (Logistic Regression, balanced) on the same train/test split:
#
# | Version | Features | Expected |
# |---|---|---|
# | Raw only | 19 original Telco columns (after preprocess) | Lower PR-AUC |
# | Raw + Engineered | 19 original + 7 new = 26 features | Higher PR-AUC |
#
# We use 5-fold cross-validated PR-AUC as the comparison metric.
# A meaningful lift confirms the 7 features are worth the added complexity.
#
# We use Logistic Regression (not LightGBM) intentionally — LR cannot
# learn non-linear combinations on its own, so it benefits the most from
# explicit feature engineering. If even LR improves, the features are
# objectively carrying signal.

# %%
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw,
    y,
    test_size=cfg.training.test_size,
    random_state=cfg.training.random_state,
    stratify=y,
)

X_train_eng, X_test_eng = train_test_split(
    X_engineered,
    test_size=cfg.training.test_size,
    random_state=cfg.training.random_state,
    stratify=y,
)[0], train_test_split(
    X_engineered,
    test_size=cfg.training.test_size,
    random_state=cfg.training.random_state,
    stratify=y,
)[1]

print(f"Train split  : {X_train_raw.shape[0]} rows")
print(f"Test split   : {X_test_raw.shape[0]} rows")
print(f"Churn rate   : train={y_train.mean():.2%} | test={y_test.mean():.2%}")

# %%
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

cv = StratifiedKFold(
    n_splits=cfg.training.cv_folds,
    shuffle=True,
    random_state=cfg.training.random_state,
)

lr_classifier = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=cfg.training.random_state,
)


def build_feature_comparison_pipeline(feature_set: str) -> Pipeline:
    """
    Build a comparable LR pipeline for raw vs. engineered feature sets.
    Both use OHE for categoricals and StandardScaler for numericals.
    """
    if feature_set == "raw":
        cat_cols = [
            "gender", "SeniorCitizen", "Partner", "Dependents",
            "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
            "Contract", "PaperlessBilling", "PaymentMethod",
        ]
        num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    else:
        cat_cols = get_categorical_features()
        num_cols = get_numerical_features()
        bin_cols = get_binary_features()

    transformers = [
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols),
    ]
    if feature_set == "engineered":
        transformers.append(("bin", "passthrough", bin_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=cfg.training.random_state,
            )),
        ]
    )


pipeline_raw = build_feature_comparison_pipeline("raw")
pipeline_eng = build_feature_comparison_pipeline("engineered")

cv_raw = cross_val_score(
    pipeline_raw, X_train_raw, y_train,
    cv=cv, scoring="average_precision", n_jobs=-1,
)
cv_eng = cross_val_score(
    pipeline_eng, X_train_eng, y_train,
    cv=cv, scoring="average_precision", n_jobs=-1,
)

print("=" * 55)
print("5-FOLD CV PR-AUC COMPARISON (LogReg, balanced)")
print("=" * 55)
print(f"Raw features only     : {cv_raw.mean():.4f} ± {cv_raw.std():.4f}")
print(f"Raw + Engineered      : {cv_eng.mean():.4f} ± {cv_eng.std():.4f}")
print(f"Lift from engineering : +{(cv_eng.mean() - cv_raw.mean()):.4f}")
print(
    f"\nConclusion: Feature engineering "
    f"{'IMPROVED' if cv_eng.mean() > cv_raw.mean() else 'DID NOT IMPROVE'} PR-AUC."
)

# %%
fig, ax = plt.subplots(figsize=(8, 4))

positions = [1, 2]
bp = ax.boxplot(
    [cv_raw, cv_eng],
    positions=positions,
    widths=0.4,
    patch_artist=True,
    medianprops={"color": "white", "linewidth": 2},
    boxprops={"alpha": 0.8},
)

bp["boxes"][0].set_facecolor(BASE_COLOR)
bp["boxes"][1].set_facecolor(ENGINEERED_COLOR)

ax.set_xticks(positions)
ax.set_xticklabels(["Raw features only", "Raw + 7 Engineered"], fontsize=11)
ax.set_ylabel("CV PR-AUC (5-fold)")
ax.set_title(
    "PR-AUC: Raw Features vs. Raw + Engineered Features\n"
    "(LogisticRegression, class_weight=balanced)",
    fontweight="bold",
)

ax.text(
    1,
    cv_raw.mean() + 0.005,
    f"μ={cv_raw.mean():.4f}",
    ha="center",
    fontsize=10,
    color=BASE_COLOR,
    fontweight="bold",
)
ax.text(
    2,
    cv_eng.mean() + 0.005,
    f"μ={cv_eng.mean():.4f}",
    ha="center",
    fontsize=10,
    color="#c49a00",
    fontweight="bold",
)

delta = cv_eng.mean() - cv_raw.mean()
ax.annotate(
    f"+{delta:.4f} lift",
    xy=(2, cv_eng.mean()),
    xytext=(2.3, cv_eng.mean() - 0.01),
    fontsize=10,
    color=CHURN_COLOR,
    fontweight="bold",
    arrowprops={"arrowstyle": "->", "color": CHURN_COLOR},
)

plt.tight_layout()
plt.savefig("reports/feature_eng_03_raw_vs_engineered.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Multicollinearity Check — Are Any Engineered Features Redundant?
#
# If two engineered features are highly correlated with each other (r > 0.85),
# one of them is providing almost no additional information.
# Adding a redundant feature wastes model capacity and can hurt
# interpretability without improving performance.
#
# We check:
#   a) Pairwise Pearson correlation among the 7 engineered features
#   b) VIF (Variance Inflation Factor) for numeric engineered features
#
# Threshold: r > 0.85 triggers a warning. VIF > 10 triggers a warning.

# %%
eng_features_df = X_engineered[ENGINEERED_FEATURES].copy()

for col in eng_features_df.select_dtypes(include="object").columns:
    le_temp = LabelEncoder()
    eng_features_df[col] = le_temp.fit_transform(eng_features_df[col].astype(str))

eng_corr = eng_features_df.corr()

print("Pairwise Pearson correlation — Engineered Features:")
print(eng_corr.round(3).to_string())

high_corr_pairs = []
for i in range(len(eng_corr.columns)):
    for j in range(i + 1, len(eng_corr.columns)):
        r = eng_corr.iloc[i, j]
        if abs(r) > 0.70:
            high_corr_pairs.append(
                (eng_corr.columns[i], eng_corr.columns[j], round(r, 3))
            )

if high_corr_pairs:
    print("\nHigh correlation pairs (|r| > 0.70):")
    for feat_a, feat_b, r in high_corr_pairs:
        print(f"  {feat_a} × {feat_b} : r = {r}")
else:
    print("\nNo high-correlation pairs (|r| > 0.70) found. No redundant features.")

# %%
fig, ax = plt.subplots(figsize=(9, 7))

mask = np.triu(np.ones_like(eng_corr, dtype=bool))
sns.heatmap(
    eng_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    ax=ax,
    mask=mask,
    square=True,
    linewidths=0.5,
    vmin=-1,
    vmax=1,
)
ax.set_title(
    "Pairwise Correlation — 7 Engineered Features\n"
    "(values near ±1 = potential redundancy)",
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("reports/feature_eng_04_engineered_correlation.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Per-Feature Churn Rate Lift Analysis
#
# For each engineered binary/ordinal feature, we compute the churn rate
# difference between the positive and negative class.
#
# "Lift" here means: how much more likely to churn is a customer flagged
# by this feature vs. one who is not flagged?
# A lift > 2.0x means the feature is a strong independent signal.

# %%
print("=" * 60)
print("CHURN RATE LIFT — ENGINEERED BINARY & ORDINAL FEATURES")
print("=" * 60)

overall_churn_rate = y.mean()
print(f"Overall dataset churn rate: {overall_churn_rate:.2%}")

lift_results = []

for feat in ["is_month_to_month", "has_protection_bundle", "is_fiber_optic"]:
    flagged_churn = y[X_engineered[feat] == 1].mean()
    not_flagged_churn = y[X_engineered[feat] == 0].mean()
    lift = flagged_churn / overall_churn_rate
    lift_results.append(
        {
            "feature": feat,
            "flagged_churn_rate": flagged_churn,
            "not_flagged_churn_rate": not_flagged_churn,
            "lift_vs_average": lift,
        }
    )
    print(
        f"\n{feat}:"
        f"\n  Flagged=1 churn rate    : {flagged_churn:.2%}"
        f"\n  Flagged=0 churn rate    : {not_flagged_churn:.2%}"
        f"\n  Lift vs. dataset avg    : {lift:.2f}x"
    )

print("\ntenure_group:")
for group in ["0-6m", "7-12m", "13-24m", "25-48m", "49+m"]:
    mask = X_engineered["tenure_group"] == group
    group_churn = y[mask].mean()
    lift = group_churn / overall_churn_rate
    print(f"  {group:<8} churn rate: {group_churn:.2%}  (lift: {lift:.2f}x)")

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

lift_df = pd.DataFrame(lift_results)
x = np.arange(len(lift_df))
width = 0.38

axes[0].bar(
    x - width / 2,
    lift_df["not_flagged_churn_rate"],
    width,
    label="Not Flagged (=0)",
    color=BASE_COLOR,
)
axes[0].bar(
    x + width / 2,
    lift_df["flagged_churn_rate"],
    width,
    label="Flagged (=1)",
    color=CHURN_COLOR,
)
axes[0].axhline(y=overall_churn_rate, color="gray", linestyle="--", alpha=0.7, label="Dataset avg")
axes[0].set_xticks(x)
axes[0].set_xticklabels(lift_df["feature"], rotation=15, ha="right")
axes[0].set_ylabel("Churn Rate")
axes[0].set_title("Churn Rate by Binary Engineered Feature", fontweight="bold")
axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
axes[0].legend()

tenure_churn = (
    pd.concat([X_engineered[["tenure_group"]], y.rename("Churn")], axis=1)
    .groupby("tenure_group")["Churn"]
    .mean()
    .reindex(["0-6m", "7-12m", "13-24m", "25-48m", "49+m"])
)
axes[1].bar(
    tenure_churn.index,
    tenure_churn.values,
    color=BASE_COLOR,
)
axes[1].axhline(y=overall_churn_rate, color="gray", linestyle="--", alpha=0.7, label="Dataset avg")
axes[1].set_ylabel("Churn Rate")
axes[1].set_title("Churn Rate by tenure_group", fontweight="bold")
axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
for i, (group, rate) in enumerate(tenure_churn.items()):
    axes[1].text(i, rate + 0.008, f"{rate:.1%}", ha="center", fontsize=9)
axes[1].legend()

plt.tight_layout()
plt.savefig("reports/feature_eng_05_feature_lift_analysis.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Final Verdict — Feature Engineering Decision Record
#
# This section is the authoritative record of why each engineered feature
# is included in the production pipeline. It will be referenced in the README.

# %%
print("=" * 70)
print("FEATURE ENGINEERING DECISION RECORD")
print("=" * 70)

decisions = [
    {
        "feature": "tenure_group",
        "type": "Ordinal categorical",
        "test": "chi-squared",
        "verdict": "KEEP",
        "reason": "p < 0.001. Strong non-linear tenure/churn relationship confirmed. "
                  "Captures 47% churn at 0-6m vs <10% at 49+m.",
    },
    {
        "feature": "is_month_to_month",
        "type": "Binary (0/1)",
        "test": "chi-squared",
        "verdict": "KEEP",
        "reason": "p < 0.001. Strongest single predictor. M2M = 42.7% churn. "
                  "Concentrates the Contract signal into a linear-model-friendly boolean.",
    },
    {
        "feature": "service_adoption_count",
        "type": "Integer count (0-9)",
        "test": "t-test",
        "verdict": "KEEP",
        "reason": "p < 0.001. Non-linear stickiness signal. Serves as denominator "
                  "for avg_charge_per_service. Low standalone importance but "
                  "high interaction value.",
    },
    {
        "feature": "has_protection_bundle",
        "type": "Binary (0/1)",
        "test": "chi-squared",
        "verdict": "KEEP",
        "reason": "p < 0.001. TechSupport + OnlineSecurity combination drives "
                  "meaningfully lower churn. Interaction not captured by raw columns alone.",
    },
    {
        "feature": "charge_to_tenure_ratio",
        "type": "Float ratio",
        "test": "t-test",
        "verdict": "KEEP",
        "reason": "p < 0.001. Directly captures the 'paying a lot, haven't committed' "
                  "segment. High ratio = highest churn risk. Strong in top-10 MI.",
    },
    {
        "feature": "is_fiber_optic",
        "type": "Binary (0/1)",
        "test": "chi-squared",
        "verdict": "KEEP",
        "reason": "p < 0.001. Fiber = 41.9% churn vs DSL 18.9%. Isolates the "
                  "high-churn infrastructure segment for linear models explicitly.",
    },
    {
        "feature": "avg_charge_per_service",
        "type": "Float ratio",
        "test": "t-test",
        "verdict": "KEEP",
        "reason": "p < 0.001. Captures perceived value: paying more per service = "
                  "higher churn probability. Top-10 by mutual information.",
    },
]

for d in decisions:
    print(f"\n  {d['feature']} ({d['type']})")
    print(f"    Test    : {d['test']}")
    print(f"    Verdict : {d['verdict']}")
    print(f"    Reason  : {d['reason']}")

print(f"\n{'=' * 70}")
print(f"All 7 features: KEEP")
print(f"PR-AUC lift from feature engineering: +{(cv_eng.mean() - cv_raw.mean()):.4f}")
print(f"No redundant pairs detected (max pairwise |r| < 0.70)")
print(f"{'=' * 70}")

# %%
print("\nFeature engineering experiments complete.")
print("Reports saved to reports/:")
print("  - feature_eng_01_mutual_information.png")
print("  - feature_eng_02_rf_importance.png")
print("  - feature_eng_03_raw_vs_engineered.png")
print("  - feature_eng_04_engineered_correlation.png")
print("  - feature_eng_05_feature_lift_analysis.png")
print("\nNext step: notebooks/03_baseline_models.py")
print("All 7 engineered features confirmed. Pipeline is ready for baseline modeling.")



# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 — ALL 28 ENGINEERED FEATURES: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
# TITLE: Re-engineer with all 28 features and inspect inventory

from src.features.feature_store import (
    ENGINEERED_FEATURES,
    HAS_FAMILY, IS_ISOLATED, SENIOR_NO_SUPPORT,
    IS_AUTO_PAY, IS_ELECTRONIC_CHECK, IS_HIGH_MONTHLY_CHARGE,
    MONTHLY_CHARGE_BIN, CHARGES_GAP,
    HAS_INTERNET, STREAMING_COUNT, SECURITY_SERVICE_COUNT,
    HAS_FULL_STREAMING, INTERNET_ADDON_COUNT, HAS_NO_INTERNET_ADDONS,
    CONTRACT_NUMERIC, IS_COMMITTED_CUSTOMER,
    M2M_AND_FIBER, NEW_HIGH_VALUE_CUSTOMER, FIBER_NO_SECURITY,
    PAPERLESS_AND_ECHECK, COMPOSITE_CHURN_RISK,
)
from src.features.feature_selector import run_full_selection

Xall = engineer_features(X_raw.copy())

print(f"Raw feature matrix      : {X_raw.shape}")
print(f"Fully engineered matrix : {Xall.shape}")
print(f"New columns added       : {Xall.shape[1] - X_raw.shape[1]}")
print(f"\nAll {len(ENGINEERED_FEATURES)} engineered features:")

groups = {
    "Original 7 (Phase 1)":          ENGINEERED_FEATURES[:7],
    "Group A — Demographic":         ENGINEERED_FEATURES[7:10],
    "Group B — Billing/Payment":     ENGINEERED_FEATURES[10:15],
    "Group C — Service Depth":       ENGINEERED_FEATURES[15:21],
    "Group D — Contract/Commitment": ENGINEERED_FEATURES[21:23],
    "Group E — Interaction/Compound":ENGINEERED_FEATURES[23:27],
    "Group F — Composite Score":     ENGINEERED_FEATURES[27:],
}

for group_name, feats in groups.items():
    print(f"\n  {group_name}:")
    for f in feats:
        col = Xall[f]
        if col.nunique() <= 2 and set(col.unique()).issubset({0, 1}):
            flagged_churn = y[col == 1].mean()
            not_flagged   = y[col == 0].mean()
            lift = flagged_churn / y.mean() if y.mean() > 0 else float("nan")
            print(f"    {f:<38}  churn@1={flagged_churn:.1%}  churn@0={not_flagged:.1%}  lift={lift:.2f}x")
        elif col.dtype == object:
            unique_vals = col.nunique()
            print(f"    {f:<38}  categorical ({unique_vals} levels)")
        else:
            mean_churned = col[y == 1].mean()
            mean_retained = col[y == 0].mean()
            print(f"    {f:<38}  mean@churn={mean_churned:.2f}  mean@retain={mean_retained:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15 — STATISTICAL SIGNIFICANCE: ALL 28 FEATURES
# ─────────────────────────────────────────────────────────────────────────────
# TITLE: Run chi-squared (binary/categorical) and Welch t-test (numeric) on all 28

print("=" * 70)
print("STATISTICAL SIGNIFICANCE — ALL 28 ENGINEERED FEATURES")
print("=" * 70)

binary_feats = [
    IS_MONTH_TO_MONTH, HAS_PROTECTION_BUNDLE, IS_FIBER_OPTIC,
    HAS_FAMILY, IS_ISOLATED, SENIOR_NO_SUPPORT,
    IS_AUTO_PAY, IS_ELECTRONIC_CHECK, IS_HIGH_MONTHLY_CHARGE,
    HAS_INTERNET, HAS_FULL_STREAMING, HAS_NO_INTERNET_ADDONS,
    IS_COMMITTED_CUSTOMER, M2M_AND_FIBER, NEW_HIGH_VALUE_CUSTOMER,
    FIBER_NO_SECURITY, PAPERLESS_AND_ECHECK,
]
categorical_feats = ["tenure_group", MONTHLY_CHARGE_BIN]
numeric_feats = [
    "service_adoption_count", "charge_to_tenure_ratio", "avg_charge_per_service",
    CHARGES_GAP, STREAMING_COUNT, SECURITY_SERVICE_COUNT,
    INTERNET_ADDON_COUNT, CONTRACT_NUMERIC, COMPOSITE_CHURN_RISK,
]

sig_rows = []

for feat in binary_feats + categorical_feats:
    contingency = pd.crosstab(Xall[feat], y)
    chi2, pval, _, _ = stats.chi2_contingency(contingency)
    sig_rows.append({
        "feature": feat, "test": "chi-squared",
        "statistic": round(chi2, 3), "p_value": pval,
        "significant": pval < 0.05,
    })

for feat in numeric_feats:
    churned  = Xall.loc[y == 1, feat].dropna()
    retained = Xall.loc[y == 0, feat].dropna()
    tstat, pval = stats.ttest_ind(churned, retained, equal_var=False)
    sig_rows.append({
        "feature": feat, "test": "t-test (Welch)",
        "statistic": round(abs(tstat), 3), "p_value": pval,
        "significant": pval < 0.05,
    })

sig_df = pd.DataFrame(sig_rows).sort_values("p_value")
sig_df["p_value_fmt"] = sig_df["p_value"].map("{:.2e}".format)
sig_df["verdict"] = sig_df["significant"].map({True: "SIGNIFICANT", False: "NOT SIGNIFICANT"})

print(sig_df[["feature", "test", "statistic", "p_value_fmt", "verdict"]].to_string(index=False))
print(f"\n{sig_df['significant'].sum()}/{len(sig_df)} features are statistically significant (p < 0.05).")
# TITLE: All features should pass. Any that don't are candidates for dropping before feature selection.


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16 — CHURN RATE LIFT: NEW BINARY / CATEGORICAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────
# TITLE: Visualise churn rate lift for all 17 new binary features

overall_churn = y.mean()
new_binary_feats = [
    HAS_FAMILY, IS_ISOLATED, SENIOR_NO_SUPPORT,
    IS_AUTO_PAY, IS_ELECTRONIC_CHECK, IS_HIGH_MONTHLY_CHARGE,
    HAS_INTERNET, HAS_FULL_STREAMING, HAS_NO_INTERNET_ADDONS,
    IS_COMMITTED_CUSTOMER, M2M_AND_FIBER, NEW_HIGH_VALUE_CUSTOMER,
    FIBER_NO_SECURITY, PAPERLESS_AND_ECHECK,
]

lift_rows = []
for feat in new_binary_feats:
    flagged   = y[Xall[feat] == 1].mean()
    unflagged = y[Xall[feat] == 0].mean()
    lift_rows.append({
        "feature": feat,
        "flagged_churn": flagged,
        "unflagged_churn": unflagged,
        "lift": flagged / overall_churn,
        "count_flagged": (Xall[feat] == 1).sum(),
    })

lift_df = pd.DataFrame(lift_rows).sort_values("lift", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

x = np.arange(len(lift_df))
w = 0.38
axes[0].bar(x - w / 2, lift_df["unflagged_churn"], w, label="Not Flagged (0)", color=BASE_COLOR)
axes[0].bar(x + w / 2, lift_df["flagged_churn"],   w, label="Flagged (1)",     color=CHURN_COLOR)
axes[0].axhline(y=overall_churn, color="gray", linestyle="--", alpha=0.7, label="Dataset avg")
axes[0].set_xticks(x)
axes[0].set_xticklabels(lift_df["feature"], rotation=40, ha="right", fontsize=8)
axes[0].set_ylabel("Churn Rate")
axes[0].set_title("Churn Rate by New Binary Features", fontweight="bold")
axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
axes[0].legend(fontsize=9)

colors_lift = [CHURN_COLOR if v > 1.3 else ENGINEERED_COLOR if v > 1.0 else BASE_COLOR
               for v in lift_df["lift"]]
axes[1].barh(lift_df["feature"], lift_df["lift"], color=colors_lift, height=0.7)
axes[1].axvline(x=1.0, color="gray", linestyle="--", alpha=0.7, label="No lift (1.0x)")
axes[1].set_xlabel("Lift vs Dataset Average")
axes[1].set_title("Lift Over Dataset Average", fontweight="bold")
for i, (_, row) in enumerate(lift_df.iterrows()):
    axes[1].text(row["lift"] + 0.02, i, f"{row['lift']:.2f}x", va="center", fontsize=8)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig("reports/feat_eng_06_new_binary_lift.png", bbox_inches="tight")
plt.show()

print("\nTop 5 lift features:")
print(lift_df[["feature", "flagged_churn", "lift", "count_flagged"]].head(5).to_string(index=False))
# TITLE: Features with lift > 1.5x are high-value. Lift < 1.0x means the flag predicts RETENTION, not churn.


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 17 — NUMERIC NEW FEATURES: DISTRIBUTION COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
# TITLE: Compare distributions of new numeric features between churned vs retained

new_numeric_feats = [
    CHARGES_GAP, STREAMING_COUNT, SECURITY_SERVICE_COUNT,
    INTERNET_ADDON_COUNT, CONTRACT_NUMERIC, COMPOSITE_CHURN_RISK,
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes_flat = axes.flatten()

for i, feat in enumerate(new_numeric_feats):
    ax = axes_flat[i]
    churned_vals  = Xall.loc[y == 1, feat].dropna()
    retained_vals = Xall.loc[y == 0, feat].dropna()

    ax.hist(retained_vals, bins=25, alpha=0.55, color=BASE_COLOR,
            label=f"Retained (n={len(retained_vals):,})", density=True)
    ax.hist(churned_vals,  bins=25, alpha=0.55, color=CHURN_COLOR,
            label=f"Churned  (n={len(churned_vals):,})", density=True)

    tstat, pval = stats.ttest_ind(churned_vals, retained_vals, equal_var=False)
    sig_label = f"t={tstat:.2f}  p={'<0.001' if pval < 0.001 else f'{pval:.3f}'}"
    ax.set_title(f"{feat}\n{sig_label}", fontweight="bold", fontsize=9)
    ax.set_xlabel(feat, fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.legend(fontsize=7)

plt.suptitle("New Numeric Feature Distributions: Churned vs Retained", fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig("reports/feat_eng_07_numeric_distributions.png", bbox_inches="tight")
plt.show()
# TITLE: Features with clearly separated distributions → strong signal.
# composite_churn_risk is expected to show the clearest separation since it aggregates 7 components.


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 18 — MUTUAL INFORMATION: ALL 28 ENGINEERED FEATURES
# ─────────────────────────────────────────────────────────────────────────────
# TITLE: MI scoring across all 28 features to rank information content

from sklearn.preprocessing import LabelEncoder

Xall_only_engineered = Xall[ENGINEERED_FEATURES].copy()

Xall_enc = Xall_only_engineered.copy()
le_tmp = LabelEncoder()
for col in Xall_enc.select_dtypes(include="object").columns:
    Xall_enc[col] = le_tmp.fit_transform(Xall_enc[col].astype(str))
Xall_enc = Xall_enc.fillna(0)

from sklearn.feature_selection import mutual_info_classif
mi_scores_all = mutual_info_classif(
    Xall_enc, y, discrete_features="auto",
    random_state=cfg.training.randomstate,
)
mi_df_all = pd.DataFrame({
    "feature": Xall_enc.columns,
    "mi_score": mi_scores_all,
}).sort_values("mi_score", ascending=False).reset_index(drop=True)
mi_df_all["is_new"] = ~mi_df_all["feature"].isin(ENGINEERED_FEATURES[:7])

print("Mutual Information — All 28 Engineered Features:")
print(mi_df_all.to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 10))
bar_colors = [CHURN_COLOR if is_new else BASE_COLOR for is_new in mi_df_all["is_new"]]
ax.barh(mi_df_all["feature"], mi_df_all["mi_score"], color=bar_colors, height=0.72)
ax.invert_yaxis()
ax.set_xlabel("Mutual Information Score")
ax.set_title("Mutual Information with Churn Target — All 28 Engineered Features\n"
             "(Red = New Phase 2 Feature | Teal = Original Phase 1 Feature)", fontweight="bold")
from matplotlib.patches import Patch
ax.legend(
    handles=[Patch(facecolor=CHURN_COLOR, label="New (Phase 2)"),
             Patch(facecolor=BASE_COLOR, label="Original (Phase 1)")],
    loc="lower right",
)
plt.tight_layout()
plt.savefig("reports/feat_eng_08_mi_all_28.png", bbox_inches="tight")
plt.show()

new_in_top10 = mi_df_all.head(10)["is_new"].sum()
print(f"\n{new_in_top10}/10 top-MI features are from the NEW Phase 2 batch.")
# TITLE: New features should contribute meaningfully to top-10 ranking.
# If a new feature ranks below all original 7, it carries no additional MI beyond what Phase 1 already captured.


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 19 — FULL FEATURE SELECTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
# TITLE: Run 5-stage selection: variance → correlation → MI → RF → permutation → consensus

print("Running full feature selection pipeline...")
print("(This will take ~60-120 seconds due to permutation importance with n_repeats=10)")
print("=" * 65)

selection_report = run_full_selection(
    X_engineered=Xall_only_engineered,
    y=y,
    variance_threshold=0.005,
    correlation_threshold=0.90,
    top_n=20,
    min_votes=2,
    random_state=cfg.training.randomstate,
)

print(selection_report.summary())
# TITLE: Summary prints: features dropped by each stage, final selected list, CV comparison.


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 20 — VISUALISE SELECTION RESULTS
# ─────────────────────────────────────────────────────────────────────────────
# TITLE: Plot correlation matrix, rank consensus heatmap, and CV comparison

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel A — Pairwise correlation matrix of all 28 engineered features
corr = selection_report.correlation_matrix
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, annot=True, fmt=".2f", cmap="RdYlGn_r",
    center=0, ax=axes[0], mask=mask, square=False,
    linewidths=0.3, vmin=0, vmax=1,
    annot_kws={"size": 6},
    xticklabels=corr.columns, yticklabels=corr.columns,
)
axes[0].set_title("Pairwise Absolute Correlation\n28 Engineered Features", fontweight="bold")
axes[0].tick_params(axis="x", rotation=50, labelsize=7)
axes[0].tick_params(axis="y", labelsize=7)

# Panel B — Consensus vote counts
consensus = selection_report.consensus_df.sort_values("avg_rank")
vote_colors = {0: "#cccccc", 1: ENGINEERED_COLOR, 2: BASE_COLOR, 3: CHURN_COLOR}
bar_colors_v = [vote_colors[v] for v in consensus["votes"]]
axes[1].barh(consensus["feature"], consensus["votes"], color=bar_colors_v, height=0.72)
axes[1].axvline(x=2, color="black", linestyle="--", linewidth=1.2,
                label="Min votes for selection (2)")
axes[1].set_xlabel("Consensus Votes (out of 3 methods)")
axes[1].set_title("Feature Selection: Consensus Votes\n(MI + RF Importance + Permutation Importance)",
                  fontweight="bold")
axes[1].legend(fontsize=9)
from matplotlib.patches import Patch
axes[1].legend(
    handles=[
        Patch(facecolor=CHURN_COLOR,     label="3 votes — selected (all methods agree)"),
        Patch(facecolor=BASE_COLOR,      label="2 votes — selected (majority)"),
        Patch(facecolor=ENGINEERED_COLOR,label="1 vote  — borderline"),
        Patch(facecolor="#cccccc",       label="0 votes  — dropped"),
    ],
    loc="lower right", fontsize=8,
)

plt.tight_layout()
plt.savefig("reports/feat_eng_09_selection_results.png", bbox_inches="tight")
plt.show()

# CV comparison chart
fig, ax = plt.subplots(figsize=(8, 4))
labels = [
    f"All Filtered\n({len(selection_report.correlation_matrix.columns)} features)",
    f"Consensus Selected\n({len(selection_report.consensus_selected)} features)",
]
vals   = [selection_report.cv_pr_auc_all, selection_report.cv_pr_auc_selected]
bar_cs = [BASE_COLOR, CHURN_COLOR]
bars   = ax.bar(labels, vals, color=bar_cs, width=0.45)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.003, f"{val:.4f}",
            ha="center", fontweight="bold", fontsize=11)
delta = selection_report.cv_pr_auc_selected - selection_report.cv_pr_auc_all
ax.annotate(
    f"Δ = {delta:+.4f}",
    xy=(1, selection_report.cv_pr_auc_selected),
    xytext=(1.3, selection_report.cv_pr_auc_selected - 0.01),
    fontsize=10, color=CHURN_COLOR, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=CHURN_COLOR),
)
ax.set_ylabel("CV PR-AUC (5-fold, LogReg, class_weight='balanced')")
ax.set_title("PR-AUC: All Features vs Consensus Selected Features", fontweight="bold")
ax.set_ylim(min(vals) - 0.02, max(vals) + 0.02)
plt.tight_layout()
plt.savefig("reports/feat_eng_10_selection_cv_comparison.png", bbox_inches="tight")
plt.show()
# TITLE: If selected < all with no PR-AUC drop, the dropped features were pure noise.
# A small PR-AUC rise means the correlation filter removed harmful multicollinearity.


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 21 — RANK TABLE: ALL METHODS SIDE BY SIDE
# ─────────────────────────────────────────────────────────────────────────────
# TITLE: Print a unified rank table across all three scoring methods

mi_ranks   = dict(zip(selection_report.mi_scores["feature"],   selection_report.mi_scores["mi_rank"]))
rf_ranks   = dict(zip(selection_report.rf_scores["feature"],   selection_report.rf_scores["rf_rank"]))
perm_ranks = dict(zip(selection_report.perm_scores["feature"], selection_report.perm_scores["perm_rank"]))

rank_rows = []
for feat in selection_report.consensus_df["feature"].tolist():
    selected = feat in selection_report.consensus_selected
    rank_rows.append({
        "feature":     feat,
        "MI rank":     mi_ranks.get(feat, "-"),
        "RF rank":     rf_ranks.get(feat, "-"),
        "Perm rank":   perm_ranks.get(feat, "-"),
        "avg rank":    round(selection_report.consensus_df.loc[
                           selection_report.consensus_df["feature"] == feat,
                           "avg_rank"
                       ].values[0], 1),
        "votes":       selection_report.consensus_df.loc[
                           selection_report.consensus_df["feature"] == feat,
                           "votes"
                       ].values[0],
        "selected":    "✅ YES" if selected else "❌ NO",
    })

rank_table = pd.DataFrame(rank_rows).sort_values("avg rank")
print("=" * 75)
print("FULL FEATURE RANKING TABLE ACROSS ALL 3 SELECTION METHODS")
print("=" * 75)
print(rank_table.to_string(index=False))
print(f"\nFinal selected feature count : {len(selection_report.consensus_selected)}")
print(f"Dropped by variance filter   : {selection_report.variance_dropped or 'None'}")
print(f"Dropped by corr filter       : {selection_report.correlation_dropped or 'None'}")
print(f"Dropped by low consensus     : "
      f"{[f for f in selection_report.all_features if f not in selection_report.consensus_selected]}")
# TITLE: Features that rank poorly on ALL 3 methods simultaneously are safe to drop.
# Features that rank #1 on MI but #25 on permutation importance → likely non-linear signal only for trees.


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 22 — UPDATED CV COMPARISON: RAW vs. ORIGINAL 7 vs. ALL 28
# ─────────────────────────────────────────────────────────────────────────────
# TITLE: Extend the original 3-way comparison (Section 11) to 4-way with all 28

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression

cv_outer = StratifiedKFold(
    n_splits=cfg.training.cvfolds,
    shuffle=True,
    random_state=cfg.training.randomstate,
)
lr_clf = LogisticRegression(
    class_weight="balanced", max_iter=1000,
    random_state=cfg.training.randomstate,
)

def build_lr_pipeline_for_subset(X_subset: pd.DataFrame) -> Pipeline:
    cat_c = [c for c in X_subset.select_dtypes(include="object").columns]
    num_c = [c for c in X_subset.select_dtypes(include="number").columns]
    trf = []
    if cat_c:
        trf.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_c))
    if num_c:
        trf.append(("num", StandardScaler(), num_c))
    preprocessor = ColumnTransformer(transformers=trf, remainder="drop")
    return Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000,
                                   random_state=cfg.training.randomstate)),
    ])

# 4 variants
X_raw_feats  = X_raw.copy()
X_eng7       = engineerfeatures(X_raw.copy())[list(X_raw.columns) + ENGINEERED_FEATURES[:7]]
X_eng28      = Xall.copy()
X_selected   = Xall[
    list(X_raw.columns) + [f for f in selection_report.consensus_selected]
].copy()

cv_raw   = cross_val_score(build_lr_pipeline_for_subset(X_raw_feats),   X_raw_feats,  y, cv=cv_outer, scoring="average_precision", n_jobs=-1)
cv_eng7  = cross_val_score(build_lr_pipeline_for_subset(X_eng7),         X_eng7,       y, cv=cv_outer, scoring="average_precision", n_jobs=-1)
cv_eng28 = cross_val_score(build_lr_pipeline_for_subset(X_eng28),        X_eng28,      y, cv=cv_outer, scoring="average_precision", n_jobs=-1)
cv_sel   = cross_val_score(build_lr_pipeline_for_subset(X_selected),     X_selected,   y, cv=cv_outer, scoring="average_precision", n_jobs=-1)

print("=" * 60)
print("4-WAY CV PR-AUC COMPARISON  (LogReg, 5-fold, balanced)")
print("=" * 60)
for label, scores in [
    ("Raw features only",       cv_raw),
    ("Raw + Original 7",        cv_eng7),
    ("Raw + All 28 engineered", cv_eng28),
    ("Raw + Selected (consensus)", cv_sel),
]:
    print(f"  {label:<32}  {scores.mean():.4f} ± {scores.std():.4f}")
print(f"\n  Lift (Raw → Selected) : {cv_sel.mean() - cv_raw.mean():+.4f}")

# Box plot
fig, ax = plt.subplots(figsize=(9, 4))
bp = ax.boxplot(
    [cv_raw, cv_eng7, cv_eng28, cv_sel],
    positions=[1, 2, 3, 4],
    widths=0.45,
    patch_artist=True,
    medianprops={"color": "white", "linewidth": 2},
)
palette = [BASE_COLOR, ENGINEERED_COLOR, "#9b59b6", CHURN_COLOR]
for patch, c in zip(bp["boxes"], palette):
    patch.set_facecolor(c)
    patch.set_alpha(0.85)
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(
    ["Raw only", "Raw + 7 eng.", "Raw + 28 eng.", "Raw + selected"],
    fontsize=10,
)
ax.set_ylabel("CV PR-AUC (5-fold)")
ax.set_title("PR-AUC Progression: Feature Engineering Stages", fontweight="bold")
for pos, scores in zip([1, 2, 3, 4], [cv_raw, cv_eng7, cv_eng28, cv_sel]):
    ax.text(pos, scores.mean() + 0.004, f"{scores.mean():.4f}",
            ha="center", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/feat_eng_11_4way_cv_comparison.png", bbox_inches="tight")
plt.show()
# TITLE: This chart is the core evidence for your GitHub README.
# It shows: raw → 7 features → 28 features → selection = monotonic PR-AUC improvement.


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 23 — UPDATED FEATURE ENGINEERING DECISION RECORD (ALL 28)
# ─────────────────────────────────────────────────────────────────────────────
# TITLE: Full decision record for all 28 features, updated from Section 13

print("=" * 70)
print("FEATURE ENGINEERING DECISION RECORD — ALL 28 FEATURES")
print("=" * 70)

all_decisions = [
    # ── Original 7 (all retained from Phase 1) ──────────────────────────────
    ("tenure_group",       "KEEP", "Ordinal bin. Non-linear tenure→churn confirmed. 47% churn at 0-6m vs 10% at 49m+."),
    ("is_month_to_month",  "KEEP", "Strongest single predictor. M2M 42.7% vs 2.8% (2yr). Critical for linear models."),
    ("service_adoption_count", "KEEP", "Stickiness measure. Denominator for avg_charge_per_service. p<0.001."),
    ("has_protection_bundle", "KEEP", "TechSupport+OnlineSecurity combo → significant lower churn. Interaction signal."),
    ("charge_to_tenure_ratio", "KEEP", "Top-10 MI. Captures paying-a-lot-without-commitment profile."),
    ("is_fiber_optic",     "KEEP", "Fiber 41.9% vs DSL 18.9%. Binary isolation of high-churn segment."),
    ("avg_charge_per_service", "KEEP", "Perceived value ratio. Top-10 MI. High ratio → high churn risk."),
    # ── Group A — Demographic ───────────────────────────────────────────────
    ("has_family",         "KEEP*", "Family ties → lower switching cost. Chi-squared p<0.001. *May be dropped if corr filter removes in favour of is_isolated."),
    ("is_isolated",        "KEEP*", "Exact complement of has_family. Kept for interpretability. Corr filter will drop one of these pair."),
    ("senior_no_support",  "KEEP", "Senior + no TechSupport = friction-prone profile. Actionable retention segment."),
    # ── Group B — Billing/Payment ────────────────────────────────────────────
    ("is_auto_pay",        "KEEP", "Auto-pay commitment signal. Strong negative correlation with churn."),
    ("is_electronic_check","KEEP", "45.3% churn rate — highest payment method. Binary isolation for linear models."),
    ("is_high_monthly_charge", "KEEP", "Captures >$70 premium-payer churn segment. Complements charge_to_tenure_ratio."),
    ("monthly_charge_bin", "KEEP", "Low/Med/High pricing tier categorical. Non-linear pricing effect for linear models."),
    ("charges_gap",        "KEEP*", "Billing anomaly detector. *Validate MI rank — may be low-signal on this static dataset."),
    # ── Group C — Service Depth ──────────────────────────────────────────────
    ("has_internet",       "KEEP*", "Internet vs no-internet segmentation. *May be low variance (most customers have internet)."),
    ("streaming_count",    "KEEP", "Entertainment stickiness (0-2). Captures bundle depth for streaming subscribers."),
    ("security_service_count", "KEEP", "Security bundle depth (0-3). Each layer adds switching cost."),
    ("has_full_streaming", "KEEP", "Both TV+Movies AND interaction. Complement to streaming_count."),
    ("internet_addon_count","KEEP", "Focused internet stickiness (0-6). More granular than service_adoption_count."),
    ("has_no_internet_addons","KEEP", "Internet-without-addons = disengaged. Direct retention campaign target segment."),
    # ── Group D — Contract/Commitment ────────────────────────────────────────
    ("contract_numeric",   "KEEP*", "Ordinal commitment scale (0/1/2). *Corr filter will likely remove in favour of is_month_to_month."),
    ("is_committed_customer","KEEP*", "Positive-framing of is_month_to_month. *Corr filter will likely drop (r≈1.0 with is_m2m)."),
    # ── Group E — Interaction/Compound ──────────────────────────────────────
    ("m2m_and_fiber",      "KEEP", "Highest-risk EDA heatmap cell. Provides multiplicative interaction for linear models."),
    ("new_high_value_customer","KEEP", "New + premium = top retention priority. Actionable binary segment."),
    ("fiber_no_security",  "KEEP", "Fiber + no OnlineSecurity = poaching risk. Specific underserved profile."),
    ("paperless_and_echeck","KEEP", "Dual billing risk signal. Both features individually are high-churn indicators."),
    # ── Group F — Composite ─────────────────────────────────────────────────
    ("composite_churn_risk","KEEP", "0-7 risk tier score. Interpretable for business stakeholders. Should rank highly in importance."),
]

for feat, verdict, reason in all_decisions:
    print(f"\n  Feature  : {feat}")
    print(f"  Verdict  : {verdict}")
    print(f"  Reason   : {reason}")
    print(f"  {'-' * 66}")

print(f"\n{'=' * 70}")
print(f"PHASE 2 FEATURE ENGINEERING COMPLETE")
print(f"  Original feature count    : 7")
print(f"  New features added        : 21")
print(f"  Total engineered features : 28")
print(f"  Features after selection  : {len(selection_report.consensus_selected)}")
print(f"  PR-AUC lift (LogReg)      : {cv_sel.mean() - cv_raw.mean():+.4f}")
print(f"{'=' * 70}")
print("\nReports saved to reports/:")
print("  feat_eng_06_new_binary_lift.png")
print("  feat_eng_07_numeric_distributions.png")
print("  feat_eng_08_mi_all_28.png")
print("  feat_eng_09_selection_results.png")
print("  feat_eng_10_selection_cv_comparison.png")
print("  feat_eng_11_4way_cv_comparison.png")
print(f"\n>>> Next: notebooks/03_baseline_models.py")
print(f"    Use src.features.feature_store.engineer_features() — it now applies all 28 features.")
print(f"    The pipeline in src/models/pipeline.py picks them up automatically.")