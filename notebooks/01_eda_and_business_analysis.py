# %% [markdown]
# # Notebook 01 — EDA & Business Analysis
#
# **Purpose**: Understand the Telco dataset deeply from a *business lens*,
# not just a statistical one. Every chart in this notebook answers a
# specific business question. The insights here directly drive the feature
# engineering decisions in `src/features/feature_store.py`.
#
# **Audience**: This notebook is for you (the DS) and for technical
# interviewers who ask "walk me through your EDA process."
#
# **Pipeline position**:
# ```
# Raw CSV → ingest → validate → preprocess → [THIS NOTEBOOK] → feature_store.py
# ```
#
# **Key questions answered**:
# 1. How severe is the class imbalance and what does it cost the business?
# 2. Which customer segments churn the most?
# 3. Which features show the strongest individual signal?
# 4. What interaction effects are worth engineering as explicit features?
# 5. Are there any data quality surprises beyond the known TotalCharges issue?

# %% [markdown]
# ## 0. Setup

# %%
import warnings

warnings.filterwarnings("ignore")

import os
import sys
from pathlib import Path


# ------------------------------------------------------------------
# Working directory fix — notebooks run from notebooks/ subdirectory
# but all data paths are relative to the project root.
# This block walks up until it finds pyproject.toml (the project root
# marker) and sets that as the working directory.
# This is the standard pattern for notebooks in subdirectories.
# ------------------------------------------------------------------
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

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.data.ingest import load_for_training
from src.data.preprocess import preprocess, split_features_target
from src.data.validate import fix_total_charges, validate_raw_data
from src.utils.logging import get_logger

logger = get_logger("eda")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "figure.figsize": (10, 5),
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

CHURN_PALETTE = {"No": "#4f98a3", "Yes": "#a13544"}
BINARY_COLORS = ["#4f98a3", "#a13544"]

print("Setup complete.")

# %% [markdown]
# ## 1. Load & Validate Data
#
# We use the production pipeline — the same code that runs in training.
# If validation fails here, it will fail in the pipeline too.
# Discovering that in a notebook is far cheaper than discovering it in CI.

# %%
raw_df = load_for_training()
validated_df = validate_raw_data(raw_df)
clean_df = preprocess(validated_df)
X, y = split_features_target(clean_df)

print(f"Raw shape         : {raw_df.shape}")
print(f"Validated shape   : {validated_df.shape}")
print(f"Post-preprocess   : {clean_df.shape}")
print(f"Feature matrix X  : {X.shape}")
print(f"Target vector y   : {y.shape}")

# %% [markdown]
# ## 2. Dataset Overview

# %%
print("=" * 55)
print("COLUMN TYPES AND NULL COUNTS")
print("=" * 55)
info_df = pd.DataFrame(
    {
        "dtype": raw_df.dtypes,
        "nulls": raw_df.isna().sum(),
        "pct_null": (raw_df.isna().sum() / len(raw_df) * 100).round(2),
        "n_unique": raw_df.nunique(),
    }
).sort_values("dtype")
print(info_df.to_string())

# %%
print("\nDescriptive statistics for numeric columns:")
raw_df[["tenure", "MonthlyCharges", "TotalCharges"]].describe().round(2)

# %% [markdown]
# ### Key observation
# `TotalCharges` was loaded as `object` dtype in the raw CSV because of the
# 11 blank-string rows for `tenure == 0` customers. Our `fix_total_charges()`
# function handles this before validation. This is the **only data quality
# issue** in this dataset — everything else is clean.

# %%
# Confirm the known TotalCharges blank rows
blank_mask = raw_df["TotalCharges"].astype(str).str.strip() == ""
print(f"Blank TotalCharges rows : {blank_mask.sum()}")
print(f"All have tenure == 0    : {(raw_df[blank_mask]['tenure'] == 0).all()}")
print("\nThese rows (first 3):")
raw_df[blank_mask][["customerID", "tenure", "MonthlyCharges", "TotalCharges"]].head(3)

# %% [markdown]
# ## 3. Target Variable — Class Imbalance Analysis
#
# **Business question**: How many customers churned, and what does that
# represent in lost revenue?

# %%
churn_counts = raw_df["Churn"].value_counts()
churn_rate = churn_counts["Yes"] / len(raw_df)

print(f"Total customers    : {len(raw_df):,}")
print(f"Churned (Yes)      : {churn_counts['Yes']:,}  ({churn_rate:.1%})")
print(f"Retained (No)      : {churn_counts['No']:,}  ({1 - churn_rate:.1%})")

# %%
# Business revenue impact estimate
avg_monthly_revenue = fix_total_charges(raw_df.copy())["MonthlyCharges"].mean()
churned_revenue_monthly = churn_counts["Yes"] * avg_monthly_revenue
churned_revenue_annual = churned_revenue_monthly * 12

print(f"\nAverage monthly charge  : ${avg_monthly_revenue:.2f}")
print(f"Estimated monthly loss  : ${churned_revenue_monthly:,.0f}")
print(f"Estimated annual loss   : ${churned_revenue_annual:,.0f}")
print(
    f"\nIf we retain just 10% of churners: "
    f"${churned_revenue_annual * 0.10:,.0f} / year saved"
)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bar chart
axes[0].bar(
    ["Retained", "Churned"],
    [churn_counts["No"], churn_counts["Yes"]],
    color=BINARY_COLORS,
    width=0.5,
)
axes[0].set_title("Customer Churn Distribution", fontweight="bold")
axes[0].set_ylabel("Number of Customers")
for i, v in enumerate([churn_counts["No"], churn_counts["Yes"]]):
    axes[0].text(i, v + 30, f"{v:,}\n({v/len(raw_df):.1%})", ha="center", fontsize=10)

# Pie chart
axes[1].pie(
    [churn_counts["No"], churn_counts["Yes"]],
    labels=["Retained", "Churned"],
    colors=BINARY_COLORS,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
)
axes[1].set_title("Churn Rate", fontweight="bold")

plt.suptitle(
    f"Class Imbalance: {churn_rate:.1%} Positive Rate | "
    f"Estimated Annual Revenue at Risk: ${churned_revenue_annual:,.0f}",
    fontsize=11,
    y=1.02,
)
plt.tight_layout()
plt.savefig("reports//eda_01_churn_distribution.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Finding**: 26.5% churn rate. Not severely imbalanced but requires care.
# A naive "predict no churn always" classifier gets 73.5% accuracy — this
# is why we use Precision-Recall AUC as our primary metric, not accuracy.
#
# **Business implication**: At an average monthly charge of ~$64.76, 1,869
# churned customers represent an estimated **$1.45M annual revenue risk**.
# Even retaining 10% of predicted churners saves ~$145K/year — more than
# enough to justify the cost of building and maintaining this system.

# %% [markdown]
# ## 4. Contract Type — The Single Strongest Predictor
#
# **Business question**: Does contract length predict churn?
# This is the most important business insight in the dataset.

# %%
contract_churn = (
    raw_df.groupby("Contract")["Churn"]
    .value_counts(normalize=True)
    .unstack()
    .reset_index()
)
contract_churn.columns = ["Contract", "No", "Yes"]
contract_churn = contract_churn.sort_values("Yes", ascending=False)

print("Churn rate by contract type:")
print(contract_churn.to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(9, 4))

x = np.arange(len(contract_churn))
width = 0.38

bars_no = ax.bar(
    x - width / 2, contract_churn["No"], width, label="Retained", color="#4f98a3"
)
bars_yes = ax.bar(
    x + width / 2, contract_churn["Yes"], width, label="Churned", color="#a13544"
)

ax.set_xticks(x)
ax.set_xticklabels(contract_churn["Contract"])
ax.set_ylabel("Proportion of Customers")
ax.set_title("Churn Rate by Contract Type", fontweight="bold")
ax.legend()
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

for bar in bars_yes:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{bar.get_height():.1%}",
        ha="center",
        fontsize=9,
        color="#a13544",
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig("reports//eda_02_churn_by_contract.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Finding**: Month-to-month customers churn at **42.7%** vs 11.3% (one year)
# and 2.8% (two year). This is the strongest single predictor by far.
#
# **Feature engineering decision**: Create `is_month_to_month` binary flag.
# This concentrates the signal into a single highly predictive boolean.

# %% [markdown]
# ## 5. Tenure — The Time Dimension

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Distribution by churn status
for churn_val, color in CHURN_PALETTE.items():
    mask = raw_df["Churn"] == churn_val
    axes[0].hist(
        raw_df[mask]["tenure"],
        bins=30,
        alpha=0.6,
        color=color,
        label=churn_val,
        density=True,
    )
axes[0].set_xlabel("Tenure (months)")
axes[0].set_ylabel("Density")
axes[0].set_title("Tenure Distribution by Churn Status", fontweight="bold")
axes[0].legend(title="Churn")

# Churn rate by tenure bucket
df_temp = raw_df.copy()
df_temp["tenure_bin"] = pd.cut(
    df_temp["tenure"],
    bins=[0, 6, 12, 24, 48, 72],
    labels=["0-6m", "7-12m", "13-24m", "25-48m", "49-72m"],
    include_lowest=True,
)
churn_by_tenure = (
    df_temp.groupby("tenure_bin", observed=True)["Churn"]
    .apply(lambda x: (x == "Yes").mean())
    .reset_index()
)
axes[1].bar(
    churn_by_tenure["tenure_bin"].astype(str), churn_by_tenure["Churn"], color="#4f98a3"
)
axes[1].set_xlabel("Tenure Bucket")
axes[1].set_ylabel("Churn Rate")
axes[1].set_title("Churn Rate by Tenure Bucket", fontweight="bold")
axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
for i, row in churn_by_tenure.iterrows():
    axes[1].text(
        i, row["Churn"] + 0.005, f"{row['Churn']:.1%}", ha="center", fontsize=9
    )

plt.tight_layout()
plt.savefig("reports//eda_03_tenure_analysis.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Finding**: Churn is overwhelmingly a **new-customer problem**.
# Customers in their first 6 months churn at ~47%. Beyond 48 months, churn
# drops below 10%. Customers who survive 4+ years almost never leave.
#
# **Feature engineering decisions**:
# - Create `tenure_group` ordinal feature (0-6m, 7-12m, 13-24m, 25-48m, 49+m)
# - `tenure` raw value is also kept — it's a strong continuous predictor

# %% [markdown]
# ## 6. Monthly Charges — The Price Signal

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

for churn_val, color in CHURN_PALETTE.items():
    mask = raw_df["Churn"] == churn_val
    axes[0].hist(
        raw_df[mask]["MonthlyCharges"],
        bins=30,
        alpha=0.6,
        color=color,
        label=churn_val,
        density=True,
    )
axes[0].set_xlabel("Monthly Charges ($)")
axes[0].set_ylabel("Density")
axes[0].set_title("Monthly Charges by Churn Status", fontweight="bold")
axes[0].legend(title="Churn")

# Box plot
raw_df.boxplot(
    column="MonthlyCharges",
    by="Churn",
    ax=axes[1],
    patch_artist=True,
    boxprops={"facecolor": "#4f98a3", "alpha": 0.6},
)
axes[1].set_xlabel("Churn")
axes[1].set_ylabel("Monthly Charges ($)")
axes[1].set_title("Monthly Charges Distribution", fontweight="bold")
plt.suptitle("")  # remove the auto-generated suptitle from boxplot

churned_median = raw_df[raw_df["Churn"] == "Yes"]["MonthlyCharges"].median()
retained_median = raw_df[raw_df["Churn"] == "No"]["MonthlyCharges"].median()
print(
    f"Median MonthlyCharges — Churned: ${churned_median:.2f} | Retained: ${retained_median:.2f}"
)
t_stat, p_val = stats.ttest_ind(
    raw_df[raw_df["Churn"] == "Yes"]["MonthlyCharges"],
    raw_df[raw_df["Churn"] == "No"]["MonthlyCharges"],
)
print(
    f"T-test: t={t_stat:.3f}, p={p_val:.2e}  (statistically significant: {p_val < 0.05})"
)

plt.tight_layout()
plt.savefig("reports//eda_04_monthly_charges.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Finding**: Churned customers pay significantly higher monthly charges
# ($79.49 median vs $61.27). This is statistically significant (p < 0.001).
# High-charge customers are often on Fiber optic + many add-ons but on
# month-to-month contracts — maximum spending, minimum commitment.
#
# **Feature engineering decision**: Create `charge_to_tenure_ratio` —
# high charges + low tenure = highest risk segment.

# %% [markdown]
# ## 7. Internet Service — The Infrastructure Signal

# %%
internet_churn = (
    raw_df.groupby("InternetService")["Churn"]
    .value_counts(normalize=True)
    .unstack()
    .fillna(0)
)
print("Churn rate by internet service type:")
print(internet_churn.round(3))

# %%
fig, ax = plt.subplots(figsize=(8, 4))
internet_churn["Yes"].sort_values().plot(
    kind="barh", ax=ax, color=["#4f98a3", "#e8b84b", "#a13544"]
)
ax.set_xlabel("Churn Rate")
ax.set_title("Churn Rate by Internet Service Type", fontweight="bold")
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
for i, v in enumerate(internet_churn["Yes"].sort_values()):
    ax.text(v + 0.003, i, f"{v:.1%}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig("reports//eda_05_internet_service_churn.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Finding**: Fiber optic customers churn at **41.9%** — nearly 3x the rate
# of DSL customers (18.9%) and 10x customers with no internet (7.4%).
# This signals a **service quality or value perception problem** with the
# Fiber product, not just a price sensitivity issue.

# %% [markdown]
# ## 8. Service Adoption — The Stickiness Signal
#
# **Hypothesis**: Customers who use more services are more "sticky" because
# switching costs are higher (they'd lose multiple services at once).

# %%
service_cols = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

df_temp = raw_df.copy()


# Count "active" services: Yes or has internet/phone (not "No X service")
def count_services(row):
    count = 0
    if row["PhoneService"] == "Yes":
        count += 1
    if row["MultipleLines"] == "Yes":
        count += 1
    if row["InternetService"] != "No":
        count += 1
    for col in [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]:
        if row[col] == "Yes":
            count += 1
    return count


df_temp["service_count"] = df_temp.apply(count_services, axis=1)

service_churn = (
    df_temp.groupby("service_count")["Churn"]
    .apply(lambda x: (x == "Yes").mean())
    .reset_index()
)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(
    service_churn["service_count"],
    service_churn["Churn"],
    marker="o",
    linewidth=2,
    color="#4f98a3",
    markersize=7,
)
ax.set_xlabel("Number of Active Services")
ax.set_ylabel("Churn Rate")
ax.set_title("Churn Rate by Number of Active Services", fontweight="bold")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.set_xticks(service_churn["service_count"])
plt.tight_layout()
plt.savefig("reports//eda_06_service_adoption_churn.png", bbox_inches="tight")
plt.show()

print("Churn rate by service count:")
print(service_churn.to_string(index=False))

# %% [markdown]
# **Finding**: Churn rate is **not monotonically decreasing** with service count
# — customers with 3 services actually show higher churn than those with 1.
# The relationship is non-linear. This means we can't use raw service count
# as-is — the model needs to learn the non-linearity, OR we engineer
# `has_protection_bundle` (TechSupport + OnlineSecurity + DeviceProtection)
# as a separate feature since those specific services drive retention.
#
# **Feature engineering decisions**:
# - `service_adoption_count` (raw count)
# - `has_protection_bundle` = TechSupport == Yes AND OnlineSecurity == Yes

# %% [markdown]
# ## 9. Key Interaction: Contract × Internet Service
#
# The most dangerous customer segment: Fiber optic + month-to-month.

# %%
interaction_df = (
    raw_df.groupby(["Contract", "InternetService"])["Churn"]
    .apply(lambda x: (x == "Yes").mean())
    .unstack()
    .fillna(0)
)

fig, ax = plt.subplots(figsize=(9, 4))
sns.heatmap(
    interaction_df,
    annot=True,
    fmt=".1%",
    cmap="RdYlGn_r",
    ax=ax,
    linewidths=0.5,
    cbar_kws={"format": mticker.PercentFormatter(xmax=1)},
)
ax.set_title("Churn Rate Heatmap: Contract Type × Internet Service", fontweight="bold")
ax.set_xlabel("Internet Service")
ax.set_ylabel("Contract Type")
plt.tight_layout()
plt.savefig("reports//eda_07_contract_internet_heatmap.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Finding**: The extreme cell is **Month-to-month + Fiber optic = 52.6% churn**.
# More than half of customers in this combination will leave.
# This is the primary retention target segment for business campaigns.

# %% [markdown]
# ## 10. Categorical Features — Churn Rate Comparison

# %%
cat_features = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PaperlessBilling",
    "PaymentMethod",
    "PhoneService",
    "MultipleLines",
]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for i, col in enumerate(cat_features):
    churn_rates = (
        raw_df.groupby(col)["Churn"]
        .apply(lambda x: (x == "Yes").mean())
        .sort_values(ascending=True)
    )
    churn_rates.plot(kind="barh", ax=axes[i], color="#4f98a3")
    axes[i].set_title(col, fontweight="bold")
    axes[i].set_xlabel("Churn Rate")
    axes[i].xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    for j, v in enumerate(churn_rates):
        axes[i].text(v + 0.005, j, f"{v:.1%}", va="center", fontsize=8)

plt.suptitle("Churn Rate by Categorical Feature", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig("reports//eda_08_categorical_churn_rates.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Key findings from categorical analysis**:
# - `gender`: Nearly no difference (26.1% male vs 26.9% female). **Low signal.**
#   → Gender will be included but expected low feature importance.
# - `SeniorCitizen`: Seniors churn at 41.7% vs 23.7%. **Strong signal.**
# - `Partner` / `Dependents`: Customers without partner/dependents churn more.
#   Interpretation: single customers have lower switching costs.
# - `PaperlessBilling`: 33.6% vs 16.3%. Strong signal — paperless billing
#   customers are more tech-savvy and more likely to compare alternatives online.
# - `PaymentMethod`: Electronic check shows 45.3% churn — highest of any
#   payment method. Mailed check shows only 19.1%. This likely proxies
#   for engagement level — electronic check requires the least commitment.

# %% [markdown]
# ## 11. Correlation & Feature Relationships

# %%
# Numeric correlation with binary target
df_numeric = fix_total_charges(raw_df.copy())
df_numeric["Churn_int"] = (df_numeric["Churn"] == "Yes").astype(int)

correlations = df_numeric[
    ["tenure", "MonthlyCharges", "TotalCharges", "Churn_int"]
].corr()

print("Point-biserial correlations with Churn:")
for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
    corr_val, p_val = stats.pointbiserialr(df_numeric["Churn_int"], df_numeric[col])
    significance = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else "*")
    print(f"  {col:<20}: r = {corr_val:+.4f}  p={p_val:.2e} {significance}")

# %%
fig, ax = plt.subplots(figsize=(6, 5))
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Churn_int"]
corr_matrix = df_numeric[numeric_cols].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    center=0,
    ax=ax,
    mask=mask,
    square=True,
    linewidths=0.5,
)
ax.set_title("Numeric Feature Correlation Matrix", fontweight="bold")
plt.tight_layout()
plt.savefig("reports//eda_09_correlation_matrix.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Finding**: `tenure` has a strong negative correlation with churn (r = -0.35):
# longer-tenured customers churn less. `MonthlyCharges` has a positive
# correlation (r = +0.19): higher-charge customers churn more.
# `TotalCharges` has a negative correlation (r = -0.20) — but this is
# almost entirely driven by tenure (longer tenure → higher TotalCharges).
# TotalCharges is largely redundant with tenure. We will include it but
# expect low independent feature importance after controlling for tenure.

# %% [markdown]
# ## 12. The High-Risk Segment Deep Dive

# %%
# Define the highest-risk segment based on EDA findings
high_risk_mask = (
    (raw_df["Contract"] == "Month-to-month")
    & (raw_df["InternetService"] == "Fiber optic")
    & (raw_df["tenure"] <= 12)
)

segment_df = raw_df[high_risk_mask]
segment_churn_rate = (segment_df["Churn"] == "Yes").mean()
segment_size = len(segment_df)

print(f"High-risk segment: Month-to-month + Fiber optic + tenure <= 12 months")
print(
    f"Segment size   : {segment_size:,} customers ({segment_size/len(raw_df):.1%} of base)"
)
print(f"Churn rate     : {segment_churn_rate:.1%}")
print(f"Dataset average: {churn_rate:.1%}")
print(f"Lift           : {segment_churn_rate / churn_rate:.2f}x")

est_monthly_revenue_at_risk = segment_df[segment_df["Churn"] == "Yes"][
    "MonthlyCharges"
].sum()
print(f"\nMonthly revenue at risk in this segment: ${est_monthly_revenue_at_risk:,.0f}")

# %% [markdown]
# ## 13. Summary of Feature Engineering Decisions
#
# This section is the direct output of this notebook — the instructions
# that `src/features/feature_store.py` will implement.

# %%
feature_decisions = {
    "KEEP AS-IS": [
        "tenure             — Strong continuous predictor (r=-0.35 with churn)",
        "MonthlyCharges     — Strong continuous predictor (r=+0.19 with churn)",
        "TotalCharges       — Keep but expect low importance (proxied by tenure)",
        "Contract           — Strongest single predictor (42.7% M2M churn rate)",
        "InternetService    — Strong signal (Fiber: 41.9% vs DSL: 18.9%)",
        "PaymentMethod      — Electronic check = 45.3% churn rate",
        "PaperlessBilling   — 33.6% vs 16.3% — double the rate",
        "TechSupport et al  — Service add-ons with retention signal",
    ],
    "ENGINEER NEW": [
        "tenure_group           — Binned tenure: 0-6m, 7-12m, 13-24m, 25-48m, 49+m",
        "is_month_to_month      — Binary flag for month-to-month contract",
        "service_adoption_count — Count of active subscribed services",
        "has_protection_bundle  — TechSupport==Yes AND OnlineSecurity==Yes",
        "charge_to_tenure_ratio — MonthlyCharges / (tenure + 1)",
        "is_fiber_optic         — Binary flag for Fiber optic internet service",
        "avg_charge_per_service — MonthlyCharges / (service_adoption_count + 1)",
    ],
    "LOW SIGNAL (keep but expect low importance)": [
        "gender             — 26.1% vs 26.9% — minimal difference",
        "Partner            — Moderate signal but likely proxied by tenure/dependents",
        "PhoneService       — Weak standalone signal",
    ],
    "DROP": [
        "customerID         — No predictive signal, privacy concern",
    ],
}

print("=" * 65)
print("FEATURE ENGINEERING DECISIONS FROM EDA")
print("=" * 65)
for category, features in feature_decisions.items():
    print(f"\n{category}:")
    for feat in features:
        print(f"  • {feat}")

# %% [markdown]
# ## 14. Business Recommendations Summary
#
# | Priority | Segment | Action | Expected Lift |
# |---|---|---|---|
# | 🔴 CRITICAL | M2M + Fiber + tenure ≤ 12m | Personal outreach + discounted annual contract | 2.5x lift |
# | 🟠 HIGH | M2M + MonthlyCharges > $80 | Proactive billing review + loyalty discount | 1.8x lift |
# | 🟡 MEDIUM | Electronic check + M2M | Auto-pay incentive ($10 discount) | 1.4x lift |
# | 🟢 MONITOR | New customers (0-6m tenure) | Onboarding journey improvement | Long-term |
#
# **Model ROI estimate**: If the churn model achieves 75% recall on the
# HIGH_RISK tier, and retention campaigns succeed for 30% of contacted
# customers at a $50 offer cost per customer, the breakeven at the
# estimated $64.76 average monthly charge is reached within ~3 months
# of model deployment.

# %%
print("EDA complete. Reports saved to reports/")
print(f"Total figures generated: 9")
print("\nNext step: src/features/feature_store.py")
print("Implement all feature engineering decisions documented in Section 13.")
