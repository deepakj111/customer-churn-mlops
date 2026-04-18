"""
Train Causal Uplift Model.

Demonstrates advanced prescriptive analytics by training an econml T-Learner
to predict the treatment effect of adding 'TechSupport' on churn probability.
"""

import json
from pathlib import Path

import pandas as pd

from src.data.ingest import load_for_training
from src.data.preprocess import preprocess
from src.features.feature_store import engineer_features
from src.models.uplift import ChurnUpliftModel
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_uplift_experiment():
    logger.info("Starting Causal Uplift experiment.")

    # 1. Load and process data
    df_raw = load_for_training()
    df_clean = preprocess(df_raw)

    # In causal inference, confounding factors are a big problem.
    # Example: Customers without internet cannot get Tech Support.
    # We must filter to only those eligible for the treatment.
    df_eligible = df_clean[df_clean["InternetService"] != "No"].copy()
    logger.info(f"Filtered to {len(df_eligible)} internet-eligible customers.")

    # Cast TotalCharges to numeric since the sklearn pipeline is bypassed
    if "TotalCharges" in df_eligible.columns:
        df_eligible["TotalCharges"] = pd.to_numeric(
            df_eligible["TotalCharges"].replace(" ", ""), errors="coerce"
        ).fillna(0.0)

    # We engineer features
    df_features = engineer_features(df_eligible)

    # Add a synthetic "Discount_Offered" treatment, or use TechSupport.
    # TechSupport is strongly correlated with retention, so we act as if
    # we want to "intervene" by offering it free to those who lack it.
    T = (df_features["TechSupport"] == "Yes").astype(int)
    y = df_features["Churn"]

    # 2. X represents the features *before* treatment
    # We must drop TechSupport and anything highly collinear with the treatment
    # or outcome from X to satisfy unconfoundedness.
    X_cols = [c for c in df_features.columns if c not in ["Churn", "TechSupport"]]
    X = df_features[X_cols].copy()

    # EconML meta-learners require completely numeric matrices.
    # We will one-hot encode all categorical string columns.
    X = pd.get_dummies(X, drop_first=True, dtype=float)

    # Also ensure there are no blank strings in TotalCharges (which happen for tenure=0)
    if "TotalCharges" in X.columns and X["TotalCharges"].dtype == object:
        X["TotalCharges"] = pd.to_numeric(
            X["TotalCharges"].replace(" ", ""), errors="coerce"
        ).fillna(0.0)

    # 3. Train Uplift Model
    uplift_model = ChurnUpliftModel(random_state=42)
    uplift_model.fit(X, T, y)

    # 4. Predict CATE for untreated customers (the target cohort)
    X_target = X[T == 0].copy()
    cate_preds = uplift_model.predict_uplift(X_target)

    # 5. Base Churn Probabilities (control model).
    # T-Learner has model_0 and model_1; extract control predictions:
    base_churn_prob = pd.Series(
        uplift_model.model.models[0].predict_proba(X_target.values)[:, 1],
        index=X_target.index,
    )

    # 6. Segment Customers
    segments_df = uplift_model.segment_customers(X_target, cate_preds, base_churn_prob)

    # Calculate aggregation
    segment_counts = segments_df["uplift_segment"].value_counts().to_dict()
    avg_cate = {
        k: float(segments_df[segments_df["uplift_segment"] == k]["uplift_cate"].mean())
        for k in segment_counts.keys()
    }

    logger.info(f"Segmentation Results:\n{json.dumps(segment_counts, indent=2)}")

    # 7. Save Report
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_data = {
        "eligible_customers": len(X_target),
        "segments": segment_counts,
        "avg_cate_by_segment": avg_cate,
    }

    with open(reports_dir / "uplift_report.json", "w") as f:
        json.dump(report_data, f, indent=4)

    logger.info("Saved uplift report to reports/uplift_report.json")


if __name__ == "__main__":
    run_uplift_experiment()
