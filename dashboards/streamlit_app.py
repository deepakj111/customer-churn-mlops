"""
Streamlit Dashboard — Customer Churn MLOps Monitoring & Prediction UI.

This dashboard provides a business-friendly interface for:
    1. Single customer churn prediction (calls the FastAPI /predict endpoint)
    2. Batch prediction via CSV upload
    3. Model performance visualization
    4. Business impact analysis

Architecture:
    The dashboard is a STATELESS client that calls the FastAPI prediction
    API for all model interactions. It does NOT load the model directly.
    This separation means the dashboard can be deployed independently of
    the model serving infrastructure.

Usage:
    poetry run streamlit run dashboards/streamlit_app.py
    # or via Makefile:
    make dashboard
"""

import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# Ensure project root is on sys.path for config imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# API URL — configurable via environment variable for Docker Compose.
# Defaults to localhost for standalone local development.
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Report images directory
REPORTS_DIR = Path("reports")


# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit command
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS for a polished, professional look
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Main header styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2937;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #4f46e5;
        margin-bottom: 1.5rem;
    }

    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .metric-card h3 {
        font-size: 0.85rem;
        margin: 0;
        opacity: 0.85;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card h2 {
        font-size: 1.8rem;
        margin: 0.3rem 0 0 0;
        font-weight: 700;
    }

    /* Risk tier badges */
    .risk-high {
        background: #ef4444;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .risk-medium {
        background: #f59e0b;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .risk-low {
        background: #10b981;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def check_api_health() -> dict | None:
    """Check if the prediction API is reachable and return health status."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception:
        return None
    return None


def predict_customer_feast(customer_id: str) -> dict | None:
    """Call the /predict/customer/{customer_id} endpoint and return the response."""
    try:
        response = requests.get(
            f"{API_URL}/predict/customer/{customer_id}",
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            st.error(f"Customer '{customer_id}' not found in the Feast online store.")
            return None
        else:
            st.error(f"Prediction failed: {response.status_code} — {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot connect to the prediction API at {API_URL}. "
            "Start the API with `make serve` or `docker compose up`."
        )
        return None
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


def get_risk_badge(risk_tier: str) -> str:
    """Return HTML badge for the risk tier."""
    tier_map = {
        "HIGH_RISK": '<span class="risk-high">🔴 HIGH RISK</span>',
        "MEDIUM_RISK": '<span class="risk-medium">🟡 MEDIUM RISK</span>',
        "LOW_RISK": '<span class="risk-low">🟢 LOW RISK</span>',
    }
    return tier_map.get(risk_tier, risk_tier)


def predict_single(customer_data: dict) -> dict | None:
    """Call the /predict endpoint and return the response."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=customer_data,
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Prediction failed: {response.status_code} — {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot connect to the prediction API at {API_URL}. "
            "Start the API with `make serve` or `docker compose up`."
        )
        return None
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


def predict_batch(customers: list[dict]) -> dict | None:
    """Call the /predict/batch endpoint and return the response."""
    try:
        response = requests.post(
            f"{API_URL}/predict/batch",
            json={"customers": customers},
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(
                f"Batch prediction failed: {response.status_code} — {response.text}"
            )
            return None
    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot connect to the prediction API at {API_URL}. "
            "Start the API with `make serve` or `docker compose up`."
        )
        return None
    except Exception as e:
        st.error(f"Batch prediction error: {e}")
        return None


# ---------------------------------------------------------------------------
# Sidebar — API status and model info
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🔗 API Status")

    health = check_api_health()
    if health is not None:
        st.success(f"✅ Connected — Status: {health.get('status', 'unknown')}")
        st.caption(f"Model loaded: {health.get('model_loaded', False)}")
        st.caption(f"API version: {health.get('version', 'unknown')}")
    else:
        st.error(f"❌ API unreachable at {API_URL}")
        st.caption("Start the API: `make serve`")

    st.markdown("---")
    st.markdown("## ℹ️ About")
    st.markdown(
        """
        **Customer Churn Predictor**

        Predicts customer churn probability using
        a LightGBM model with 28 engineered features.

        Built with:
        - 🐍 Python + scikit-learn
        - 🚀 FastAPI serving layer
        - 📊 MLflow experiment tracking
        - 🐳 Docker containerization
        - 🔄 GitHub Actions CI/CD
        """
    )


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.markdown(
    '<h1 class="main-header">📊 Customer Churn Predictor</h1>', unsafe_allow_html=True
)

(
    tab_feast,
    tab_predict,
    tab_batch,
    tab_performance,
    tab_drift,
    tab_uplift,
    tab_business,
) = st.tabs(
    [
        "⚡ Feast Online Predict",
        "🎯 Predict (Manual Form)",
        "📋 Batch Analysis",
        "📈 Model Performance",
        "📉 Data Drift",
        "🏥 Causal Uplift",
        "💰 Business Impact",
    ]
)

# ---------------------------------------------------------------------------
# Tab 0: Feast Online Store Prediction
# ---------------------------------------------------------------------------

with tab_feast:
    st.markdown("### Bleeding-Edge 2026 MLOps: Feast Feature Store Integration")
    st.caption(
        "Enter a `CustomerID` to fetch their real-time features from Feast SQLite online store in milliseconds, then immediately predict churn."
    )

    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        fe_customer_id = st.text_input(
            "Customer ID",
            value="7590-VHVEG",
            help="Try: 7590-VHVEG, 5575-GNVDE, or 3668-QPYBK",
        )
        predict_btn = st.button(
            "⚡ Fetch & Predict", type="primary", use_container_width=True
        )

    if predict_btn and fe_customer_id:
        with st.spinner("Fetching from Feast and predicting..."):
            result = predict_customer_feast(fe_customer_id)

        if result is not None:
            st.markdown("---")
            st.markdown(f"### Prediction Result for `{fe_customer_id}`")

            # Display metrics in cards
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                proba = result["churn_probability"]
                st.markdown(
                    f"""<div class="metric-card">
                        <h3>Churn Probability</h3>
                        <h2>{proba:.1%}</h2>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with r2:
                risk = result["risk_tier"]
                st.markdown(
                    f"""<div class="metric-card">
                        <h3>Risk Tier</h3>
                        <h2>{get_risk_badge(risk)}</h2>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with r3:
                churn = "YES 🔴" if result["will_churn"] else "NO 🟢"
                st.markdown(
                    f"""<div class="metric-card">
                        <h3>Will Churn?</h3>
                        <h2>{churn}</h2>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with r4:
                threshold = result["threshold_used"]
                st.markdown(
                    f"""<div class="metric-card">
                        <h3>Threshold Used</h3>
                        <h2>{threshold:.2f}</h2>
                    </div>""",
                    unsafe_allow_html=True,
                )

            # Display SHAP explanation if available
            if result.get("explanation"):
                st.markdown("---")
                st.markdown("#### 🧠 Top Churn Drivers")
                for feat, val in result["explanation"].items():
                    direction = "⬆️ Increases Risk" if val > 0 else "⬇️ Decreases Risk"
                    st.markdown(f"- **{feat}**: {val:+.4f} ({direction})")

            # Recommendation based on risk tier
            st.markdown("---")
            if risk == "HIGH_RISK":
                st.error("⚠️ **Immediate Action Required**")
            elif risk == "MEDIUM_RISK":
                st.warning("🔔 **Monitor Closely**")
            else:
                st.success("✅ **Low Risk**")

# ---------------------------------------------------------------------------
# Tab 1: Single Customer Prediction (Manual Form)
# ---------------------------------------------------------------------------

with tab_predict:
    st.markdown("### Predict Churn for a Single Customer")
    st.caption("Fill in the customer's features below and click **Predict**.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", ["Male", "Female"], key="pred_gender")
        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            key="pred_senior",
        )
        partner = st.selectbox("Partner", ["Yes", "No"], key="pred_partner")
        dependents = st.selectbox("Dependents", ["Yes", "No"], key="pred_deps")

        st.markdown("**Account**")
        tenure = st.slider("Tenure (months)", 0, 72, 12, key="pred_tenure")
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
            key="pred_contract",
        )

    with col2:
        st.markdown("**Services**")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"], key="pred_phone")
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"],
            key="pred_lines",
        )
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
            key="pred_internet",
        )
        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"],
            key="pred_security",
        )
        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"],
            key="pred_backup",
        )
        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"],
            key="pred_device",
        )

    with col3:
        st.markdown("**More Services**")
        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"],
            key="pred_tech",
        )
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"],
            key="pred_tv",
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"],
            key="pred_movies",
        )

        st.markdown("**Billing**")
        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"], key="pred_paperless"
        )
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            key="pred_payment",
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)", 18.0, 120.0, 70.35, step=0.05, key="pred_monthly"
        )
        total_charges = st.number_input(
            "Total Charges ($)", 0.0, 10000.0, 844.20, step=0.10, key="pred_total"
        )

    if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
        customer_data = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        with st.spinner("Running prediction..."):
            result = predict_single(customer_data)

        if result is not None:
            st.markdown("---")
            st.markdown("### Prediction Result")

            # Display metrics in cards
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                proba = result["churn_probability"]
                st.markdown(
                    f"""<div class="metric-card">
                        <h3>Churn Probability</h3>
                        <h2>{proba:.1%}</h2>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with r2:
                risk = result["risk_tier"]
                st.markdown(
                    f"""<div class="metric-card">
                        <h3>Risk Tier</h3>
                        <h2>{get_risk_badge(risk)}</h2>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with r3:
                churn = "YES 🔴" if result["will_churn"] else "NO 🟢"
                st.markdown(
                    f"""<div class="metric-card">
                        <h3>Will Churn?</h3>
                        <h2>{churn}</h2>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with r4:
                threshold = result["threshold_used"]
                st.markdown(
                    f"""<div class="metric-card">
                        <h3>Threshold Used</h3>
                        <h2>{threshold:.2f}</h2>
                    </div>""",
                    unsafe_allow_html=True,
                )

            # Display SHAP explanation if available
            if result.get("explanation"):
                st.markdown("---")
                st.markdown("#### 🧠 Top Churn Drivers")
                for feat, val in result["explanation"].items():
                    direction = "⬆️ Increases Risk" if val > 0 else "⬇️ Decreases Risk"
                    st.markdown(f"- **{feat}**: {val:+.4f} ({direction})")

            # Recommendation based on risk tier
            st.markdown("---")
            if risk == "HIGH_RISK":
                st.error(
                    "⚠️ **Immediate Action Required** — "
                    "This customer is at high risk of "
                    "churning. Recommend: personal outreach call,  "
                    "retention offer (contract upgrade discount), "
                    "and service review."
                )
            elif risk == "MEDIUM_RISK":
                st.warning(
                    "🔔 **Monitor Closely** — This customer shows moderate churn risk. "
                    "Recommend: automated retention email campaign, usage pattern "
                    "monitoring, and proactive support check-in."
                )
            else:
                st.success(
                    "✅ **Low Risk** — This customer appears satisfied and committed. "
                    "Recommend: standard engagement, loyalty rewards, and "
                    "cross-sell/upsell opportunities."
                )


# ---------------------------------------------------------------------------
# Tab 2: Batch Analysis
# ---------------------------------------------------------------------------

with tab_batch:
    st.markdown("### Batch Customer Churn Analysis")
    st.caption(
        "Upload a CSV file with customer features to predict churn for multiple "
        "customers at once. Maximum 100 customers per batch."
    )

    # Provide a downloadable template
    template_cols = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
    ]
    template_df = pd.DataFrame(columns=template_cols)
    template_csv = template_df.to_csv(index=False)

    st.download_button(
        "📥 Download CSV Template",
        data=template_csv,
        file_name="churn_prediction_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        "Upload customer data CSV",
        type=["csv"],
        help="CSV must contain all 19 feature columns. See template.",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.markdown(f"**Uploaded:** {len(df)} customers, {len(df.columns)} columns")

            if len(df) > 100:
                st.warning("Maximum 100 customers per batch. Truncating to first 100.")
                df = df.head(100)

            # Validate columns
            missing_cols = set(template_cols) - set(df.columns)
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
            else:
                if st.button("🚀 Run Batch Prediction", type="primary"):
                    customers = df.to_dict(orient="records")

                    with st.spinner(
                        f"Predicting churn for {len(customers)} customers..."
                    ):
                        result = predict_batch(customers)

                    if result is not None:
                        st.markdown("---")
                        st.markdown("### Batch Results")

                        # Summary metrics
                        s1, s2, s3, s4 = st.columns(4)
                        with s1:
                            st.metric("Total Customers", result["total_customers"])
                        with s2:
                            st.metric("🔴 High Risk", result["high_risk_count"])
                        with s3:
                            st.metric("🟡 Medium Risk", result["medium_risk_count"])
                        with s4:
                            st.metric("🟢 Low Risk", result["low_risk_count"])

                        # Build results DataFrame
                        predictions = result["predictions"]
                        results_df = df.copy()
                        results_df["churn_probability"] = [
                            p["churn_probability"] for p in predictions
                        ]
                        results_df["risk_tier"] = [p["risk_tier"] for p in predictions]
                        results_df["will_churn"] = [
                            p["will_churn"] for p in predictions
                        ]

                        # Sort by churn probability (highest risk first)
                        results_df = results_df.sort_values(
                            "churn_probability", ascending=False
                        )

                        st.dataframe(
                            results_df,
                            use_container_width=True,
                            height=400,
                        )

                        # Download results
                        csv_output = results_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results CSV",
                            data=csv_output,
                            file_name="churn_predictions_results.csv",
                            mime="text/csv",
                        )

        except Exception as e:
            st.error(f"Error reading CSV: {e}")


# ---------------------------------------------------------------------------
# Tab 3: Model Performance
# ---------------------------------------------------------------------------

with tab_performance:
    st.markdown("### Model Performance Dashboard")
    st.caption("Training and evaluation metrics from the latest model training run.")

    # Display report images if they exist
    report_images = {
        "Confusion Matrices": "confusion_matrices.png",
        "Feature Importance (Top 20)": "feature_importance.png",
        "Threshold Sensitivity Analysis": "threshold_sensitivity.png",
        "PR-AUC: Cross-Validation vs Test": "cv_vs_test_pr_auc.png",
        "Precision-Recall Trade-off": "precision_recall_scatter.png",
        "Metrics Heatmap (All Algorithms)": "metrics_heatmap.png",
    }

    found_any = False
    for title, filename in report_images.items():
        img_path = REPORTS_DIR / filename
        if img_path.exists():
            found_any = True
            st.markdown(f"#### {title}")
            st.image(str(img_path), use_container_width=True)
            st.markdown("---")

    if not found_any:
        st.info(
            "📭 No report images found in `reports/`. "
            "Run the training notebooks (notebooks/03c_champion_evaluation.py) "
            "to generate performance visualizations."
        )

    # Additional EDA images section
    with st.expander("📊 EDA Visualizations", expanded=False):
        eda_images = {
            "Churn Distribution": "eda_01_churn_distribution.png",
            "Churn by Contract Type": "eda_02_churn_by_contract.png",
            "Tenure Analysis": "eda_03_tenure_analysis.png",
            "Monthly Charges Distribution": "eda_04_monthly_charges.png",
            "Internet Service & Churn": "eda_05_internet_service_churn.png",
            "Service Adoption & Churn": "eda_06_service_adoption_churn.png",
            "Contract × Internet Heatmap": "eda_07_contract_internet_heatmap.png",
        }

        for title, filename in eda_images.items():
            img_path = REPORTS_DIR / filename
            if img_path.exists():
                st.markdown(f"**{title}**")
                st.image(str(img_path), use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4: Visual Data Drift
# ---------------------------------------------------------------------------

with tab_drift:
    st.markdown("### Data Drift Monitoring")
    st.caption("Visualizing the latest data drift report from Evidently AI.")

    drift_report_path = REPORTS_DIR / "drift_report.json"
    if drift_report_path.exists():
        try:
            import json

            with open(drift_report_path, "r") as f:
                drift_data = json.load(f)

            metrics = drift_data.get("metrics", [])
            dataset_drift = None
            drift_table = None

            for m in metrics:
                if m.get("metric") == "DatasetDriftMetric":
                    dataset_drift = m.get("result", {})
                elif m.get("metric") == "DataDriftTable":
                    drift_table = m.get("result", {}).get("drift_by_columns", {})

            if dataset_drift:
                st.markdown("#### Dataset Drift Summary")
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    is_drifted = dataset_drift.get("dataset_drift", False)
                    status_color = "#ef4444" if is_drifted else "#10b981"
                    status_text = "DRIFT DETECTED" if is_drifted else "ALL CLEAR"
                    st.markdown(
                        f"<h3 style='color: {status_color};'>{status_text}</h3>",
                        unsafe_allow_html=True,
                    )
                with col_d2:
                    st.metric(
                        "Drifted Features",
                        f"{dataset_drift.get('number_of_drifted_features', 0)} / {dataset_drift.get('number_of_columns', 0)}",
                    )

            if drift_table:
                df_drift = pd.DataFrame(
                    [
                        {
                            "Feature": feat,
                            "Drift Score": round(float(info.get("drift_score", 0)), 4),
                            "Drift Detected": info.get("drift_detected", False),
                            "Stat Test": info.get("stat_test_name", ""),
                        }
                        for feat, info in drift_table.items()
                    ]
                )
                df_drift = df_drift.sort_values(
                    "Drift Score", ascending=False
                ).reset_index(drop=True)

                st.markdown("#### Feature-Level Drift")

                def color_boolean(val):
                    color = "red" if val is True else "green" if val is False else ""
                    return f"color: {color}; font-weight: bold;"

                st.dataframe(
                    df_drift.style.map(color_boolean, subset=["Drift Detected"]),
                    use_container_width=True,
                    height=500,
                )
            else:
                st.info("No feature-level drift table found in the report.")

        except Exception as e:
            st.error(f"Failed to load drift report: {e}")
    else:
        st.info(
            "📭 No drift report found. Run `scripts/generate_drift_report.py` to generate one."
        )


# ---------------------------------------------------------------------------
# Tab 5: Causal Uplift Modeling
# ---------------------------------------------------------------------------

with tab_uplift:
    st.markdown("### 🏥 Prescriptive Analytics: Causal Uplift Modeling")
    st.caption(
        "Moving beyond 'Who will churn' to 'Who will change their mind if we intervene?'. Models the Conditional Average Treatment Effect (CATE)."
    )

    uplift_report_path = REPORTS_DIR / "uplift_report.json"
    if uplift_report_path.exists():
        try:
            import json

            with open(uplift_report_path, "r") as f:
                uplift_data = json.load(f)

            st.markdown(
                f"**Target Population**: {uplift_data.get('eligible_customers', 0)} customers analyzed for 'TechSupport' intervention."
            )

            segs = uplift_data.get("segments", {})
            cates = uplift_data.get("avg_cate_by_segment", {})

            # Display metrics in cards
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.markdown(
                    f"""<div class="metric-card" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
                        <h3>Persuadables (TARGET)</h3>
                        <h2>{segs.get('Persuadable (Target)', 0)}</h2>
                        <h4 style="font-size:0.8rem; margin-top:5px;">Avg Effect: {cates.get('Persuadable (Target)', 0):.1%}</h4>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with s2:
                st.markdown(
                    f"""<div class="metric-card" style="background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);">
                        <h3>Sure Things (IGNORE)</h3>
                        <h2>{segs.get('Sure Thing (Ignore)', 0)}</h2>
                        <h4 style="font-size:0.8rem; margin-top:5px;">Avg Effect: {cates.get('Sure Thing (Ignore)', 0):.1%}</h4>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with s3:
                st.markdown(
                    f"""<div class="metric-card" style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);">
                        <h3>Lost Causes (IGNORE)</h3>
                        <h2>{segs.get('Lost Cause (Ignore)', 0)}</h2>
                        <h4 style="font-size:0.8rem; margin-top:5px;">Avg Effect: {cates.get('Lost Cause (Ignore)', 0):.1%}</h4>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with s4:
                st.markdown(
                    f"""<div class="metric-card" style="background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);">
                        <h3>Sleeping Dogs (DANGER)</h3>
                        <h2>{segs.get('Sleeping Dog (Do Not Disturb)', 0)}</h2>
                        <h4 style="font-size:0.8rem; margin-top:5px;">Avg Effect: +{cates.get('Sleeping Dog (Do Not Disturb)', 0):.1%}</h4>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            st.markdown(
                """
            #### 🧠 Causal ML Interpretation
            *   **Persuadables**: These customers have a high base churn risk, but the model proves that giving them TechSupport heavily *reduces* their risk. **Give them the offer.**
            *   **Sure Things**: They are not at risk of churning anyway. Giving them free TechSupport is a waste of money.
            *   **Lost Causes**: High churn risk, but TechSupport won't save them. The money is better spent elsewhere.
            *   **Sleeping Dogs**: Very dangerous. They have low base risk, but actively reaching out triggers them to evaluate their plan and churn. **Do not contact.**
            """
            )

        except Exception as e:
            st.error(f"Failed to load uplift report: {e}")
    else:
        st.info(
            "📭 No causal uplift report found. Run `scripts/train_uplift.py` to generate prescriptive modeling insights."
        )

# ---------------------------------------------------------------------------
# Tab 6: Business Impact
# ---------------------------------------------------------------------------

with tab_business:
    st.markdown("### Business Impact Analysis")
    st.caption(
        "Demonstrates the revenue impact of using the churn prediction model "
        "vs. having no model at all."
    )

    # Business metrics calculator
    st.markdown("#### 💰 Cost-Benefit Calculator")
    st.markdown(
        "Adjust the sliders to see how the model's predicted performance "
        "translates to real dollar savings."
    )

    b1, b2 = st.columns(2)

    with b1:
        fn_cost = st.slider(
            "Cost per missed churner (FN) $",
            100,
            2000,
            500,
            step=50,
            help="Revenue lost when we fail to identify a churning customer.",
        )
        fp_cost = st.slider(
            "Cost per false alarm (FP) $",
            5,
            200,
            20,
            step=5,
            help="""Cost of a retention offer sent to a customer
            who wasn't going to churn.""",
        )

    with b2:
        total_customers = st.slider(
            "Total customers in cohort",
            1000,
            50000,
            7043,
            step=100,
        )
        churn_rate = (
            st.slider(
                "Expected churn rate (%)",
                5,
                50,
                27,
                step=1,
            )
            / 100.0
        )

    # Assume model performance metrics
    recall = (
        st.slider(
            "Model recall (% of churners caught)",
            50,
            100,
            78,
            step=1,
            help="From the model's test set evaluation.",
        )
        / 100.0
    )
    precision = (
        st.slider(
            "Model precision (% of predictions correct)",
            30,
            100,
            55,
            step=1,
        )
        / 100.0
    )

    # Calculate business impact
    n_churners = int(total_customers * churn_rate)
    n_retained = int(total_customers * (1 - churn_rate))

    # No model baseline: miss ALL churners
    baseline_cost = n_churners * fn_cost

    # With model:
    true_positives = int(n_churners * recall)
    false_negatives = n_churners - true_positives
    # From precision: TP / (TP + FP) = precision → FP = TP * (1/precision - 1)
    false_positives = int(true_positives * (1 / precision - 1)) if precision > 0 else 0

    model_cost = (false_negatives * fn_cost) + (false_positives * fp_cost)
    savings = baseline_cost - model_cost
    roi_percent = (savings / model_cost * 100) if model_cost > 0 else 0

    st.markdown("---")
    st.markdown("#### 📊 Impact Summary")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Baseline Cost (No Model)", f"${baseline_cost:,.0f}")
    with m2:
        st.metric("Cost With Model", f"${model_cost:,.0f}")
    with m3:
        st.metric("💵 Estimated Savings", f"${savings:,.0f}", delta=f"{savings:+,.0f}")
    with m4:
        st.metric("ROI", f"{roi_percent:,.0f}%")

    st.markdown("---")

    # Confusion matrix breakdown
    with st.expander("📋 Detailed Breakdown", expanded=True):
        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            st.markdown("**Prediction Outcomes**")
            st.markdown(f"- ✅ Churners caught (TP): **{true_positives:,}**")
            st.markdown(f"- ❌ Churners missed (FN): **{false_negatives:,}**")
            st.markdown(f"- ⚠️ False alarms (FP): **{false_positives:,}**")

        with detail_col2:
            st.markdown("**Cost Breakdown**")
            st.markdown(
                f"- FN cost (missed churners): **${false_negatives * fn_cost:,.0f}**"
            )
            st.markdown(
                f"- FP cost (wasted offers): **${false_positives * fp_cost:,.0f}**"
            )
            st.markdown(f"- **Total model cost: ${model_cost:,.0f}**")

    # Display business impact image if available
    biz_img = REPORTS_DIR / "business_impact.png"
    if biz_img.exists():
        st.markdown("#### Training Run Business Impact")
        st.image(str(biz_img), use_container_width=True)
