import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.models.pipeline import build_pipeline, get_preprocessor


@pytest.fixture
def minimal_X() -> pd.DataFrame:
    """Minimal valid feature DataFrame post-preprocess (2 rows)."""
    return pd.DataFrame(
        [
            {
                "gender": "Male",
                "SeniorCitizen": "0",
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 844.20,
            },
            {
                "gender": "Female",
                "SeniorCitizen": "1",
                "Partner": "No",
                "Dependents": "No",
                "tenure": 60,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 45.0,
                "TotalCharges": 2700.0,
            },
        ]
    )


@pytest.fixture
def minimal_y() -> pd.Series:
    return pd.Series([1, 0], name="Churn")


class TestGetPreprocessor:

    def test_returns_column_transformer(self):
        result = get_preprocessor()
        assert isinstance(result, ColumnTransformer)

    def test_has_three_transformers(self):
        result = get_preprocessor()
        assert len(result.transformers) == 3

    def test_transformer_names(self):
        result = get_preprocessor()
        names = [t[0] for t in result.transformers]
        assert "cat" in names
        assert "num" in names
        assert "bin" in names


class TestBuildPipeline:

    def test_returns_pipeline(self):
        result = build_pipeline()
        assert isinstance(result, Pipeline)

    def test_has_three_steps(self):
        result = build_pipeline()
        assert len(result.steps) == 3

    def test_step_names(self):
        result = build_pipeline()
        names = [s[0] for s in result.steps]
        assert names == ["feature_engineering", "preprocessor", "classifier"]

    def test_param_override_applied(self):
        pipeline = build_pipeline(params={"n_estimators": 10})
        clf = pipeline.named_steps["classifier"]
        assert clf.n_estimators == 10

    def test_pipeline_can_fit_and_predict(self, minimal_X, minimal_y):
        pipeline = build_pipeline(params={"n_estimators": 5})
        pipeline.fit(minimal_X, minimal_y)
        predictions = pipeline.predict(minimal_X)
        assert len(predictions) == 2
        assert set(predictions).issubset({0, 1})

    def test_predict_proba_returns_two_columns(self, minimal_X, minimal_y):
        pipeline = build_pipeline(params={"n_estimators": 5})
        pipeline.fit(minimal_X, minimal_y)
        probas = pipeline.predict_proba(minimal_X)
        assert probas.shape == (2, 2)

    def test_probabilities_sum_to_one(self, minimal_X, minimal_y):
        pipeline = build_pipeline(params={"n_estimators": 5})
        pipeline.fit(minimal_X, minimal_y)
        probas = pipeline.predict_proba(minimal_X)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)

    def test_unfitted_pipeline_has_no_classes(self):
        pipeline = build_pipeline()
        assert not hasattr(pipeline.named_steps["classifier"], "classes_")
