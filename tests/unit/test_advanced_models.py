from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.models.fairness_audit import (
    compute_demographic_parity,
    compute_equalized_odds,
    run_fairness_audit,
)
from src.models.learning_curves import compute_learning_curves
from src.models.statistical_comparison import compare_models, corrected_cv_ttest
from src.models.train import _log_metrics_to_mlflow
from src.models.uplift import ChurnUpliftModel


def test_fairness_demographic_parity():
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    res = compute_demographic_parity(y_pred, groups, "group")
    assert res.parity_ratio == 0.5
    assert not res.passes_four_fifths


def test_fairness_equalized_odds():
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    res = compute_equalized_odds(y_true, y_pred, groups, "group")
    assert res.tpr_disparity == 0.5
    assert res.fpr_disparity == 0.5


def test_run_fairness_audit():
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    y_prob = np.array([0.9, 0.4, 0.6, 0.2, 0.8, 0.7, 0.1, 0.3])
    sensitive_features = {"group": np.array(["A", "A", "A", "A", "B", "B", "B", "B"])}
    report = run_fairness_audit(y_true, y_pred, y_prob, sensitive_features)
    assert not report.overall_fair
    assert "group" in report.demographic_parity


def test_learning_curves():
    X = pd.DataFrame(np.random.rand(100, 5))
    y = pd.Series(np.random.randint(0, 2, 100))
    pipeline = Pipeline([("clf", DummyClassifier(strategy="prior"))])
    report = compute_learning_curves(
        pipeline, X, y, train_sizes=[0.5, 1.0], cv_folds=2, scoring="roc_auc"
    )
    assert len(report.train_sizes_abs) == 2
    assert report.total_samples == 100


def test_statistical_comparison():
    scores_a = np.array([0.9, 0.85, 0.88, 0.92, 0.87])
    scores_b = np.array([0.8, 0.75, 0.78, 0.82, 0.77])
    res = corrected_cv_ttest(scores_a, scores_b, n_train=80, n_test=20)
    assert res.p_value >= 0.0

    fold_scores = {"ModelA": scores_a, "ModelB": scores_b}
    report = compare_models(fold_scores, n_train=80, n_test=20)
    assert report.n_comparisons == 1


def test_churn_uplift_model():
    model = ChurnUpliftModel(random_state=42)
    X = pd.DataFrame(np.random.rand(100, 5))
    T = pd.Series(np.random.randint(0, 2, 100))
    y = pd.Series(np.random.randint(0, 2, 100))

    model.fit(X, T, y)
    assert model.is_fitted

    cate = model.predict_uplift(X)
    assert len(cate) == 100

    base_churn_prob = pd.Series(np.random.rand(100))
    segments = model.segment_customers(X, cate, base_churn_prob)
    assert "uplift_segment" in segments.columns
    assert len(segments) == 100


@patch("src.models.train.mlflow")
def test_log_metrics_to_mlflow(mock_mlflow):
    metrics = {"roc_auc": 0.85, "pr_auc": 0.6}
    _log_metrics_to_mlflow(metrics, prefix="test_")
    assert mock_mlflow.log_metric.call_count == 2
