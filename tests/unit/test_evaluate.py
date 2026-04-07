import numpy as np
import pytest

from src.models.evaluate import compute_business_metrics, compute_ml_metrics, evaluate


@pytest.fixture
def perfect_predictions():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.1, 0.95])
    return y_true, y_proba


@pytest.fixture
def realistic_predictions():
    """Simulates a real model output with some errors."""
    np.random.seed(42)
    y_true = np.array([0] * 70 + [1] * 30)
    y_proba = np.concatenate(
        [
            np.random.beta(2, 8, 70),  # retained: mostly low probability
            np.random.beta(6, 4, 30),  # churners: mostly high probability
        ]
    )
    return y_true, y_proba


class TestComputeMlMetrics:

    def test_returns_dict(self, perfect_predictions):
        y_true, y_proba = perfect_predictions
        result = compute_ml_metrics(y_true, y_proba)
        assert isinstance(result, dict)

    def test_contains_required_keys(self, perfect_predictions):
        y_true, y_proba = perfect_predictions
        result = compute_ml_metrics(y_true, y_proba)
        for key in ["roc_auc", "pr_auc", "f1", "precision", "recall", "threshold"]:
            assert key in result

    def test_perfect_model_roc_auc_is_one(self, perfect_predictions):
        y_true, y_proba = perfect_predictions
        result = compute_ml_metrics(y_true, y_proba)
        assert result["roc_auc"] == 1.0

    def test_threshold_stored_in_output(self, perfect_predictions):
        y_true, y_proba = perfect_predictions
        result = compute_ml_metrics(y_true, y_proba, threshold=0.3)
        assert result["threshold"] == 0.3

    def test_all_values_between_0_and_1(self, realistic_predictions):
        y_true, y_proba = realistic_predictions
        result = compute_ml_metrics(y_true, y_proba)
        for key in ["roc_auc", "pr_auc", "f1", "precision", "recall"]:
            assert 0.0 <= result[key] <= 1.0

    def test_lower_threshold_increases_recall(self, realistic_predictions):
        y_true, y_proba = realistic_predictions
        high_t = compute_ml_metrics(y_true, y_proba, threshold=0.7)
        low_t = compute_ml_metrics(y_true, y_proba, threshold=0.2)
        assert low_t["recall"] >= high_t["recall"]

    def test_higher_threshold_increases_precision(self, realistic_predictions):
        y_true, y_proba = realistic_predictions
        high_t = compute_ml_metrics(y_true, y_proba, threshold=0.7)
        low_t = compute_ml_metrics(y_true, y_proba, threshold=0.2)
        assert high_t["precision"] >= low_t["precision"]


class TestComputeBusinessMetrics:

    def test_returns_dict(self, perfect_predictions):
        y_true, y_proba = perfect_predictions
        y_pred = (y_proba >= 0.5).astype(int)
        result = compute_business_metrics(y_true, y_pred)
        assert isinstance(result, dict)

    def test_contains_required_keys(self, perfect_predictions):
        y_true, y_proba = perfect_predictions
        y_pred = (y_proba >= 0.5).astype(int)
        result = compute_business_metrics(y_true, y_pred)
        for key in [
            "true_positives",
            "false_positives",
            "false_negatives",
            "true_negatives",
            "cost_of_fn",
            "cost_of_fp",
            "total_cost",
            "estimated_savings",
        ]:
            assert key in result

    def test_perfect_predictions_zero_cost(self, perfect_predictions):
        y_true, y_proba = perfect_predictions
        y_pred = (y_proba >= 0.5).astype(int)
        result = compute_business_metrics(y_true, y_pred)
        assert result["cost_of_fn"] == 0.0
        assert result["cost_of_fp"] == 0.0

    def test_savings_non_negative_for_decent_model(self, realistic_predictions):
        y_true, y_proba = realistic_predictions
        y_pred = (y_proba >= 0.5).astype(int)
        result = compute_business_metrics(y_true, y_pred)
        assert result["estimated_savings"] >= 0

    def test_cm_counts_sum_to_total_rows(self, realistic_predictions):
        y_true, y_proba = realistic_predictions
        y_pred = (y_proba >= 0.5).astype(int)
        result = compute_business_metrics(y_true, y_pred)
        total = (
            result["true_positives"]
            + result["false_positives"]
            + result["false_negatives"]
            + result["true_negatives"]
        )
        assert total == len(y_true)


class TestEvaluate:

    def test_returns_combined_dict(self, realistic_predictions):
        y_true, y_proba = realistic_predictions
        result = evaluate(y_true, y_proba)
        assert "roc_auc" in result
        assert "estimated_savings" in result

    def test_no_overlap_lost_between_ml_and_business(self, realistic_predictions):
        y_true, y_proba = realistic_predictions
        result = evaluate(y_true, y_proba)
        assert len(result) >= 14
