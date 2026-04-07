import numpy as np
import pytest

from src.models.threshold import (
    find_cost_optimal_threshold,
    find_f1_optimal_threshold,
    get_risk_tier,
)


@pytest.fixture
def imbalanced_predictions():
    np.random.seed(0)
    y_true = np.array([0] * 70 + [1] * 30)
    y_proba = np.concatenate(
        [
            np.random.beta(2, 8, 70),
            np.random.beta(6, 4, 30),
        ]
    )
    return y_true, y_proba


class TestFindCostOptimalThreshold:

    def test_returns_float(self, imbalanced_predictions):
        y_true, y_proba = imbalanced_predictions
        result = find_cost_optimal_threshold(y_true, y_proba)
        assert isinstance(result, float)

    def test_threshold_between_0_and_1(self, imbalanced_predictions):
        y_true, y_proba = imbalanced_predictions
        result = find_cost_optimal_threshold(y_true, y_proba)
        assert 0.0 < result < 1.0

    def test_cost_threshold_lower_than_default(self, imbalanced_predictions):
        # With FN >> FP in cost, the optimal threshold should be
        # below 0.5 to maximise recall
        y_true, y_proba = imbalanced_predictions
        result = find_cost_optimal_threshold(y_true, y_proba)
        assert result <= 0.5

    def test_rounded_to_two_decimals(self, imbalanced_predictions):
        y_true, y_proba = imbalanced_predictions
        result = find_cost_optimal_threshold(y_true, y_proba)
        assert result == round(result, 2)


class TestFindF1OptimalThreshold:

    def test_returns_float(self, imbalanced_predictions):
        y_true, y_proba = imbalanced_predictions
        result = find_f1_optimal_threshold(y_true, y_proba)
        assert isinstance(result, float)

    def test_threshold_between_0_and_1(self, imbalanced_predictions):
        y_true, y_proba = imbalanced_predictions
        result = find_f1_optimal_threshold(y_true, y_proba)
        assert 0.0 < result < 1.0

    def test_rounded_to_two_decimals(self, imbalanced_predictions):
        y_true, y_proba = imbalanced_predictions
        result = find_f1_optimal_threshold(y_true, y_proba)
        assert result == round(result, 2)


class TestGetRiskTier:

    def test_high_probability_is_high_risk(self):
        assert get_risk_tier(0.85) == "HIGH_RISK"

    def test_medium_probability_is_medium_risk(self):
        assert get_risk_tier(0.45) == "MEDIUM_RISK"

    def test_low_probability_is_low_risk(self):
        assert get_risk_tier(0.10) == "LOW_RISK"

    def test_exactly_at_high_threshold_is_high_risk(self):
        # 0.60 is the default high_risk_threshold
        assert get_risk_tier(0.60) == "HIGH_RISK"

    def test_just_below_high_threshold_is_medium_risk(self):
        assert get_risk_tier(0.59) == "MEDIUM_RISK"

    def test_exactly_at_medium_threshold_is_medium_risk(self):
        assert get_risk_tier(0.35) == "MEDIUM_RISK"

    def test_just_below_medium_threshold_is_low_risk(self):
        assert get_risk_tier(0.34) == "LOW_RISK"

    def test_zero_probability_is_low_risk(self):
        assert get_risk_tier(0.0) == "LOW_RISK"

    def test_returns_string(self):
        assert isinstance(get_risk_tier(0.5), str)
