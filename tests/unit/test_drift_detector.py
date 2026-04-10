"""
Unit tests for the data drift detection module.

Tests cover:
    - PSI computation (identical, shifted, and edge-case distributions)
    - Chi-squared categorical drift detection
    - Prediction drift computation
    - Full report generation
    - Threshold-based alerting

Each test uses small synthetic DataFrames to keep execution fast (<1s total).
"""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_detector import DriftDetector, DriftReport, FeatureDriftResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reference_df() -> pd.DataFrame:
    """Create a reference DataFrame with known distributions."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame(
        {
            "tenure": np.random.randint(0, 72, n),
            "MonthlyCharges": np.random.uniform(18, 120, n),
            "TotalCharges": np.random.uniform(0, 8000, n),
            "Contract": np.random.choice(
                ["Month-to-month", "One year", "Two year"],
                n,
                p=[0.55, 0.25, 0.20],
            ),
            "InternetService": np.random.choice(
                ["DSL", "Fiber optic", "No"],
                n,
                p=[0.34, 0.44, 0.22],
            ),
            "gender": np.random.choice(["Male", "Female"], n),
        }
    )


@pytest.fixture
def identical_current_df(reference_df) -> pd.DataFrame:
    """Current data identical to reference — should detect NO drift."""
    return reference_df.copy()


@pytest.fixture
def shifted_current_df() -> pd.DataFrame:
    """Current data with obvious distributional shift — should detect drift."""
    np.random.seed(99)
    n = 500
    return pd.DataFrame(
        {
            # tenure shifted to much higher values
            "tenure": np.random.randint(50, 100, n),
            # MonthlyCharges shifted to much higher values
            "MonthlyCharges": np.random.uniform(80, 200, n),
            # TotalCharges shifted dramatically
            "TotalCharges": np.random.uniform(5000, 20000, n),
            # Contract distribution changed (way more Two year)
            "Contract": np.random.choice(
                ["Month-to-month", "One year", "Two year"],
                n,
                p=[0.10, 0.20, 0.70],
            ),
            # InternetService distribution changed
            "InternetService": np.random.choice(
                ["DSL", "Fiber optic", "No"],
                n,
                p=[0.10, 0.80, 0.10],
            ),
            "gender": np.random.choice(["Male", "Female"], n),
        }
    )


@pytest.fixture
def detector(reference_df) -> DriftDetector:
    """DriftDetector initialized with the reference DataFrame."""
    return DriftDetector(
        reference_df,
        numerical_features=["tenure", "MonthlyCharges", "TotalCharges"],
        categorical_features=["Contract", "InternetService", "gender"],
    )


# ---------------------------------------------------------------------------
# PSI Tests
# ---------------------------------------------------------------------------


class TestComputePSI:
    """Tests for PSI computation on numerical features."""

    def test_identical_distributions_have_low_psi(self, detector, identical_current_df):
        """PSI should be near zero when distributions are identical."""
        results = detector.compute_psi(identical_current_df)
        assert len(results) > 0
        for result in results:
            assert result.drift_score < 0.10, (
                f"PSI for {result.feature_name} should be < 0.10 "
                f"for identical data, got {result.drift_score}"
            )
            assert result.is_drifted is False

    def test_shifted_distributions_have_high_psi(self, detector, shifted_current_df):
        """PSI should be high when distributions are obviously shifted."""
        results = detector.compute_psi(shifted_current_df)
        drifted = [r for r in results if r.is_drifted]
        # At least one numerical feature should trigger drift
        assert len(drifted) > 0, "Expected at least one drifted numerical feature"

    def test_psi_result_structure(self, detector, identical_current_df):
        """Each result should have the correct structure."""
        results = detector.compute_psi(identical_current_df)
        for result in results:
            assert isinstance(result, FeatureDriftResult)
            assert result.method == "psi"
            assert result.threshold > 0
            assert result.drift_score >= 0
            assert isinstance(result.details, dict)

    def test_psi_with_missing_feature_skips(self, detector):
        """Missing features in current data should be skipped, not error."""
        partial_df = pd.DataFrame({"tenure": [10, 20, 30]})
        results = detector.compute_psi(partial_df)
        feature_names = [r.feature_name for r in results]
        assert "tenure" in feature_names
        # MonthlyCharges and TotalCharges are missing — should be skipped
        assert "MonthlyCharges" not in feature_names

    def test_single_psi_computation(self):
        """Direct test of the static PSI computation method."""
        np.random.seed(42)
        ref = np.random.normal(50, 10, 1000)
        cur = np.random.normal(50, 10, 1000)
        psi = DriftDetector._compute_single_psi(ref, cur)
        assert psi < 0.10, f"PSI for same distribution should be low, got {psi}"

    def test_psi_detects_mean_shift(self):
        """PSI should detect a shift in the mean of a distribution."""
        np.random.seed(42)
        ref = np.random.normal(50, 10, 1000)
        cur = np.random.normal(80, 10, 1000)  # mean shifted from 50 to 80
        psi = DriftDetector._compute_single_psi(ref, cur)
        assert psi > 0.25, f"PSI should detect strong mean shift, got {psi}"

    def test_psi_constant_feature_returns_zero(self):
        """A constant feature in reference should return PSI = 0."""
        ref = np.array([5.0] * 100)
        cur = np.array([5.0] * 100)
        psi = DriftDetector._compute_single_psi(ref, cur)
        assert psi == 0.0


# ---------------------------------------------------------------------------
# Chi-Squared Tests
# ---------------------------------------------------------------------------


class TestComputeCategoricalDrift:
    """Tests for chi-squared categorical drift detection."""

    def test_identical_distributions_no_drift(self, detector, identical_current_df):
        """Identical categorical distributions should not trigger drift."""
        results = detector.compute_categorical_drift(identical_current_df)
        assert len(results) > 0
        for result in results:
            # p-value should be high (not significant) for identical data
            assert result.is_drifted is False, (
                f"Feature {result.feature_name} should not drift "
                f"with identical data, p={result.drift_score}"
            )

    def test_shifted_distributions_detect_drift(self, detector, shifted_current_df):
        """Shifted categorical distributions should trigger drift."""
        results = detector.compute_categorical_drift(shifted_current_df)
        drifted = [r for r in results if r.is_drifted]
        # Contract and InternetService are heavily shifted
        assert len(drifted) > 0, "Expected at least one drifted categorical feature"

    def test_chi_squared_result_structure(self, detector, identical_current_df):
        """Each result should have the correct structure."""
        results = detector.compute_categorical_drift(identical_current_df)
        for result in results:
            assert isinstance(result, FeatureDriftResult)
            assert result.method == "chi_squared"
            assert "chi2_statistic" in result.details
            assert "p_value" in result.details
            assert "n_categories" in result.details

    def test_missing_categorical_feature_skips(self, detector):
        """Missing categorical features should be skipped."""
        partial_df = pd.DataFrame({"Contract": ["Month-to-month"] * 10})
        results = detector.compute_categorical_drift(partial_df)
        feature_names = [r.feature_name for r in results]
        assert "Contract" in feature_names
        assert "InternetService" not in feature_names


# ---------------------------------------------------------------------------
# Prediction Drift Tests
# ---------------------------------------------------------------------------


class TestComputePredictionDrift:
    """Tests for prediction probability drift detection."""

    def test_identical_predictions_no_drift(self):
        """Identical prediction distributions should have zero drift."""
        ref_proba = np.array([0.2, 0.3, 0.5, 0.7])
        cur_proba = np.array([0.2, 0.3, 0.5, 0.7])
        drift = DriftDetector.compute_prediction_drift(ref_proba, cur_proba)
        assert drift == 0.0

    def test_shifted_predictions_detect_drift(self):
        """Shifted predictions should have non-zero drift."""
        ref_proba = np.array([0.2, 0.3, 0.2, 0.3])
        cur_proba = np.array([0.8, 0.9, 0.7, 0.85])
        drift = DriftDetector.compute_prediction_drift(ref_proba, cur_proba)
        assert drift > 0.15, f"Expected significant drift, got {drift}"

    def test_drift_is_relative(self):
        """Drift score should be relative to the reference mean."""
        ref_proba = np.array([0.5, 0.5, 0.5])
        # 20% relative shift up: mean goes from 0.5 to 0.6
        cur_proba = np.array([0.6, 0.6, 0.6])
        drift = DriftDetector.compute_prediction_drift(ref_proba, cur_proba)
        assert abs(drift - 0.2) < 0.01, f"Expected ~0.20 relative drift, got {drift}"

    def test_zero_reference_mean_returns_zero(self):
        """Edge case: reference mean is zero — should return 0, not error."""
        ref_proba = np.array([0.0, 0.0, 0.0])
        cur_proba = np.array([0.5, 0.5, 0.5])
        drift = DriftDetector.compute_prediction_drift(ref_proba, cur_proba)
        assert drift == 0.0

    def test_drift_returns_float(self):
        """Drift score should always be a float."""
        ref_proba = np.array([0.3, 0.4])
        cur_proba = np.array([0.5, 0.6])
        drift = DriftDetector.compute_prediction_drift(ref_proba, cur_proba)
        assert isinstance(drift, float)


# ---------------------------------------------------------------------------
# Report Generation Tests
# ---------------------------------------------------------------------------


class TestGenerateReport:
    """Tests for the full drift report generation."""

    def test_report_structure(self, detector, identical_current_df):
        """Report should contain all expected fields."""
        report = detector.generate_report(identical_current_df)
        assert isinstance(report, DriftReport)
        assert report.timestamp is not None
        assert isinstance(report.numerical_drift, list)
        assert isinstance(report.categorical_drift, list)
        assert isinstance(report.overall_drift_detected, bool)
        assert isinstance(report.drifted_features, list)
        assert isinstance(report.summary, str)

    def test_no_drift_with_identical_data(self, detector, identical_current_df):
        """No drift should be detected when data is identical to reference."""
        report = detector.generate_report(identical_current_df)
        assert report.overall_drift_detected is False
        assert len(report.drifted_features) == 0

    def test_drift_detected_with_shifted_data(self, detector, shifted_current_df):
        """Drift should be detected with obviously shifted data."""
        report = detector.generate_report(shifted_current_df)
        assert report.overall_drift_detected is True
        assert len(report.drifted_features) > 0

    def test_report_with_prediction_drift(self, detector, identical_current_df):
        """Report should include prediction drift when probabilities provided."""
        ref_proba = np.array([0.3] * 500)
        cur_proba = np.array([0.8] * 500)
        report = detector.generate_report(
            identical_current_df,
            reference_proba=ref_proba,
            current_proba=cur_proba,
        )
        assert report.prediction_drift_score is not None
        assert report.prediction_drift_score > 0

    def test_report_to_dict(self, detector, identical_current_df):
        """Report should serialize to a dict with expected keys."""
        report = detector.generate_report(identical_current_df)
        report_dict = report.to_dict()
        assert isinstance(report_dict, dict)
        assert "timestamp" in report_dict
        assert "overall_drift_detected" in report_dict
        assert "n_drifted_features" in report_dict
        assert "summary" in report_dict

    def test_report_shape_metadata(self, detector, identical_current_df):
        """Report should correctly capture reference and current shapes."""
        report = detector.generate_report(identical_current_df)
        assert report.reference_shape[0] == 500
        assert report.current_shape[0] == 500


# ---------------------------------------------------------------------------
# Auto-detection Tests
# ---------------------------------------------------------------------------


class TestAutoDetection:
    """Tests for automatic feature type detection."""

    def test_auto_detects_numerical_features(self, reference_df):
        """Detector should auto-detect numerical columns."""
        detector = DriftDetector(reference_df)
        # tenure, MonthlyCharges, TotalCharges should be detected
        assert "tenure" in detector._numerical_features
        assert "MonthlyCharges" in detector._numerical_features

    def test_auto_detects_categorical_features(self, reference_df):
        """Detector should auto-detect categorical (object) columns."""
        detector = DriftDetector(reference_df)
        assert "Contract" in detector._categorical_features
        assert "gender" in detector._categorical_features
