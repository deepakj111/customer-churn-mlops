"""
Unit tests for the reference data builder module.

Tests cover:
    - Saving and loading reference features (Parquet round-trip)
    - Saving and loading reference predictions
    - Metadata creation and retrieval
    - Error handling for missing files
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.monitoring.reference_builder import (
    build_reference,
    get_reference_metadata,
    load_reference,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Small feature DataFrame for testing."""
    return pd.DataFrame(
        {
            "tenure": [12, 24, 6, 48, 1],
            "MonthlyCharges": [70.0, 45.0, 90.0, 30.0, 85.0],
            "Contract": [
                "Month-to-month",
                "One year",
                "Month-to-month",
                "Two year",
                "Month-to-month",
            ],
        }
    )


@pytest.fixture
def sample_target() -> pd.Series:
    """Binary target for testing."""
    return pd.Series([1, 0, 1, 0, 1], name="Churn")


@pytest.fixture
def sample_predictions() -> np.ndarray:
    """Churn probabilities for testing."""
    return np.array([0.85, 0.12, 0.73, 0.05, 0.91])


@pytest.fixture
def tmp_reference_dir(tmp_path):
    """Create a temporary directory for reference data."""
    ref_dir = tmp_path / "reference"
    ref_dir.mkdir()
    return ref_dir


# ---------------------------------------------------------------------------
# build_reference Tests
# ---------------------------------------------------------------------------


class TestBuildReference:
    """Tests for building and saving reference data."""

    def test_creates_features_parquet(
        self, sample_features, sample_target, tmp_reference_dir
    ):
        """Should create a Parquet file for features."""
        build_reference(
            X=sample_features,
            y=sample_target,
            output_dir=tmp_reference_dir,
        )
        assert (tmp_reference_dir / "reference_features.parquet").exists()

    def test_creates_metadata_json(
        self, sample_features, sample_target, tmp_reference_dir
    ):
        """Should create a metadata JSON file."""
        build_reference(
            X=sample_features,
            y=sample_target,
            output_dir=tmp_reference_dir,
        )
        assert (tmp_reference_dir / "reference_metadata.json").exists()

    def test_creates_probabilities_npy_when_provided(
        self, sample_features, sample_target, sample_predictions, tmp_reference_dir
    ):
        """Should create a .npy file when predictions are provided."""
        build_reference(
            X=sample_features,
            y=sample_target,
            predictions=sample_predictions,
            output_dir=tmp_reference_dir,
        )
        assert (tmp_reference_dir / "reference_probabilities.npy").exists()

    def test_no_probabilities_npy_when_not_provided(
        self, sample_features, sample_target, tmp_reference_dir
    ):
        """Should NOT create a .npy file when predictions are not provided."""
        build_reference(
            X=sample_features,
            y=sample_target,
            predictions=None,
            output_dir=tmp_reference_dir,
        )
        assert not (tmp_reference_dir / "reference_probabilities.npy").exists()

    def test_metadata_contents(self, sample_features, sample_target, tmp_reference_dir):
        """Metadata should contain correct values."""
        build_reference(
            X=sample_features,
            y=sample_target,
            model_version="v1.2.3",
            output_dir=tmp_reference_dir,
        )
        metadata = json.loads(
            (tmp_reference_dir / "reference_metadata.json").read_text()
        )
        assert metadata["n_rows"] == 5
        assert metadata["n_features"] == 3
        assert metadata["model_version"] == "v1.2.3"
        assert "created_at" in metadata
        assert "churn_rate" in metadata

    def test_returns_output_directory(
        self, sample_features, sample_target, tmp_reference_dir
    ):
        """Should return the path to the output directory."""
        result = build_reference(
            X=sample_features,
            y=sample_target,
            output_dir=tmp_reference_dir,
        )
        assert result == tmp_reference_dir


# ---------------------------------------------------------------------------
# load_reference Tests
# ---------------------------------------------------------------------------


class TestLoadReference:
    """Tests for loading reference data from disk."""

    def test_loads_features_correctly(
        self, sample_features, sample_target, tmp_reference_dir
    ):
        """Should load features with correct shape and columns."""
        build_reference(
            X=sample_features, y=sample_target, output_dir=tmp_reference_dir
        )
        df, _ = load_reference(data_dir=tmp_reference_dir)
        assert df.shape == sample_features.shape
        assert list(df.columns) == list(sample_features.columns)

    def test_loads_predictions_when_available(
        self,
        sample_features,
        sample_target,
        sample_predictions,
        tmp_reference_dir,
    ):
        """Should load predictions when they were saved."""
        build_reference(
            X=sample_features,
            y=sample_target,
            predictions=sample_predictions,
            output_dir=tmp_reference_dir,
        )
        _, predictions = load_reference(data_dir=tmp_reference_dir)
        assert predictions is not None
        np.testing.assert_array_almost_equal(predictions, sample_predictions)

    def test_returns_none_predictions_when_not_saved(
        self, sample_features, sample_target, tmp_reference_dir
    ):
        """Should return None for predictions when they weren't saved."""
        build_reference(
            X=sample_features, y=sample_target, output_dir=tmp_reference_dir
        )
        _, predictions = load_reference(data_dir=tmp_reference_dir)
        assert predictions is None

    def test_raises_on_missing_file(self, tmp_reference_dir):
        """Should raise FileNotFoundError if reference file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Reference data not found"):
            load_reference(data_dir=tmp_reference_dir)

    def test_parquet_preserves_dtypes(
        self, sample_features, sample_target, tmp_reference_dir
    ):
        """Parquet should preserve data types exactly (unlike CSV)."""
        build_reference(
            X=sample_features, y=sample_target, output_dir=tmp_reference_dir
        )
        df, _ = load_reference(data_dir=tmp_reference_dir)
        assert df["tenure"].dtype == sample_features["tenure"].dtype
        assert df["MonthlyCharges"].dtype == sample_features["MonthlyCharges"].dtype


# ---------------------------------------------------------------------------
# get_reference_metadata Tests
# ---------------------------------------------------------------------------


class TestGetReferenceMetadata:
    """Tests for loading reference metadata."""

    def test_loads_metadata(self, sample_features, sample_target, tmp_reference_dir):
        """Should load metadata as a dict."""
        build_reference(
            X=sample_features, y=sample_target, output_dir=tmp_reference_dir
        )
        metadata = get_reference_metadata(data_dir=tmp_reference_dir)
        assert isinstance(metadata, dict)
        assert "created_at" in metadata
        assert "n_rows" in metadata

    def test_raises_on_missing_metadata(self, tmp_reference_dir):
        """Should raise FileNotFoundError if metadata file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Reference metadata not found"):
            get_reference_metadata(data_dir=tmp_reference_dir)

    def test_metadata_feature_names(
        self, sample_features, sample_target, tmp_reference_dir
    ):
        """Metadata should include the feature names list."""
        build_reference(
            X=sample_features, y=sample_target, output_dir=tmp_reference_dir
        )
        metadata = get_reference_metadata(data_dir=tmp_reference_dir)
        assert metadata["feature_names"] == list(sample_features.columns)
