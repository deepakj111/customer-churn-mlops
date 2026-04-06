import pytest
import yaml

from src.utils.config_loader import (
    ConfigLoader,
    FeatureConfig,
    ModelConfig,
    MonitoringConfig,
    TrainingConfig,
)

# ---------------------------------------------------------------------------
# Fixtures — write minimal valid YAML files into a temp directory
# ---------------------------------------------------------------------------


@pytest.fixture
def config_dir(tmp_path):
    """Create a temporary configs/ directory with all four YAML files."""

    model_data = {
        "model_name": "test-model",
        "algorithm": "lightgbm",
        "champion_stage": "Production",
        "hyperparameters": {"n_estimators": 100, "learning_rate": 0.05},
    }
    feature_data = {
        "target_column": "Churn",
        "customer_id_column": "customerID",
        "numerical_features": ["tenure", "MonthlyCharges"],
        "categorical_features": ["Contract"],
        "features_to_drop": ["customerID"],
        "engineered_features": ["is_month_to_month"],
    }
    training_data = {
        "test_size": 0.2,
        "val_size": 0.1,
        "random_seed": 42,
        "cv_folds": 5,
        "experiment_name": "test-experiment",
        "fn_cost": 500.0,
        "fp_cost": 20.0,
    }
    monitoring_data = {
        "psi_threshold": 0.25,
        "chi_squared_alpha": 0.05,
        "prediction_drift_threshold": 0.15,
        "performance_drop_threshold": 0.05,
        "reference_window_days": 90,
        "monitoring_window_days": 7,
        "label_delay_days": 45,
    }

    (tmp_path / "model_config.yaml").write_text(yaml.dump(model_data))
    (tmp_path / "feature_config.yaml").write_text(yaml.dump(feature_data))
    (tmp_path / "training_config.yaml").write_text(yaml.dump(training_data))
    (tmp_path / "monitoring_config.yaml").write_text(yaml.dump(monitoring_data))

    return tmp_path


# ---------------------------------------------------------------------------
# ModelConfig tests
# ---------------------------------------------------------------------------


def test_model_config_loads_correctly(config_dir):
    loader = ConfigLoader(config_dir=str(config_dir))
    config = loader.model
    assert isinstance(config, ModelConfig)
    assert config.model_name == "test-model"
    assert config.algorithm == "lightgbm"
    assert config.champion_stage == "Production"
    assert config.hyperparameters["n_estimators"] == 100


# ---------------------------------------------------------------------------
# FeatureConfig tests
# ---------------------------------------------------------------------------


def test_feature_config_loads_correctly(config_dir):
    loader = ConfigLoader(config_dir=str(config_dir))
    config = loader.features
    assert isinstance(config, FeatureConfig)
    assert config.target_column == "Churn"
    assert "tenure" in config.numerical_features
    assert "customerID" in config.features_to_drop


# ---------------------------------------------------------------------------
# TrainingConfig tests
# ---------------------------------------------------------------------------


def test_training_config_loads_correctly(config_dir):
    loader = ConfigLoader(config_dir=str(config_dir))
    config = loader.training
    assert isinstance(config, TrainingConfig)
    assert config.random_seed == 42
    assert config.cv_folds == 5
    assert config.fn_cost == 500.0
    assert config.fp_cost == 20.0


# ---------------------------------------------------------------------------
# MonitoringConfig tests
# ---------------------------------------------------------------------------


def test_monitoring_config_loads_correctly(config_dir):
    loader = ConfigLoader(config_dir=str(config_dir))
    config = loader.monitoring
    assert isinstance(config, MonitoringConfig)
    assert config.psi_threshold == 0.25
    assert config.label_delay_days == 45


# ---------------------------------------------------------------------------
# Lazy loading and caching behaviour
# ---------------------------------------------------------------------------


def test_config_is_none_before_first_access(config_dir):
    loader = ConfigLoader(config_dir=str(config_dir))
    assert loader._model_config is None
    assert loader._feature_config is None
    assert loader._training_config is None
    assert loader._monitoring_config is None


def test_config_cached_after_first_access(config_dir):
    loader = ConfigLoader(config_dir=str(config_dir))
    first_call = loader.model
    second_call = loader.model
    # Must be the exact same object — not reloaded from disk
    assert first_call is second_call


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_missing_config_file_raises_file_not_found():
    loader = ConfigLoader(config_dir="/path/that/does/not/exist")
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        _ = loader.model


def test_error_message_includes_path():
    loader = ConfigLoader(config_dir="/bad/path")
    with pytest.raises(FileNotFoundError, match="model_config.yaml"):
        _ = loader.model
