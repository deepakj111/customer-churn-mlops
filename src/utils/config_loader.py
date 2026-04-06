from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config dataclasses — one per YAML file in configs/
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """
    Maps to configs/model_config.yaml.
    Holds the champion model's identity and hyperparameters.
    """

    model_name: str
    algorithm: str
    champion_stage: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureConfig:
    """
    Maps to configs/feature_config.yaml.
    Single source of truth for every column and engineered feature name.
    All modules import from here — never hardcode column names in business logic.
    """

    target_column: str
    customer_id_column: str
    numerical_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    features_to_drop: List[str] = field(default_factory=list)
    engineered_features: List[str] = field(default_factory=list)


@dataclass
class TrainingConfig:
    """
    Maps to configs/training_config.yaml.
    Controls train/val/test splits, cross-validation, and the business
    cost matrix used for threshold optimization.
    """

    test_size: float
    val_size: float
    random_seed: int
    cv_folds: int
    experiment_name: str
    fn_cost: float  # cost of a false negative (missing a churner)
    fp_cost: float  # cost of a false positive (unnecessary retention offer)


@dataclass
class MonitoringConfig:
    """
    Maps to configs/monitoring_config.yaml.
    All thresholds that trigger alerts or retraining.
    """

    psi_threshold: float  # PSI > this = input drift alert
    chi_squared_alpha: float  # p-value < this = categorical drift alert
    prediction_drift_threshold: float  # relative mean shift > this = prediction drift
    performance_drop_threshold: float  # F1 drop > this = retraining trigger
    reference_window_days: int  # days of data used as the drift reference
    monitoring_window_days: int  # days of recent data to compare against reference
    label_delay_days: int  # days before ground truth labels are available


# ---------------------------------------------------------------------------
# Loader class
# ---------------------------------------------------------------------------


class ConfigLoader:
    """
    Loads YAML config files from the configs/ directory into typed dataclasses.

    Uses lazy loading — each config is read from disk only on first access,
    then cached for all subsequent calls. This means the files are never
    read more than once per process lifetime.

    Usage:
        from src.utils.config_loader import get_config
        cfg = get_config()
        print(cfg.model.algorithm)
        print(cfg.features.target_column)
    """

    def __init__(self, config_dir: str = "configs") -> None:
        self._config_dir = Path(config_dir)
        self._model_config: Optional[ModelConfig] = None
        self._feature_config: Optional[FeatureConfig] = None
        self._training_config: Optional[TrainingConfig] = None
        self._monitoring_config: Optional[MonitoringConfig] = None

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Read a YAML file and return its contents as a dict."""
        path = self._config_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {path.resolve()}\n"
                f"Make sure you are running from the project root directory."
            )
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        logger.debug("Loaded config file: %s", path)
        return data

    @property
    def model(self) -> ModelConfig:
        if self._model_config is None:
            data = self._load_yaml("model_config.yaml")
            self._model_config = ModelConfig(**data)
        return self._model_config

    @property
    def features(self) -> FeatureConfig:
        if self._feature_config is None:
            data = self._load_yaml("feature_config.yaml")
            self._feature_config = FeatureConfig(**data)
        return self._feature_config

    @property
    def training(self) -> TrainingConfig:
        if self._training_config is None:
            data = self._load_yaml("training_config.yaml")
            self._training_config = TrainingConfig(**data)
        return self._training_config

    @property
    def monitoring(self) -> MonitoringConfig:
        if self._monitoring_config is None:
            data = self._load_yaml("monitoring_config.yaml")
            self._monitoring_config = MonitoringConfig(**data)
        return self._monitoring_config


# ---------------------------------------------------------------------------
# Module-level singleton — the only instance used across the entire project
# ---------------------------------------------------------------------------

_config_loader: Optional[ConfigLoader] = None


def get_config(config_dir: str = "configs") -> ConfigLoader:
    """
    Return the singleton ConfigLoader instance.

    Call this from any module that needs configuration values:
        from src.utils.config_loader import get_config
        cfg = get_config()
        target = cfg.features.target_column
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_dir=config_dir)
    return _config_loader
