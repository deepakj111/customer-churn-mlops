from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Nested config dataclasses — required by evaluate.py, threshold.py, train.py
# ---------------------------------------------------------------------------
@dataclass
class CostMatrixConfig:
    """Business cost model for threshold optimisation."""

    false_negative_cost: float = 500.0
    false_positive_cost: float = 20.0


@dataclass
class RiskTiersConfig:
    """Probability boundaries for HIGH / MEDIUM / LOW risk tiers."""

    high_risk_threshold: float = 0.60
    medium_risk_threshold: float = 0.35


@dataclass
class PerformanceGatesConfig:
    """Minimum metric values a Challenger must meet before promotion."""

    min_roc_auc: float = 0.82
    min_pr_auc: float = 0.65
    min_recall_at_threshold: float = 0.70


# ---------------------------------------------------------------------------
# Top-level config dataclasses — one per YAML file
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Maps to configs/model_config.yaml."""

    model_name: str
    algorithm: str
    champion_stage: str
    registered_model_name: str = "customer-churn-lgbm"  # <-- add default
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    cost_matrix: CostMatrixConfig = field(default_factory=CostMatrixConfig)
    risk_tiers: RiskTiersConfig = field(default_factory=RiskTiersConfig)
    performance_gates: PerformanceGatesConfig = field(
        default_factory=PerformanceGatesConfig
    )


@dataclass
class FeatureConfig:
    """Maps to configs/feature_config.yaml."""

    target_column: str
    customer_id_column: str
    numerical_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    features_to_drop: list[str] = field(default_factory=list)
    engineered_features: list[str] = field(default_factory=list)


@dataclass
class TrainingConfig:
    """Maps to configs/training_config.yaml."""

    test_size: float
    val_size: float
    random_state: int
    cv_folds: int
    cv_scoring: str
    experiment_name: str


@dataclass
class MonitoringConfig:
    """Maps to configs/monitoring_config.yaml."""

    psi_threshold: float
    chi_squared_alpha: float
    prediction_drift_threshold: float
    performance_drop_threshold: float
    reference_window_days: int
    monitoring_window_days: int
    label_delay_days: int


# ---------------------------------------------------------------------------
# ConfigLoader — lazy-loading singleton
# ---------------------------------------------------------------------------


class ConfigLoader:
    """
    Loads YAML config files from the configs/ directory into typed dataclasses.
    Each config is read from disk only on first access, then cached.
    """

    def __init__(self, config_dir: str = "configs") -> None:
        self._config_dir = Path(config_dir)
        self._model_config: ModelConfig | None = None
        self._feature_config: FeatureConfig | None = None
        self._training_config: TrainingConfig | None = None
        self._monitoring_config: MonitoringConfig | None = None

    def _load_yaml(self, filename: str) -> dict[str, Any]:
        path = self._config_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {path.resolve()}. "
                "Make sure you are running from the project root directory."
            )
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        logger.debug("Loaded config file: %s", path)
        return dict(data) if data is not None else {}

    @property
    def model(self) -> ModelConfig:
        if self._model_config is None:
            data = self._load_yaml("model_config.yaml")

            # Pop nested dicts and build typed nested dataclasses.
            # This prevents ModelConfig(**data) from receiving raw dicts
            # for cost_matrix / risk_tiers / performance_gates.
            cost_matrix_raw = data.pop("cost_matrix", {})
            risk_tiers_raw = data.pop("risk_tiers", {})
            performance_gates_raw = data.pop("performance_gates", {})

            data["cost_matrix"] = CostMatrixConfig(**cost_matrix_raw)
            data["risk_tiers"] = RiskTiersConfig(**risk_tiers_raw)
            data["performance_gates"] = PerformanceGatesConfig(**performance_gates_raw)

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
# Module-level singleton
# ---------------------------------------------------------------------------

_config_loader: ConfigLoader | None = None


def get_config(config_dir: str = "configs") -> ConfigLoader:
    """
    Return the singleton ConfigLoader instance.

    If called with a different config_dir than the cached instance,
    the singleton is replaced with a new loader pointing to the
    requested directory.

    Usage:
        from src.utils.config_loader import get_config
        cfg = get_config()
        target = cfg.features.target_column
    """
    global _config_loader
    if _config_loader is None or str(_config_loader._config_dir) != config_dir:
        _config_loader = ConfigLoader(config_dir=config_dir)
    return _config_loader


def reset_config() -> None:
    """
    Reset the singleton ConfigLoader — primarily for test isolation.

    Call this in a pytest fixture (e.g. conftest.py autouse fixture)
    to guarantee each test starts with a fresh config state.
    """
    global _config_loader
    _config_loader = None
