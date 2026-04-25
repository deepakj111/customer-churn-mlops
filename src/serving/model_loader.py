"""
Thread-safe model loading and caching for the serving layer.

Abstracts away WHERE the model comes from so the API code never deals
with MLflow URIs, file paths, or registry versions directly.

Loading strategy (fallback chain):
    1. MLflow Model Registry → production deployment mode
       Uses MLFLOW_TRACKING_URI from environment. Loads by model name
       and stage (e.g. "customer-churn-lgbm" @ "Staging").

    2. Local training → development / demo mode
       Trains a fresh pipeline on the Telco dataset, finds the optimal
       threshold, and caches both. Takes ~30 seconds on first request,
       then serves from cache. No MLflow server needed.

    3. Error → raises RuntimeError with actionable message

Thread safety:
    Uses a threading.Lock around the load operation. Multiple Uvicorn
    workers each get their own process (and thus their own model copy),
    but within a single worker, concurrent async requests that trigger
    a lazy load are serialized by the lock.

Public API:
    get_model()     → fitted sklearn Pipeline (cached)
    get_threshold() → float (cached)
    get_model_info() → dict of metadata
    reload_model()  → force re-load (for production model updates)
"""

from __future__ import annotations

import threading
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level state — protected by _lock
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_pipeline: Any | None = None  # fitted sklearn Pipeline
_threshold: float = 0.5  # cost-optimal threshold
_model_info: dict[str, Any] = {}  # metadata dict


def _try_load_from_mlflow() -> bool:
    """
    Attempt to load the model from MLflow Model Registry.

    Returns True if successful, False if MLflow is unavailable or the
    model is not registered. Updates module-level _pipeline, _threshold,
    and _model_info on success.

    This is the preferred path in production. The Docker container sets
    MLFLOW_TRACKING_URI to the DagsHub or self-hosted MLflow server.
    """
    global _pipeline, _threshold, _model_info

    try:
        import os

        import mlflow
        import mlflow.sklearn

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            logger.debug("MLFLOW_TRACKING_URI not set — skipping MLflow load.")
            return False

        mlflow.set_tracking_uri(tracking_uri)

        from src.utils.config_loader import get_config

        cfg = get_config()
        model_name = cfg.model.registered_model_name
        stage = cfg.model.champion_stage

        # Load the latest model version at the given stage.
        # MLflow URI format: "models:/<model_name>/<stage>"
        model_uri = f"models:/{model_name}/{stage}"
        logger.info("Loading model from MLflow: %s (stage: %s)", model_name, stage)
        _pipeline = mlflow.sklearn.load_model(model_uri)

        # Try to retrieve the optimal threshold from the run's logged metrics.
        # If unavailable, fall back to the default.
        try:
            client = mlflow.MlflowClient()
            # Get the latest version matching the stage
            versions = client.get_latest_versions(model_name, stages=[stage])
            if versions:
                run_id = versions[0].run_id
                run = client.get_run(run_id)
                logged_threshold = run.data.metrics.get("optimal_threshold", 0.5)
                _threshold = float(logged_threshold)
                _model_info = {
                    "model_name": model_name,
                    "model_version": versions[0].version,
                    "mlflow_run_id": run_id,
                    "optimal_threshold": _threshold,
                    "algorithm": run.data.tags.get("model_type", "unknown"),
                    "feature_count": int(run.data.tags.get("feature_count", "0")),
                    "source": "mlflow_registry",
                }
        except Exception as meta_err:
            logger.warning("Model loaded but metadata retrieval failed: %s", meta_err)
            _model_info = {
                "model_name": model_name,
                "source": "mlflow_registry",
                "optimal_threshold": _threshold,
            }

        logger.info("Model loaded from MLflow — threshold: %.2f", _threshold)
        return True

    except Exception as e:
        logger.debug("MLflow load failed: %s", e)
        return False


def _train_local_model() -> None:
    """
    Train a fresh model locally for development and demo mode.

    This runs the full training pipeline without MLflow logging so
    the API can start without any external dependencies. The model
    lives only in memory — it is not persisted to disk.

    Takes ~15-30 seconds on a modern laptop.
    """
    global _pipeline, _threshold, _model_info

    logger.info(
        "Training local model for development mode " "(no MLflow server detected)..."
    )

    from sklearn.model_selection import train_test_split

    from src.data.ingest import load_for_training
    from src.data.preprocess import run_preprocessing
    from src.data.validate import validate_raw_data
    from src.models.pipeline import build_pipeline
    from src.models.threshold import find_cost_optimal_threshold
    from src.utils.config_loader import get_config

    cfg = get_config()

    # Load + validate + preprocess
    raw_df = load_for_training()
    validated_df = validate_raw_data(raw_df)
    X, y = run_preprocessing(validated_df)

    # 3-way stratified split matching run_training_experiment():
    #   train (70%) / val (10%) / test (20%)
    # The test set is discarded here — it exists only so the split
    # proportions are identical to the production training pipeline.
    # The val set is used exclusively for threshold optimisation.
    X_trainval, _X_test, y_trainval, _y_test = train_test_split(
        X,
        y,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state,
        stratify=y,
    )
    relative_val_size = cfg.training.val_size / (1.0 - cfg.training.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=relative_val_size,
        random_state=cfg.training.random_state,
        stratify=y_trainval,
    )

    # Build and fit pipeline
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Calibrate probabilities on validation set (matches production pipeline)
    from sklearn.calibration import CalibratedClassifierCV

    calibrated_pipeline = CalibratedClassifierCV(
        estimator=pipeline, method="isotonic", cv="prefit"
    )
    calibrated_pipeline.fit(X_val, y_val)

    # Find optimal threshold on validation set using calibrated probabilities
    y_val_proba = calibrated_pipeline.predict_proba(X_val)[:, 1]
    threshold = find_cost_optimal_threshold(y_val.values, y_val_proba)

    _pipeline = calibrated_pipeline
    _threshold = threshold
    _model_info = {
        "model_name": cfg.model.registered_model_name,
        "model_version": "local-dev",
        "mlflow_run_id": None,
        "optimal_threshold": _threshold,
        "algorithm": cfg.model.algorithm,
        "feature_count": len(X_train.columns),
        "source": "local_training",
    }

    logger.info(
        "Local model trained — threshold: %.2f, features: %d",
        _threshold,
        len(X_train.columns),
    )


def _ensure_loaded() -> None:
    """
    Ensure the model is loaded, using the fallback chain.

    Thread-safe: only one thread can trigger loading at a time.
    Subsequent calls return immediately if the model is already cached.
    """

    if _pipeline is not None:
        return

    with _lock:
        # Double-check inside the lock (another thread may have loaded
        # while we were waiting for the lock).
        if _pipeline is not None:
            return

        # Fallback chain: MLflow → local training → error
        if _try_load_from_mlflow():
            return

        try:
            _train_local_model()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from MLflow and local training: {e}. "
                "Ensure either MLFLOW_TRACKING_URI is set and the model is "
                "registered, or the training data is available at "
                "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv."
            ) from e


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_model() -> Any:
    """
    Return the cached fitted sklearn Pipeline.

    Triggers lazy loading on first call via the fallback chain:
    MLflow registry → local training → RuntimeError.

    Returns:
        Fitted sklearn Pipeline with feature_engineering, preprocessor,
        and classifier steps.

    Raises:
        RuntimeError: If neither MLflow nor local training succeeds.
    """
    _ensure_loaded()
    return _pipeline


def get_threshold() -> float:
    """
    Return the cost-optimal decision threshold.

    The threshold is loaded alongside the model. For MLflow models,
    it is read from the run's logged metrics. For local models,
    it is computed during training.

    Returns:
        Float threshold between 0 and 1.
    """
    _ensure_loaded()
    return _threshold


def get_model_info() -> dict[str, Any]:
    """
    Return model metadata for the /model/info endpoint.

    Returns:
        Dict with keys: model_name, model_version, mlflow_run_id,
        optimal_threshold, algorithm, feature_count, source.
    """
    _ensure_loaded()
    return _model_info.copy()


def reload_model() -> None:
    """
    Force re-load the model from the fallback chain.

    Called by the /admin/reload endpoint (if exposed) or by the
    monitoring pipeline when a new champion model is promoted.
    Thread-safe: acquires the lock before clearing the cache.
    """
    global _pipeline, _threshold, _model_info

    with _lock:
        logger.info("Force-reloading model...")
        _pipeline = None
        _threshold = 0.5
        _model_info = {}

    # Trigger the full load chain again
    _ensure_loaded()
    logger.info("Model reloaded successfully.")
