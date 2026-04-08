"""
sklearn Pipeline factory for the churn prediction model.

Builds a single serializable Pipeline object that contains:
    Step 1 — feature_engineering : FunctionTransformer(engineer_features)
    Step 2 — preprocessor        : ColumnTransformer
                                     ├── OneHotEncoder  (categorical)
                                     ├── StandardScaler (numerical)
                                     └── passthrough    (binary 0/1)
    Step 3 — classifier          : LGBMClassifier

Why one Pipeline object?
    Serialising the entire chain as a single MLflow artifact guarantees
    that inference uses the exact same fitted transformers as training.
    Training-serving skew becomes structurally impossible.

Why FunctionTransformer for feature engineering?
    engineer_features() is stateless — it never fits anything.
    FunctionTransformer wraps it into a sklearn-compatible step so the
    full chain (feature engineering → encoding → model) lives in one
    object. validate=False preserves the pandas DataFrame so downstream
    ColumnTransformer can select columns by name.

Public API:
    build_pipeline(params)   → unfitted sklearn Pipeline
    get_preprocessor()       → unfitted ColumnTransformer (for inspection)
"""

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features.feature_store import (
    engineer_features,
    get_binary_features,
    get_categorical_features,
    get_numerical_features,
)
from src.utils.config_loader import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_preprocessor() -> ColumnTransformer:
    """
    Build the ColumnTransformer that routes features to the correct encoder.

    Routes:
        Categorical → OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            'ignore' means unseen categories at inference time get all-zero
            encoding instead of raising an error — essential for production.

        Numerical → StandardScaler()
            Zero mean, unit variance. Required by regularised models and
            improves convergence for gradient boosters too.

        Binary → passthrough
            Already 0/1 integers. No transformation needed.

    remainder='drop' ensures any column not explicitly listed is silently
    dropped. This prevents future accidental column additions from leaking
    into the model without explicit review.
    """
    categorical = get_categorical_features()
    numerical = get_numerical_features()
    binary = get_binary_features()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop=None,
                ),
                categorical,
            ),
            (
                "num",
                StandardScaler(),
                numerical,
            ),
            (
                "bin",
                "passthrough",
                binary,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    logger.debug(
        "Preprocessor built — categorical: %d, numerical: %d, binary: %d cols.",
        len(categorical),
        len(numerical),
        len(binary),
    )

    return preprocessor


def build_pipeline(params: dict | None = None) -> Pipeline:
    """
    Build the full unfitted sklearn Pipeline.

    Reads LightGBM hyperparameters from model_config.yaml unless
    overridden by the params argument. Params override is used by
    hyperparameter search (Optuna / GridSearchCV) which passes trial
    params directly to this function.

    Args:
        params: Optional dict of LightGBM hyperparameters that override
                the config file values. Keys must match LGBMClassifier
                constructor arguments exactly.

    Returns:
        Unfitted sklearn Pipeline with three named steps:
            'feature_engineering', 'preprocessor', 'classifier'
    """
    cfg = get_config()
    # lgbm_params = dict(cfg.model.lightgbm)
    lgbm_params = dict(cfg.model.hyperparameters)

    if params:
        lgbm_params.update(params)
        logger.debug("Pipeline built with param overrides: %s", params)

    classifier = LGBMClassifier(**lgbm_params)

    pipeline = Pipeline(
        steps=[
            (
                "feature_engineering",
                FunctionTransformer(engineer_features, validate=False),
            ),
            (
                "preprocessor",
                get_preprocessor(),
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    logger.info(
        "Pipeline built — LightGBM n_estimators=%d, learning_rate=%.3f.",
        lgbm_params.get("n_estimators", "?"),
        lgbm_params.get("learning_rate", "?"),
    )

    return pipeline


def build_baseline_pipeline(classifier) -> Pipeline:
    """
    Build a sklearn Pipeline for baseline and ensemble experimentation.

    Uses the identical feature engineering and preprocessing as build_pipeline(),
    but accepts any sklearn-compatible estimator. This ensures every algorithm
    comparison in notebooks 03 and 04 is a pure algorithm comparison —
    same features, same preprocessing, different model.

    Args:
        classifier: Any unfitted sklearn-compatible estimator.

    Returns:
        Unfitted sklearn Pipeline with three named steps:
            'feature_engineering', 'preprocessor', 'classifier'
    """
    pipeline = Pipeline(
        steps=[
            (
                "feature_engineering",
                FunctionTransformer(engineer_features, validate=False),
            ),
            (
                "preprocessor",
                get_preprocessor(),
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    logger.debug(
        "Baseline pipeline built — classifier: %s",
        type(classifier).__name__,
    )

    return pipeline
