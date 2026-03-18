import os
from dataclasses import dataclass, field
from typing import Dict, Any


def _default_xgb_params():
    """XGBoost defaults, overridable via XGB_PARAM_* environment variables."""
    params = {
        "max_depth": 6,
        "n_estimators": 100,
        "learning_rate": 0.3,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    }
    env_map = {
        'XGB_PARAM_MAX_DEPTH': ('max_depth', int),
        'XGB_PARAM_N_ESTIMATORS': ('n_estimators', int),
        'XGB_PARAM_LEARNING_RATE': ('learning_rate', float),
        'XGB_PARAM_REG_ALPHA': ('reg_alpha', float),
        'XGB_PARAM_REG_LAMBDA': ('reg_lambda', float),
        'XGB_PARAM_SUBSAMPLE': ('subsample', float),
        'XGB_PARAM_COLSAMPLE_BYTREE': ('colsample_bytree', float),
    }
    for env_key, (param_name, cast) in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            params[param_name] = cast(val)
    return params


@dataclass
class StrategyConfig:
    name: str = "Paper Baseline"

    # --- Idea 1, 2 & 8: Walk-Forward Tuning Parameters ---
    tuning_metric: str = "sharpe"        # Options: "sharpe", "sortino"
    lambda_smoothing: bool = False       # Idea 2: Apply EWMA to selected optimal lambdas across periods
    lambda_ensemble_k: int = 1           # Idea 8: If > 1, average the OOS probability forecasts of the top K lambdas

    # --- Idea 3: Window Management ---
    validation_window_type: str = "rolling" # Options: "rolling" (fixed 5y), "expanding" (growing from 5y)

    # --- Idea 5 & 6: Signal & Execution Parameters ---
    prob_threshold: float = 0.50         # Probability threshold for binary bearish classification
    allocation_style: str = "binary"     # Options: "binary" (0% or 100%), "continuous" (invest 1 - P(bear))
    execution_mode: str = "next_open"    # Options: "close" (theoretical), "next_open" (realistic)

    # --- Idea 4 & 10: Model & Feature Parameters ---
    calculate_shap: bool = False         # Whether to calculate SHAP values (can be slow)
    dynamic_feature_selection: bool = False # Idea 4: Drop bottom 20% of features based on validation importance
    xgb_online_learning: bool = False    # Idea 10: Use XGBoost incremental learning (xgb_model) across chunks
    xgb_params: Dict[str, Any] = field(default_factory=_default_xgb_params)

    # --- Lambda Selection Strategy ---
    lambda_selection: str = "best"       # Options: "best" (argmax), "median_positive" (median of positive-Sharpe lambdas)
    lambda_subwindow_consensus: bool = False  # Split validation into sub-windows, take median best-lambda

    # --- Feature Ablation ---
    feature_ablation: str = "all"        # Options: "all" (default), "return_only", "macro_only"


def _strategy_config_from_env():
    """Create a StrategyConfig with defaults overridden by XGB_* environment variables.

    Used by scripts launched from Diagnostics Launcher (receives params via env vars).
    Explicit constructor args in EXPERIMENTS list always take precedence since those
    configs are built directly, not through this function.
    """
    kwargs = {}
    _env_overrides = {
        'XGB_TUNING_METRIC': ('tuning_metric', str),
        'XGB_VALIDATION_WINDOW_TYPE': ('validation_window_type', str),
        'XGB_LAMBDA_SMOOTHING': ('lambda_smoothing', lambda v: v.lower() == 'true'),
        'XGB_PROB_THRESHOLD': ('prob_threshold', float),
        'XGB_ALLOCATION_STYLE': ('allocation_style', str),
        'XGB_EXECUTION_MODE': ('execution_mode', str),
        'XGB_LAMBDA_ENSEMBLE_K': ('lambda_ensemble_k', int),
        'XGB_LAMBDA_SELECTION': ('lambda_selection', str),
        'XGB_LAMBDA_SUBWINDOW_CONSENSUS': ('lambda_subwindow_consensus', lambda v: v.lower() == 'true'),
    }
    for env_key, (field_name, cast) in _env_overrides.items():
        val = os.environ.get(env_key)
        if val is not None:
            kwargs[field_name] = cast(val)
    return kwargs
