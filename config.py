from dataclasses import dataclass, field
from typing import Dict, Any

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
    
    # --- Idea 4 & 10: Model & Feature Parameters ---
    dynamic_feature_selection: bool = False # Idea 4: Drop bottom 20% of features based on validation importance
    xgb_online_learning: bool = False    # Idea 10: Use XGBoost incremental learning (xgb_model) across chunks
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "max_depth": 6,
        "n_estimators": 100,
        "learning_rate": 0.3,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    })
