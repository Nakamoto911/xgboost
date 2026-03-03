"""
Diagnose why Paper Baseline doesn't beat B&H.
Tests 4 configurations to isolate the impact of:
  1. Time period (2007-2026 vs paper's 2007-2023)
  2. XGBoost hyperparameters (custom regularized vs paper "default")
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import main
from main import fetch_and_prepare_data, walk_forward_backtest, calculate_metrics
from config import StrategyConfig

# XGBoost "default" params as the paper likely intended
# (sklearn-style defaults for XGBClassifier)
XGB_DEFAULTS = {
    "max_depth": 6,
    "n_estimators": 100,
    "learning_rate": 0.3,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
}

# Our current regularized params
XGB_REGULARIZED = {
    "max_depth": 4,
    "n_estimators": 100,
    "learning_rate": 0.1,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

CONFIGS = [
    {
        "name": "A: Current (2007-2026, regularized XGB)",
        "end_date": "2026-01-01",
        "xgb_params": XGB_REGULARIZED,
    },
    {
        "name": "B: Paper period (2007-2023, regularized XGB)",
        "end_date": "2024-01-01",
        "xgb_params": XGB_REGULARIZED,
    },
    {
        "name": "C: Current period (2007-2026, default XGB)",
        "end_date": "2026-01-01",
        "xgb_params": XGB_DEFAULTS,
    },
    {
        "name": "D: Paper period (2007-2023, default XGB)",
        "end_date": "2024-01-01",
        "xgb_params": XGB_DEFAULTS,
    },
]


def run_diagnostic():
    print("=" * 90)
    print("PAPER BASELINE DIAGNOSTIC: Isolating Time Period vs XGB Hyperparameters")
    print("=" * 90)

    df = fetch_and_prepare_data()

    results = []
    for cfg in CONFIGS:
        label = cfg["name"]
        print(f"\n--- {label} ---")
        t0 = time.time()

        # Override module-level END_DATE
        main.END_DATE = cfg["end_date"]

        # Clear forecast cache to avoid stale results
        main._forecast_cache.clear()

        strategy_cfg = StrategyConfig(
            name=label,
            xgb_params=cfg["xgb_params"],
        )

        res_df = walk_forward_backtest(df, strategy_cfg)
        s_ret, s_vol, s_sharpe, s_sortino, s_mdd = calculate_metrics(
            res_df["Strat_Return"], res_df["RF_Rate"]
        )
        b_ret, b_vol, b_sharpe, b_sortino, b_mdd = calculate_metrics(
            res_df["Target_Return"], res_df["RF_Rate"]
        )

        elapsed = time.time() - t0
        result = {
            "Config": label,
            "Sharpe": s_sharpe,
            "B&H Sharpe": b_sharpe,
            "Delta": s_sharpe - b_sharpe,
            "Sortino": s_sortino,
            "B&H Sortino": b_sortino,
            "Ann Ret %": s_ret * 100,
            "B&H Ret %": b_ret * 100,
            "Max DD %": s_mdd * 100,
            "B&H MDD %": b_mdd * 100,
            "Lambda Mean": np.mean(res_df.attrs.get("lambda_history", [0])),
            "Lambda CV": (
                np.std(res_df.attrs.get("lambda_history", [0]))
                / np.mean(res_df.attrs.get("lambda_history", [1]))
                if np.mean(res_df.attrs.get("lambda_history", [1])) > 0
                else float("nan")
            ),
            "EWMA HL": res_df.attrs.get("ewma_halflife", "?"),
            "Time": f"{elapsed:.0f}s",
        }
        results.append(result)

        print(
            f"  Sharpe: {s_sharpe:.3f} vs B&H {b_sharpe:.3f} (delta {s_sharpe - b_sharpe:+.3f})"
        )
        print(f"  Sortino: {s_sortino:.3f} vs B&H {b_sortino:.3f}")
        print(f"  Ann Ret: {s_ret*100:.2f}% vs B&H {b_ret*100:.2f}%")
        print(f"  Max DD: {s_mdd*100:.1f}% vs B&H {b_mdd*100:.1f}%")
        print(f"  Lambda mean={np.mean(res_df.attrs.get('lambda_history', [0])):.1f}, CV={result['Lambda CV']:.2f}")
        print(f"  EWMA HL: {result['EWMA HL']}")
        print(f"  Runtime: {elapsed:.0f}s")

    # Reset END_DATE
    main.END_DATE = "2026-01-01"

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Config':<50} {'Sharpe':>7} {'B&H':>7} {'Delta':>7} {'Sortino':>8} {'Ann Ret%':>9} {'MDD%':>7} {'Lam CV':>7}")
    print("-" * 90)
    for r in results:
        delta_str = f"{r['Delta']:+.3f}"
        verdict = "WIN" if r["Delta"] > 0 else "LOSE"
        print(
            f"{r['Config']:<50} {r['Sharpe']:>7.3f} {r['B&H Sharpe']:>7.3f} {delta_str:>7} {r['Sortino']:>8.3f} {r['Ann Ret %']:>8.2f}% {r['Max DD %']:>6.1f}% {r['Lambda CV']:>7.2f}  {verdict}"
        )

    # Effect decomposition
    print("\n--- EFFECT DECOMPOSITION ---")
    a = results[0]  # current + reg
    b = results[1]  # paper period + reg
    c = results[2]  # current + default
    d = results[3]  # paper period + default

    period_effect = b["Delta"] - a["Delta"]
    xgb_effect = c["Delta"] - a["Delta"]
    combined = d["Delta"] - a["Delta"]
    interaction = combined - period_effect - xgb_effect

    print(f"  Time period effect (2026->2023): {period_effect:+.3f} Sharpe delta")
    print(f"  XGB params effect (reg->default): {xgb_effect:+.3f} Sharpe delta")
    print(f"  Combined effect:                  {combined:+.3f} Sharpe delta")
    print(f"  Interaction term:                 {interaction:+.3f} Sharpe delta")
    print()
    print(f"  Paper-comparable config (D): Sharpe {d['Sharpe']:.3f} vs B&H {d['B&H Sharpe']:.3f} (delta {d['Delta']:+.3f})")
    print(f"  Paper reports: Sharpe ~0.79 vs B&H ~0.50")
    if d["Delta"] > 0:
        print(f"  => Config D BEATS B&H by {d['Delta']:+.3f} Sharpe")
    else:
        print(f"  => Config D LOSES to B&H by {d['Delta']:.3f} Sharpe")


if __name__ == "__main__":
    run_diagnostic()
