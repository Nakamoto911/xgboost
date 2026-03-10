"""
Macro Feature Ablation Study
============================
Tests three XGBoost feature configurations to isolate macro feature impact:
  1. All features (Baseline)       - return + macro features
  2. Return features only          - DD, Avg_Ret, Sortino
  3. Macro features only           - Yield, VIX, Stock_Bond_Corr

Generates a timestamped markdown report in benchmarks/.
"""
import sys
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
from main import (fetch_and_prepare_data, walk_forward_backtest, calculate_metrics,
                  OOS_START_DATE, END_DATE, TARGET_TICKER, LAMBDA_GRID, EWMA_HL_GRID,
                  TRANSACTION_COST)
from config import StrategyConfig

SUB_PERIODS = [
    ("GFC",        "2007-01-01", "2009-12-31"),
    ("Recovery",   "2010-01-01", "2015-12-31"),
    ("Late Cycle", "2016-01-01", "2019-12-31"),
    ("COVID",      "2020-01-01", "2021-12-31"),
    ("Post-COVID", "2022-01-01", "2025-12-31"),
]

ABLATION_CONFIGS = [
    StrategyConfig(name="All Features (Baseline)", feature_ablation="all"),
    StrategyConfig(name="Return Features Only", feature_ablation="return_only"),
    StrategyConfig(name="Macro Features Only", feature_ablation="macro_only"),
]

RETURN_FEATURE_NAMES = ["DD_log_5", "DD_log_21", "Avg_Ret_5", "Avg_Ret_10", "Avg_Ret_21",
                        "Sortino_5", "Sortino_10", "Sortino_21"]
MACRO_FEATURE_NAMES = ["Yield_2Y_EWMA_diff", "Yield_Slope_EWMA_10",
                       "Yield_Slope_EWMA_diff_21", "VIX_EWMA_log_diff", "Stock_Bond_Corr"]


def compute_subperiod_metrics(res_df):
    rows = []
    for label, start, end in SUB_PERIODS:
        mask = (res_df.index >= start) & (res_df.index <= end)
        sub = res_df.loc[mask]
        if len(sub) < 20:
            rows.append({"Period": label, "Days": 0})
            continue
        s_ret, s_vol, s_sharpe, s_sortino, s_mdd = calculate_metrics(sub['Strat_Return'], sub['RF_Rate'])
        b_ret, b_vol, b_sharpe, b_sortino, b_mdd = calculate_metrics(sub['Target_Return'], sub['RF_Rate'])
        rows.append({
            "Period": label, "Days": len(sub),
            "Sharpe": round(s_sharpe, 3), "B&H Sharpe": round(b_sharpe, 3),
            "Delta": round(s_sharpe - b_sharpe, 3),
            "Ann Ret (%)": round(s_ret * 100, 2), "B&H Ret (%)": round(b_ret * 100, 2),
            "Max DD (%)": round(s_mdd * 100, 2), "B&H MDD (%)": round(b_mdd * 100, 2),
        })
    return pd.DataFrame(rows).set_index("Period")


def compute_xgb_quality(res_df):
    """Compute XGBoost prediction quality metrics from the backtest results."""
    metrics = {}
    if 'Raw_Prob' not in res_df.columns or 'JM_Target_State' not in res_df.columns:
        return metrics

    probs = res_df['Raw_Prob']
    true_states = res_df['JM_Target_State']

    # Accuracy: predicted state (prob > 0.5) vs JM target state
    pred_states = (probs > 0.5).astype(int)
    metrics['accuracy'] = (pred_states == true_states).mean()

    # Brier score (lower is better)
    metrics['brier_score'] = ((probs - true_states) ** 2).mean()

    # Probability calibration: mean prob when actually bearish vs bullish
    bear_mask = true_states == 1
    bull_mask = true_states == 0
    metrics['mean_prob_when_bear'] = probs[bear_mask].mean() if bear_mask.any() else np.nan
    metrics['mean_prob_when_bull'] = probs[bull_mask].mean() if bull_mask.any() else np.nan

    # Bear detection rate (recall for state 1)
    if bear_mask.any():
        metrics['bear_recall'] = ((pred_states == 1) & bear_mask).sum() / bear_mask.sum()
    else:
        metrics['bear_recall'] = np.nan

    # Bull detection rate (recall for state 0)
    if bull_mask.any():
        metrics['bull_recall'] = ((pred_states == 0) & bull_mask).sum() / bull_mask.sum()
    else:
        metrics['bull_recall'] = np.nan

    # Probability distribution stats
    metrics['prob_mean'] = probs.mean()
    metrics['prob_std'] = probs.std()

    return metrics


def run_ablation():
    start_time = time.time()
    print("=" * 70)
    print("MACRO FEATURE ABLATION STUDY")
    print("=" * 70)

    df = fetch_and_prepare_data()

    # B&H baseline
    print("\nComputing Buy & Hold baseline...")
    baseline_df = walk_forward_backtest(df, StrategyConfig(name="_bh_baseline"))
    bh_ret, bh_vol, bh_sharpe, bh_sortino, bh_mdd = calculate_metrics(
        baseline_df['Target_Return'], baseline_df['RF_Rate']
    )

    results = []
    details = {}

    for config in ABLATION_CONFIGS:
        print(f"\nRunning: {config.name} (ablation={config.feature_ablation})...")
        res_df = walk_forward_backtest(df, config)
        ret, vol, sharpe, sortino, mdd = calculate_metrics(res_df['Strat_Return'], res_df['RF_Rate'])

        # XGB quality metrics
        xgb_quality = compute_xgb_quality(res_df)

        results.append({
            "Config": config.name,
            "Ann Ret (%)": round(ret * 100, 2),
            "Ann Vol (%)": round(vol * 100, 2),
            "Sharpe": round(sharpe, 3),
            "Sortino": round(sortino, 3),
            "Max DD (%)": round(mdd * 100, 2),
            "Sharpe Delta vs B&H": round(sharpe - bh_sharpe, 3),
        })

        details[config.name] = {
            'config': config,
            'res_df': res_df,
            'xgb_quality': xgb_quality,
            'subperiods': compute_subperiod_metrics(res_df),
            'lambda_history': res_df.attrs.get('lambda_history', []),
            'lambda_dates': res_df.attrs.get('lambda_dates', []),
            'ewma_halflife': res_df.attrs.get('ewma_halflife', None),
        }

    elapsed = time.time() - start_time
    results_df = pd.DataFrame(results).set_index("Config")

    # Console output
    print("\n" + "=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)
    print(results_df.to_string())
    print(f"\nB&H Reference: Sharpe={bh_sharpe:.3f}, Ann Ret={bh_ret*100:.2f}%, MDD={bh_mdd*100:.2f}%")

    print("\n--- XGBoost Prediction Quality ---")
    for name, det in details.items():
        q = det['xgb_quality']
        if q:
            print(f"\n[{name}]")
            print(f"  Accuracy:     {q.get('accuracy', 0):.3f}")
            print(f"  Brier Score:  {q.get('brier_score', 0):.4f}")
            print(f"  Bear Recall:  {q.get('bear_recall', 0):.3f}")
            print(f"  Bull Recall:  {q.get('bull_recall', 0):.3f}")

    print(f"\nRuntime: {elapsed:.1f}s")

    # Generate report
    save_ablation_report(results_df, details, bh_sharpe, bh_ret, bh_sortino, bh_mdd, elapsed)
    return results_df


def save_ablation_report(results_df, details, bh_sharpe, bh_ret, bh_sortino, bh_mdd, elapsed):
    os.makedirs("benchmarks", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"benchmarks/macro_ablation_{timestamp}.md"

    L = []
    L.append("# Macro Feature Ablation Study")
    L.append(f"\n**Generated:** {timestamp}")
    L.append(f"**Runtime:** {elapsed:.1f}s")
    L.append(f"**Target:** {TARGET_TICKER}")
    L.append(f"**OOS Period:** {OOS_START_DATE} to {END_DATE}")
    L.append(f"**Transaction Cost:** {TRANSACTION_COST * 10000:.1f} bps")
    L.append(f"**Lambda Grid:** {LAMBDA_GRID}")

    L.append("\n## Objective")
    L.append("")
    L.append("Isolate the contribution of macro features to the JM+XGBoost strategy by comparing:")
    L.append("1. **All Features (Baseline)** -- Return features (DD, Avg_Ret, Sortino) + Macro features (Yield, VIX, Stock_Bond_Corr)")
    L.append("2. **Return Features Only** -- Only the 8 return-based features used by both JM and XGBoost")
    L.append("3. **Macro Features Only** -- Only the 5 macro features (XGBoost-exclusive)")
    L.append("")
    L.append("The Jump Model always uses return features for regime identification.")
    L.append("Only the XGBoost feature set changes across configurations.")

    # Feature sets
    L.append("\n## Feature Sets")
    L.append("")
    L.append("| Group | Features | Count |")
    L.append("|---|---|---:|")
    L.append(f"| Return | {', '.join(RETURN_FEATURE_NAMES)} | {len(RETURN_FEATURE_NAMES)} |")
    L.append(f"| Macro | {', '.join(MACRO_FEATURE_NAMES)} | {len(MACRO_FEATURE_NAMES)} |")
    L.append(f"| All | Return + Macro | {len(RETURN_FEATURE_NAMES) + len(MACRO_FEATURE_NAMES)} |")

    # Strategy performance
    L.append("\n## Strategy Performance")
    L.append("")
    L.append(f"**Buy & Hold Reference:** Sharpe={bh_sharpe:.3f}, Ann Ret={bh_ret*100:.2f}%, MDD={bh_mdd*100:.2f}%\n")
    L.append("| Config | Sharpe | Sortino | Ann Ret (%) | Ann Vol (%) | Max DD (%) | Sharpe Delta vs B&H |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, row in results_df.iterrows():
        delta = row['Sharpe Delta vs B&H']
        delta_str = f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}"
        L.append(f"| {name} | {row['Sharpe']:.3f} | {row['Sortino']:.3f} | {row['Ann Ret (%)']:.2f} "
                 f"| {row['Ann Vol (%)']:.2f} | {row['Max DD (%)']:.2f} | {delta_str} |")

    # XGBoost prediction quality
    L.append("\n## XGBoost Prediction Quality")
    L.append("")
    L.append("| Config | Accuracy | Brier Score | Bear Recall | Bull Recall | P(bear) when Bear | P(bear) when Bull | Prob Mean | Prob Std |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, det in details.items():
        q = det['xgb_quality']
        if q:
            L.append(f"| {name} "
                     f"| {q.get('accuracy', 0):.3f} "
                     f"| {q.get('brier_score', 0):.4f} "
                     f"| {q.get('bear_recall', 0):.3f} "
                     f"| {q.get('bull_recall', 0):.3f} "
                     f"| {q.get('mean_prob_when_bear', 0):.3f} "
                     f"| {q.get('mean_prob_when_bull', 0):.3f} "
                     f"| {q.get('prob_mean', 0):.3f} "
                     f"| {q.get('prob_std', 0):.3f} |")
        else:
            L.append(f"| {name} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")

    # Sub-period analysis
    L.append("\n## Sub-Period Analysis")
    L.append("")
    for name, det in details.items():
        sub_df = det['subperiods']
        if sub_df.empty or 'Sharpe' not in sub_df.columns:
            continue
        L.append(f"### {name}")
        L.append("")
        L.append("| Period | Days | Sharpe | B&H Sharpe | Delta | Ann Ret (%) | Max DD (%) |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        for period, row in sub_df.iterrows():
            if row.get('Days', 0) == 0:
                L.append(f"| {period} | 0 | N/A | N/A | N/A | N/A | N/A |")
                continue
            delta = row.get('Delta', 0)
            delta_str = f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}"
            L.append(f"| {period} | {int(row['Days'])} | {row['Sharpe']:.3f} | {row['B&H Sharpe']:.3f} "
                     f"| {delta_str} | {row['Ann Ret (%)']:.2f} | {row['Max DD (%)']:.2f} |")
        L.append("")

    # Lambda stability
    L.append("## Lambda Stability")
    L.append("")
    for name, det in details.items():
        lh = det['lambda_history']
        ld = det['lambda_dates']
        if not lh:
            continue
        lambdas = np.array(lh)
        cv = lambdas.std() / lambdas.mean() if lambdas.mean() > 0 else float('nan')
        L.append(f"### {name}")
        L.append(f"- EWMA Halflife: {det['ewma_halflife']}")
        L.append(f"- Lambda Mean: {lambdas.mean():.2f}, Std: {lambdas.std():.2f}, CV: {cv:.2f}")
        L.append("")

    # Conclusions
    L.append("## Conclusions")
    L.append("")
    # Auto-generate conclusions based on data
    sharpes = {name: results_df.loc[name, 'Sharpe'] for name in results_df.index}
    best = max(sharpes, key=sharpes.get)
    worst = min(sharpes, key=sharpes.get)
    all_sharpe = sharpes.get("All Features (Baseline)", 0)
    ret_sharpe = sharpes.get("Return Features Only", 0)
    macro_sharpe = sharpes.get("Macro Features Only", 0)

    macro_lift = all_sharpe - ret_sharpe
    L.append(f"- **Macro feature lift:** {macro_lift:+.3f} Sharpe (All vs Return-only)")
    L.append(f"- **Best config:** {best} (Sharpe {sharpes[best]:.3f})")
    L.append(f"- **Worst config:** {worst} (Sharpe {sharpes[worst]:.3f})")
    if macro_sharpe < ret_sharpe:
        L.append("- Macro features alone are insufficient for regime forecasting — return features carry the primary signal")
    elif macro_sharpe > ret_sharpe:
        L.append("- Macro features alone outperform return features — they contain strong complementary regime signal")
    if macro_lift > 0.05:
        L.append("- Macro features provide meaningful incremental value on top of return features")
    elif macro_lift < -0.05:
        L.append("- Adding macro features hurts performance — potential overfitting or noise injection")
    else:
        L.append("- Macro features have marginal impact — the strategy is primarily driven by return features")
    L.append("")

    report = "\n".join(L) + "\n"
    with open(filepath, 'w') as f:
        f.write(report)
    print(f"\nReport saved: {filepath}")
    return filepath


if __name__ == "__main__":
    run_ablation()
