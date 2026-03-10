"""
Diagnostic: Investigate why JM-XGB doesn't significantly beat B&H for ^SP500TR (2007-2023).

Paper reports Sharpe 0.79 vs B&H 0.50 (delta +0.29).
We get ~0.57 vs ~0.50 (delta +0.07).

This script decomposes the pipeline into steps:
1. JM regime quality - are regimes correctly identified?
2. XGBoost forecast quality - is the classifier actually predictive?
3. Oracle analysis - what if we had perfect regime knowledge?
4. Lambda sensitivity - how much does lambda selection matter?
5. EWMA smoothing impact - does it help or hurt?
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Override to paper period
os.environ['XGB_END_DATE'] = '2024-01-01'  # Paper: 2007-2023

import numpy as np
import pandas as pd
from collections import defaultdict

import main as backend
from config import StrategyConfig

backend.END_DATE = '2024-01-01'
backend._forecast_cache.clear()

config = StrategyConfig()

print("=" * 80)
print("DIAGNOSTIC: JM-XGB Gap Analysis for ^SP500TR (2007-2023)")
print("=" * 80)

# ---------- Load data ----------
df = backend.fetch_and_prepare_data()
oos_start = pd.to_datetime('2007-01-01')
oos_end = pd.to_datetime('2024-01-01')
oos_df = df[(df.index >= oos_start) & (df.index < oos_end)].copy()

print(f"\nOOS period: {oos_df.index[0].date()} to {oos_df.index[-1].date()} ({len(oos_df)} days)")

# ---------- B&H baseline ----------
_, _, bh_sharpe, bh_sortino, bh_mdd = backend.calculate_metrics(oos_df['Target_Return'], oos_df['RF_Rate'])
print(f"\nB&H:  Sharpe={bh_sharpe:.3f}, Sortino={bh_sortino:.3f}, MDD={bh_mdd:.1%}")

# ==========================================================================
# TEST 1: Run full walk-forward backtest (our current implementation)
# ==========================================================================
print("\n" + "=" * 80)
print("TEST 1: Full Walk-Forward Backtest (current implementation)")
print("=" * 80)

backend._forecast_cache.clear()
result_df = backend.walk_forward_backtest(df, config)
_, _, strat_sharpe, strat_sortino, strat_mdd = backend.calculate_metrics(
    result_df['Strat_Return'], result_df['RF_Rate'])

print(f"Strategy: Sharpe={strat_sharpe:.3f}, Sortino={strat_sortino:.3f}, MDD={strat_mdd:.1%}")
print(f"Delta vs B&H: Sharpe={strat_sharpe - bh_sharpe:+.3f}")
print(f"Lambda history: {result_df.attrs.get('lambda_history', 'N/A')}")
print(f"EWMA halflife: {result_df.attrs.get('ewma_halflife', 'N/A')}")

# Analyze signal stats
if 'Forecast_State' in result_df.columns:
    bear_pct = result_df['Forecast_State'].mean()
    # Count regime switches
    switches = (result_df['Forecast_State'].diff().abs() > 0).sum()
    print(f"Bear forecast %: {bear_pct:.1%}, Regime switches: {switches}")

# ==========================================================================
# TEST 2: JM Regime Quality Analysis
# ==========================================================================
print("\n" + "=" * 80)
print("TEST 2: JM Regime Quality Analysis")
print("=" * 80)

# Collect JM states across all OOS periods for a few representative lambdas
test_lambdas = [0.0, 1.0, 5.0, 10.0, 21.54, 46.42, 100.0]
return_features = [c for c in df.columns if c.startswith(('DD_', 'Avg_Ret_', 'Sortino_'))]

print("\nFor each lambda, fit JM on 11yr lookback at 2007-01-01, analyze in-sample regimes:")
train_start = oos_start - pd.DateOffset(years=11)
train_data = df[(df.index >= train_start) & (df.index < oos_start)].copy()

X_jm = train_data[return_features]
X_jm_std = (X_jm - X_jm.mean()) / X_jm.std()

print(f"\nTraining window: {train_data.index[0].date()} to {train_data.index[-1].date()} ({len(train_data)} days)")
print(f"Return features: {return_features}")

for lmbda in test_lambdas:
    jm = backend.StatisticalJumpModel(n_states=2, lambda_penalty=lmbda)
    states = jm.fit_predict(X_jm_std.values)

    # Align states
    cum_ret_0 = train_data['Excess_Return'][states == 0].sum()
    cum_ret_1 = train_data['Excess_Return'][states == 1].sum()
    if cum_ret_1 > cum_ret_0:
        states = 1 - states

    n_bull = (states == 0).sum()
    n_bear = (states == 1).sum()
    switches_count = (np.diff(states) != 0).sum()

    # Average return in each regime
    bull_ret = train_data['Excess_Return'][states == 0].mean() * 252
    bear_ret = train_data['Excess_Return'][states == 1].mean() * 252

    print(f"  λ={lmbda:6.1f}: Bull={n_bull:4d}d ({n_bull/len(states):.0%}), Bear={n_bear:4d}d ({n_bear/len(states):.0%}), "
          f"Switches={switches_count:3d}, Bull ret={bull_ret:+.1%}, Bear ret={bear_ret:+.1%}")

# ==========================================================================
# TEST 3: XGBoost Forecast Accuracy per 6-month chunk
# ==========================================================================
print("\n" + "=" * 80)
print("TEST 3: XGBoost Forecast Accuracy Per Period")
print("=" * 80)

# Use the lambda history from the walk-forward backtest
lambda_hist = result_df.attrs.get('lambda_history', [])
lambda_dates_list = result_df.attrs.get('lambda_dates', [])

print(f"\nPeriod-by-period XGBoost accuracy (class 1 = bearish):")
print(f"{'Period':<25} {'Lambda':>8} {'Accuracy':>10} {'Bear%Pred':>10} {'Bear%True':>10} {'AvgProb':>10} {'StdProb':>10}")
print("-" * 90)

backend._forecast_cache.clear()
current = oos_start
period_idx = 0
all_raw_probs = []
all_jm_targets = []

while current < oos_end:
    if period_idx < len(lambda_hist):
        lmbda = lambda_hist[period_idx]
    else:
        lmbda = 10.0  # fallback

    chunk = backend.run_period_forecast(df, current, lmbda, config, include_xgboost=True)
    if chunk is not None and 'Raw_Prob' in chunk.columns and 'JM_Target_State' in chunk.columns:
        raw_prob = chunk['Raw_Prob']
        jm_target = chunk['JM_Target_State']

        all_raw_probs.extend(raw_prob.values)
        all_jm_targets.extend(jm_target.values)

        pred_bear = (raw_prob > 0.5).astype(int)
        accuracy = (pred_bear == jm_target).mean()
        bear_pct_pred = pred_bear.mean()
        bear_pct_true = jm_target.mean()

        period_str = f"{current.strftime('%Y-%m')}"
        chunk_end = min(current + pd.DateOffset(months=6), oos_end)
        period_str += f" to {chunk_end.strftime('%Y-%m')}"

        print(f"{period_str:<25} {lmbda:8.2f} {accuracy:10.1%} {bear_pct_pred:10.1%} {bear_pct_true:10.1%} "
              f"{raw_prob.mean():10.3f} {raw_prob.std():10.3f}")

    current += pd.DateOffset(months=6)
    period_idx += 1

# Overall XGB accuracy
all_raw_probs = np.array(all_raw_probs)
all_jm_targets = np.array(all_jm_targets)
if len(all_raw_probs) > 0:
    overall_pred = (all_raw_probs > 0.5).astype(int)
    overall_acc = (overall_pred == all_jm_targets).mean()
    print(f"\n{'OVERALL':<25} {'':>8} {overall_acc:10.1%} {overall_pred.mean():10.1%} {all_jm_targets.mean():10.1%} "
          f"{all_raw_probs.mean():10.3f} {all_raw_probs.std():10.3f}")

# ==========================================================================
# TEST 4: Oracle Analysis - What if we had perfect JM regime knowledge?
# ==========================================================================
print("\n" + "=" * 80)
print("TEST 4: Oracle Analysis - Perfect JM Regime Knowledge")
print("=" * 80)

# Use JM_Target_State as the actual signal (no XGBoost, no EWMA, just JM online states)
if 'JM_Target_State' in result_df.columns:
    oracle_signals = result_df['JM_Target_State'].shift(1).fillna(0)
    oracle_alloc = 1.0 - oracle_signals  # 0=bull → invest, 1=bear → cash
    oracle_ret = (oracle_alloc * result_df['Target_Return']) + ((1.0 - oracle_alloc) * result_df['RF_Rate'])
    oracle_trades = oracle_alloc.diff().abs().fillna(0)
    oracle_ret_net = oracle_ret - (oracle_trades * backend.TRANSACTION_COST)

    _, _, oracle_sharpe, oracle_sortino, oracle_mdd = backend.calculate_metrics(oracle_ret_net, result_df['RF_Rate'])
    print(f"Oracle JM (online states): Sharpe={oracle_sharpe:.3f}, MDD={oracle_mdd:.1%}")
    print(f"Delta vs B&H: {oracle_sharpe - bh_sharpe:+.3f}")

    # True oracle: use actual positive/negative return days
    true_oracle_signal = (result_df['Target_Return'] < 0).astype(int).shift(1).fillna(0)
    true_oracle_alloc = 1.0 - true_oracle_signal
    true_oracle_ret = (true_oracle_alloc * result_df['Target_Return']) + ((1.0 - true_oracle_alloc) * result_df['RF_Rate'])
    true_oracle_ret_net = true_oracle_ret - (true_oracle_alloc.diff().abs().fillna(0) * backend.TRANSACTION_COST)
    _, _, true_oracle_sharpe, _, _ = backend.calculate_metrics(true_oracle_ret_net, result_df['RF_Rate'])
    print(f"\nTrue Oracle (avoid neg days, 1-day lag): Sharpe={true_oracle_sharpe:.3f}")

# ==========================================================================
# TEST 5: Fixed Lambda Sweep (bypass walk-forward tuning)
# ==========================================================================
print("\n" + "=" * 80)
print("TEST 5: Fixed Lambda Sweep (no walk-forward tuning)")
print("=" * 80)

print(f"\n{'Lambda':>8} {'Sharpe':>8} {'Sortino':>8} {'MDD':>8} {'Delta':>8} {'Bear%':>8} {'Switches':>8}")
print("-" * 60)

best_fixed_sharpe = -np.inf
best_fixed_lambda = None

for lmbda in [0.0, 1.0, 2.15, 4.64, 10.0, 21.54, 46.42, 100.0]:
    backend._forecast_cache.clear()

    # Run simulate_strategy over the full OOS period with fixed lambda
    res = backend.simulate_strategy(df, oos_start, oos_end, lmbda, config,
                                     include_xgboost=True, ewma_halflife=8)
    if not res.empty:
        _, _, sharpe, sortino, mdd = backend.calculate_metrics(res['Strat_Return'], res['RF_Rate'])
        bear_pct = res['Forecast_State'].mean() if 'Forecast_State' in res.columns else 0
        switches_count = (res['Forecast_State'].diff().abs() > 0).sum() if 'Forecast_State' in res.columns else 0

        delta = sharpe - bh_sharpe
        print(f"{lmbda:8.2f} {sharpe:8.3f} {sortino:8.3f} {mdd:8.1%} {delta:+8.3f} {bear_pct:8.1%} {switches_count:8d}")

        if sharpe > best_fixed_sharpe:
            best_fixed_sharpe = sharpe
            best_fixed_lambda = lmbda

print(f"\nBest fixed lambda: {best_fixed_lambda} → Sharpe={best_fixed_sharpe:.3f} (delta={best_fixed_sharpe - bh_sharpe:+.3f})")

# ==========================================================================
# TEST 6: EWMA Smoothing Impact
# ==========================================================================
print("\n" + "=" * 80)
print("TEST 6: EWMA Smoothing Impact (fixed lambda=best)")
print("=" * 80)

print(f"\nUsing lambda={best_fixed_lambda}")
print(f"{'EWMA HL':>8} {'Sharpe':>8} {'Delta':>8} {'Bear%':>8}")
print("-" * 40)

for hl in [0, 2, 4, 8, 16]:
    backend._forecast_cache.clear()
    res = backend.simulate_strategy(df, oos_start, oos_end, best_fixed_lambda, config,
                                     include_xgboost=True, ewma_halflife=hl)
    if not res.empty:
        _, _, sharpe, _, _ = backend.calculate_metrics(res['Strat_Return'], res['RF_Rate'])
        bear_pct = res['Forecast_State'].mean() if 'Forecast_State' in res.columns else 0
        print(f"{hl:8d} {sharpe:8.3f} {sharpe - bh_sharpe:+8.3f} {bear_pct:8.1%}")

# ==========================================================================
# TEST 7: JM-only vs JM-XGB comparison
# ==========================================================================
print("\n" + "=" * 80)
print("TEST 7: JM-only vs JM-XGB (fixed lambda sweep)")
print("=" * 80)

print(f"\n{'Lambda':>8} {'JM Sharpe':>10} {'XGB Sharpe':>11} {'XGB Gain':>10}")
print("-" * 45)

for lmbda in [0.0, 1.0, 10.0, 21.54, 46.42, 100.0]:
    backend._forecast_cache.clear()

    # JM only
    res_jm = backend.simulate_strategy(df, oos_start, oos_end, lmbda, config, include_xgboost=False)
    # JM + XGB
    res_xgb = backend.simulate_strategy(df, oos_start, oos_end, lmbda, config, include_xgboost=True, ewma_halflife=8)

    jm_sharpe = xgb_sharpe = 0
    if not res_jm.empty:
        _, _, jm_sharpe, _, _ = backend.calculate_metrics(res_jm['Strat_Return'], res_jm['RF_Rate'])
    if not res_xgb.empty:
        _, _, xgb_sharpe, _, _ = backend.calculate_metrics(res_xgb['Strat_Return'], res_xgb['RF_Rate'])

    print(f"{lmbda:8.2f} {jm_sharpe:10.3f} {xgb_sharpe:11.3f} {xgb_sharpe - jm_sharpe:+10.3f}")

# ==========================================================================
# TEST 8: Sub-period analysis for the walk-forward result
# ==========================================================================
print("\n" + "=" * 80)
print("TEST 8: Sub-Period Analysis")
print("=" * 80)

periods = {
    'GFC (2007-2009)': ('2007-01-01', '2010-01-01'),
    'Recovery (2010-2015)': ('2010-01-01', '2016-01-01'),
    'Late Cycle (2016-2019)': ('2016-01-01', '2020-01-01'),
    'COVID (2020-2021)': ('2020-01-01', '2022-01-01'),
    'Post-COVID (2022-2023)': ('2022-01-01', '2024-01-01'),
}

print(f"\n{'Period':<25} {'Strat Sharpe':>12} {'B&H Sharpe':>12} {'Delta':>8} {'Bear%':>8}")
print("-" * 70)

for name, (ps, pe) in periods.items():
    mask = (result_df.index >= ps) & (result_df.index < pe)
    sub = result_df[mask]

    if len(sub) < 20:
        continue

    oos_sub = oos_df[(oos_df.index >= ps) & (oos_df.index < pe)]

    _, _, s_sharpe, _, _ = backend.calculate_metrics(sub['Strat_Return'], sub['RF_Rate'])
    _, _, b_sharpe, _, _ = backend.calculate_metrics(oos_sub['Target_Return'], oos_sub['RF_Rate'])

    bear_pct = sub['Forecast_State'].mean() if 'Forecast_State' in sub.columns else 0

    print(f"{name:<25} {s_sharpe:12.3f} {b_sharpe:12.3f} {s_sharpe - b_sharpe:+8.3f} {bear_pct:8.1%}")

# ==========================================================================
# TEST 9: Raw probability distribution analysis
# ==========================================================================
print("\n" + "=" * 80)
print("TEST 9: Raw Probability Distribution")
print("=" * 80)

if len(all_raw_probs) > 0:
    print(f"\nRaw P(bearish) statistics:")
    print(f"  Mean:   {all_raw_probs.mean():.3f}")
    print(f"  Median: {np.median(all_raw_probs):.3f}")
    print(f"  Std:    {all_raw_probs.std():.3f}")
    print(f"  Min:    {all_raw_probs.min():.3f}")
    print(f"  Max:    {all_raw_probs.max():.3f}")
    print(f"  P > 0.5: {(all_raw_probs > 0.5).mean():.1%}")
    print(f"  P > 0.6: {(all_raw_probs > 0.6).mean():.1%}")
    print(f"  P > 0.7: {(all_raw_probs > 0.7).mean():.1%}")
    print(f"  P < 0.3: {(all_raw_probs < 0.3).mean():.1%}")

    # Calibration: when XGB says bear, is it actually bear?
    bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
    print(f"\n  Calibration (XGB prob bins vs JM online target):")
    print(f"  {'Prob Range':<15} {'Count':>8} {'Actual Bear%':>12} {'Avg Return':>12}")

    # Align with returns
    if 'Raw_Prob' in result_df.columns:
        for lo, hi in bins:
            mask = (result_df['Raw_Prob'] >= lo) & (result_df['Raw_Prob'] < hi)
            n = mask.sum()
            if n > 0:
                jm_bear = result_df.loc[mask, 'JM_Target_State'].mean() if 'JM_Target_State' in result_df.columns else float('nan')
                avg_ret = result_df.loc[mask, 'Target_Return'].mean() * 252
                print(f"  [{lo:.1f}, {hi:.1f}){'':<6} {n:8d} {jm_bear:12.1%} {avg_ret:+12.1%}")


print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
B&H Sharpe:               {bh_sharpe:.3f}
Walk-Forward Sharpe:       {strat_sharpe:.3f}  (delta {strat_sharpe - bh_sharpe:+.3f})
Best Fixed Lambda Sharpe:  {best_fixed_sharpe:.3f}  (delta {best_fixed_sharpe - bh_sharpe:+.3f}, λ={best_fixed_lambda})
Paper reports:             0.79  (delta +0.29 vs B&H 0.50)
""")
