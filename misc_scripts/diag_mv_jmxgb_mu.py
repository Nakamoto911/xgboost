#!/usr/bin/env python3
"""Diagnose why MV(JM-XGB) leverage is 0.28 vs paper 0.86.

Checks: distribution of μ (regime-conditional mean) values fed into the MVO,
how often the ≤3-bullish rule fires, and what an oracle/alternative μ would do.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import pandas as pd

import portfolio

OOS_START, OOS_END = '2007-01-01', '2023-12-31'
cache_path = portfolio._signal_cache_path('bloomberg', OOS_START, OOS_END)
with open(cache_path, 'rb') as f:
    signals = pickle.load(f)

panel = portfolio.build_asset_panel(signals, OOS_START, OOS_END)

print("=" * 70)
print("Bullish-count histogram (out of 12 assets)")
print("=" * 70)
bull_count = (panel.forecast == 0).sum(axis=1)
print(bull_count.value_counts().sort_index().to_string())
print(f"\n  Days with ≤3 bullish (forced cash): {(bull_count <= 3).sum()} / {len(bull_count)}  "
      f"({(bull_count <= 3).mean()*100:.1f}%)")
print(f"  Mean bullish count: {bull_count.mean():.2f}")

# Sample the μ values at biannual rebalance dates
print("\n" + "=" * 70)
print("μ samples at semi-annual rebalance dates (regime-conditional mean × 252)")
print("Annualized = daily μ * 252. Paper expects ~3-15% annualized for bullish.")
print("=" * 70)

rebalance_dates = pd.date_range(OOS_START, OOS_END, freq='6MS')
rows = []
for date in rebalance_dates:
    # Use closest available date in panel
    eligible = panel.returns.index[panel.returns.index <= date]
    if len(eligible) == 0:
        continue
    d = eligible[-1]
    history = panel.returns.loc[:d].iloc[:-1]
    if len(history) < 30:
        continue
    mu_window_start = d - pd.DateOffset(years=11)
    mu_history = history.loc[mu_window_start:]
    fc_window = panel.forecast.reindex(mu_history.index).fillna(0).astype(int)
    fc_today = panel.forecast.loc[d].reindex(panel.asset_order)

    mu = portfolio.regime_conditional_mu(
        mu_history[panel.asset_order],
        fc_window[panel.asset_order],
        fc_today, lookback_days=len(mu_history))

    bull = (fc_today == 0).astype(int).values
    n_bull = int(bull.sum())
    rows.append({
        'date': d.date(), 'n_bull': n_bull,
        'mu_avg_bull_ann_%': float(mu[bull == 1].mean() * 252 * 100) if n_bull > 0 else np.nan,
        'mu_avg_bear_ann_%': float(mu[bull == 0].mean() * 252 * 100) if n_bull < 12 else np.nan,
        'mu_min_ann_%': float(mu.min() * 252 * 100),
        'mu_max_ann_%': float(mu.max() * 252 * 100),
    })

df_diag = pd.DataFrame(rows)
print(df_diag.to_string(index=False, float_format='%+.2f'))

print(f"\n  Average bullish μ (annualized): {df_diag['mu_avg_bull_ann_%'].mean():.2f}%")
print(f"  Average bearish μ (annualized): {df_diag['mu_avg_bear_ann_%'].mean():.2f}%")
print(f"  Bullish μ pre-2012 (annualized): "
      f"{df_diag[df_diag['date'].astype(str) < '2012']['mu_avg_bull_ann_%'].mean():.2f}%")
print(f"  Bullish μ post-2012 (annualized): "
      f"{df_diag[df_diag['date'].astype(str) >= '2012']['mu_avg_bull_ann_%'].mean():.2f}%")

# Now look at how the lookback's pre-OOS data (no forecast available) is contaminating
print("\n" + "=" * 70)
print("Forecast availability in 11y lookback (% days with real OOS forecast)")
print("=" * 70)
for d in [pd.Timestamp('2008-01-02'), pd.Timestamp('2012-01-02'),
          pd.Timestamp('2018-01-02'), pd.Timestamp('2023-01-02')]:
    eligible = panel.returns.index[panel.returns.index <= d]
    if len(eligible) == 0:
        continue
    actual = eligible[-1]
    lb_start = actual - pd.DateOffset(years=11)
    lb_dates = panel.returns.index[(panel.returns.index >= lb_start)
                                    & (panel.returns.index <= actual)]
    pct_oos = (lb_dates >= pd.Timestamp(OOS_START)).mean() * 100
    print(f"  {actual.date()}  lookback {lb_start.date()} → {actual.date()}  "
          f"% OOS forecasts available: {pct_oos:.1f}%")
