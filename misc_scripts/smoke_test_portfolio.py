#!/usr/bin/env python3
"""Smoke test for portfolio.py — runs 2 BBG assets end-to-end."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd

import portfolio
import main as backend

# Use a subset for speed
portfolio.BBG_ASSETS = portfolio.BBG_ASSETS[:3]  # LargeCap, MidCap, SmallCap

print("=" * 70)
print("Smoke test: 3 BBG assets, 2007-2010")
print("=" * 70)

t0 = time.time()
signals = portfolio.compute_asset_signals(
    universe='bloomberg',
    oos_start='2007-01-01',
    oos_end='2010-12-31',
    force_refresh=True,
)
print(f"\n✓ Computed signals in {time.time()-t0:.1f}s")
print(f"  assets: {list(signals.keys())}")
for name, df in signals.items():
    oos = df[(df.index >= '2007-01-01') & (df.index <= '2010-12-31')]
    bear_pct = (oos['Forecast_State'] == 1).mean() * 100 if 'Forecast_State' in oos.columns else 0
    print(f"  {name:<12} rows={len(oos)}  bear={bear_pct:.1f}%")

print("\nBuilding panel...")
panel = portfolio.build_asset_panel(signals, '2007-01-01', '2010-12-31')
print(f"  returns shape: {panel.returns.shape}")
print(f"  forecast shape: {panel.forecast.shape}")

print("\nRunning portfolios (3 assets — bullish-count rule will trip → cash often)...")
t0 = time.time()
results = portfolio.run_all_portfolios(panel, rebal_freq='monthly')
print(f"✓ Done in {time.time()-t0:.1f}s")

print("\nMetrics:")
for label, res in results.items():
    m = portfolio.portfolio_metrics(res['returns'], panel.rf_daily, res['weights'], res['turnover_daily'])
    print(f"  {label:<18}  ret={m['Return']*100:+.2f}%  vol={m['Volatility']*100:.2f}%  S={m['Sharpe']:.2f}  "
          f"MDD={m['MDD']*100:.1f}%  turnover={m['Turnover']:.2f}  lev={m['Leverage']:.2f}")

print("\nDone.")
