#!/usr/bin/env python3
"""Run the portfolio reproduction on Bloomberg data, 2007-2023, and compare
to paper Tables 6 & 7. This is the canonical paper-comparison script.

After running, the on-disk cache cache/portfolio_signals_bloomberg_2007-01_2023-12.pkl
will be populated, making the Streamlit page instant on the first visit.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd

import portfolio
import main as backend

OOS_START = '2007-01-01'
OOS_END   = '2023-12-31'

print("=" * 78)
print("Portfolio reproduction: Bloomberg, 2007-2023")
print("=" * 78)

def cb(i, n, name):
    print(f"  [{i+1}/{n}] {name}")

t0 = time.time()
signals = portfolio.compute_asset_signals(
    universe='bloomberg',
    oos_start=OOS_START, oos_end=OOS_END,
    force_refresh=False,
    progress_callback=cb,
)
elapsed = time.time() - t0
print(f"\n✓ Computed signals for {len(signals)}/12 assets in {elapsed/60:.1f} min")

print("\nLoading in-sample regime means (paper-faithful MV(JM-XGB) μ)…")
t0 = time.time()
insample_mu = portfolio.compute_insample_regime_means(
    universe='bloomberg', oos_start=OOS_START, oos_end=OOS_END,
    progress_callback=cb)
print(f"✓ In-sample μ ready in {(time.time()-t0)/60:.1f} min")

panel = portfolio.build_asset_panel(signals, OOS_START, OOS_END, insample_mu=insample_mu)
print(f"Panel: {len(panel.returns)} days × {len(panel.asset_order)} assets")
print(f"  Date range: {panel.returns.index.min().date()} → {panel.returns.index.max().date()}")

print("\nRegime stats per asset:")
for a in panel.asset_order:
    pct_bear = (panel.forecast[a] == 1).mean() * 100
    shifts   = int(panel.forecast[a].diff().abs().fillna(0).sum())
    print(f"  {a:<12}  %bear={pct_bear:5.1f}%   shifts={shifts}")

print("\n--- Running 7 portfolios (daily rebalance) ---")
t0 = time.time()
results = portfolio.run_all_portfolios(panel, rebal_freq='daily', progress_callback=cb)
print(f"\n✓ Backtests done in {(time.time()-t0)/60:.2f} min")

# Build Table 6
print("\n" + "=" * 78)
print("TABLE 6 — Portfolio performance (annualized excess returns)")
print("=" * 78)

rows = {}
for label in portfolio.STRATEGY_LABELS:
    if label not in results:
        continue
    rows[label] = portfolio.portfolio_metrics(results[label]['returns'], panel.rf_daily,
                                              results[label]['weights'],
                                              results[label]['turnover_daily'])

ours = pd.DataFrame(rows)
paper = portfolio.PAPER_TABLE_6.reindex(columns=ours.columns)

print("\n[Ours]")
print(ours.to_string(formatters={
    c: (lambda v: f"{v*100:+5.1f}%" if abs(v) < 1 else f"{v:.2f}") for c in ours.columns
}))

print("\n[Paper]")
print(paper.to_string(formatters={
    c: (lambda v: f"{v*100:+5.1f}%" if abs(v) < 1 else f"{v:.2f}") for c in paper.columns
}))

print("\nSharpe gap (ours − paper):")
print((ours.loc['Sharpe'] - paper.loc['Sharpe']).round(3).to_string())

print("\nMDD gap (ours − paper):")
print(((ours.loc['MDD'] - paper.loc['MDD']) * 100).round(2).to_string())

# Build Table 7
print("\n" + "=" * 78)
print("TABLE 7 — Forecast correlation with realized")
print("=" * 78)

ewma_mu  = results.get('MV', {}).get('mu_forecast', pd.DataFrame())
jmxgb_mu = results.get('MV(JM-XGB)', {}).get('mu_forecast', pd.DataFrame())

tbl7_ewma  = portfolio.forecast_correlation(panel, ewma_mu)
tbl7_jmxgb = portfolio.forecast_correlation(panel, jmxgb_mu)

idx_order = ['Overall'] + panel.asset_order
tbl7_ours = pd.DataFrame({
    'EWMA_ours':   tbl7_ewma.reindex(idx_order),
    'EWMA_paper':  portfolio.PAPER_TABLE_7.reindex(idx_order)['EWMA'],
    'JMXGB_ours':  tbl7_jmxgb.reindex(idx_order),
    'JMXGB_paper': portfolio.PAPER_TABLE_7.reindex(idx_order)['JM-XGB'],
})
print(tbl7_ours.to_string(formatters={c: '{:+.4f}'.format for c in tbl7_ours.columns}))

print("\nDone.")
