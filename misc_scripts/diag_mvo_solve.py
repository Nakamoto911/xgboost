#!/usr/bin/env python3
"""Inspect the MVO solution at a sample rebalance date."""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import portfolio

OOS_START, OOS_END = '2007-01-01', '2023-12-31'
with open(portfolio._signal_cache_path('bloomberg', OOS_START, OOS_END), 'rb') as f:
    signals = pickle.load(f)
panel = portfolio.build_asset_panel(signals, OOS_START, OOS_END)


def trace_at(date_str: str):
    date = panel.returns.index[panel.returns.index <= pd.Timestamp(date_str)][-1]
    history = panel.returns.loc[:date].iloc[:-1]
    fc_today = panel.forecast.loc[date].reindex(panel.asset_order)

    bull = (fc_today == 0).astype(int).values
    n_bull = int(bull.sum())

    mu_window_start = date - pd.DateOffset(years=11)
    mu_history = history.loc[mu_window_start:]
    fc_window = panel.forecast.reindex(mu_history.index).fillna(0).astype(int)

    mu = portfolio.regime_conditional_mu(
        mu_history[panel.asset_order],
        fc_window[panel.asset_order],
        fc_today, lookback_days=len(mu_history)).values

    Sigma = portfolio._cov_window(history[panel.asset_order], 252)

    print(f"\n{'='*78}\nDate: {date.date()}   bullish count: {n_bull}")
    if n_bull <= 3:
        print("  → ≤3 bullish: forced 100% cash (no MVO solve)")
        return

    print(f"  μ_ann = {(mu*252*100).round(2)}")
    print(f"  σ_ann (diag) = {(np.sqrt(np.diag(Sigma)) * np.sqrt(252) * 100).round(2)}")
    eig = np.linalg.eigvalsh(Sigma)
    print(f"  Σ eigenvalues × 1e4: {(eig*1e4).round(3)}")

    # Solve under different gamma_risk
    for gr in [1.0, 3.0, 5.0, 10.0]:
        w = portfolio.solve_mvo(mu, Sigma, np.zeros(len(mu)), gamma_risk=gr,
                                gamma_trade=1.0, tc_oneway=0.0005, w_ub=0.40, L=1.0,
                                active=bull.astype(float))
        print(f"  γ_risk={gr:>4.1f}  sum(w)={w.sum():.3f}  "
              f"w={w.round(3).tolist()}")


for d in ['2008-12-30', '2011-06-30', '2015-06-30', '2018-12-31', '2021-06-30', '2023-06-30']:
    trace_at(d)
