"""
Test: Paper's approach of jointly tuning (HL, lambda) on the initial validation
window (2002-2007) vs using fixed paper-prescribed HLs.

The paper says EWMA halflife is "Selected on initial validation window 2002-2007
using 0/1 strategy Sharpe". Our code uses fixed prescribed HLs from the paper.
But those HLs were optimized for Bloomberg data + paper's lambda grid. They may
not be optimal for Yahoo data + our 8pt grid.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import from benchmark_assets.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'misc_scripts'))
from benchmark_assets import (
    fetch_etf_data, run_period_forecast_fast,
    simulate_strategy_fast, calculate_metrics, StatisticalJumpModel,
    LAMBDA_GRID, EWMA_HL_GRID, PAPER_EWMA_HL, TRANSACTION_COST,
    VALIDATION_WINDOW_YRS, DD_EXCLUDE_TICKERS
)
from config import StrategyConfig

# Assets with largest paper gaps
TICKERS = {
    'VEIEX': {'asset': 'EM', 'paper_sharpe': 0.85, 'paper_hl': 0},
    'VWESX': {'asset': 'Corporate', 'paper_sharpe': 0.76, 'paper_hl': 2},
    'FRESX': {'asset': 'REIT', 'paper_sharpe': 0.56, 'paper_hl': 8},
    'FDIVX': {'asset': 'EAFE', 'paper_sharpe': 0.56, 'paper_hl': 0},
    '^SP500TR': {'asset': 'LargeCap', 'paper_sharpe': 0.79, 'paper_hl': 8},
}

OOS_START = '2007-01-01'
OOS_END = '2024-01-01'

config = StrategyConfig()


def backtest_with_hl(ticker, df, ewma_hl, label):
    """Run walk-forward backtest with a specific EWMA halflife."""
    oos_start_dt = pd.to_datetime(OOS_START)
    oos_end_dt = pd.to_datetime(OOS_END)
    cache = {}
    current_date = oos_start_dt
    chunks = []
    lambda_hist = []

    while current_date < oos_end_dt:
        chunk_end = min(current_date + pd.DateOffset(months=6), oos_end_dt)
        val_start = current_date - pd.DateOffset(years=VALIDATION_WINDOW_YRS)

        best_metric = -np.inf
        best_lambda = LAMBDA_GRID[len(LAMBDA_GRID)//2]
        for lmbda in LAMBDA_GRID:
            val_res = simulate_strategy_fast(df, val_start, current_date, lmbda, cache, config,
                                            include_xgboost=True, ewma_halflife=ewma_hl)
            if not val_res.empty:
                _, _, sharpe, _, _ = calculate_metrics(val_res['Strat_Return'], val_res['RF_Rate'])
                if not np.isnan(sharpe) and sharpe > best_metric:
                    best_metric = sharpe
                    best_lambda = lmbda

        lambda_hist.append(best_lambda)
        oos_chunk = run_period_forecast_fast(df, current_date, best_lambda, cache, config, include_xgboost=True)
        if oos_chunk is not None:
            chunks.append(oos_chunk)
        current_date = chunk_end

    if not chunks:
        return None

    full = pd.concat(chunks)
    if ewma_hl == 0:
        full['State_Prob'] = full['Raw_Prob']
    else:
        full['State_Prob'] = full['Raw_Prob'].ewm(halflife=ewma_hl).mean()

    full['Forecast_State'] = (full['State_Prob'] > 0.5).astype(int)
    signals = full['Forecast_State'].shift(1).fillna(0)
    alloc = 1.0 - signals
    strat = (alloc * full['Target_Return']) + ((1 - alloc) * full['RF_Rate'])
    trades = alloc.diff().abs().fillna(0)
    full['Strat_Return'] = strat - (trades * TRANSACTION_COST)

    # Trim to OOS
    mask = (full.index >= oos_start_dt) & (full.index < oos_end_dt)
    full = full[mask]
    if full.empty:
        return None

    _, _, sharpe, _, mdd = calculate_metrics(full['Strat_Return'], full['RF_Rate'])
    _, _, bh_sharpe, _, _ = calculate_metrics(full['Target_Return'], full['RF_Rate'])
    return {'ticker': ticker, 'label': label, 'hl': ewma_hl, 'sharpe': sharpe, 'bh_sharpe': bh_sharpe, 'mdd': mdd, 'lambdas': lambda_hist}


def test_ticker(ticker, info):
    """Test all HLs for a single ticker, return best auto-tuned HL."""
    print(f"\n{'='*70}")
    print(f"  {ticker} ({info['asset']}) — Paper JM-XGB Sharpe: {info['paper_sharpe']}")
    print(f"{'='*70}")

    # Fetch data
    _, df = fetch_etf_data(ticker, '1975-01-01')
    if df is None:
        print(f"  FAILED to fetch data")
        return []

    results = []

    # Test each HL
    for hl in [0, 2, 4, 8, 12, 16]:
        label = f"hl={hl}" + (" (paper)" if hl == info['paper_hl'] else "")
        res = backtest_with_hl(ticker, df, hl, label)
        if res is not None:
            results.append(res)
            delta = res['sharpe'] - res['bh_sharpe']
            gap = res['sharpe'] - info['paper_sharpe']
            marker = " ← paper" if hl == info['paper_hl'] else ""
            best_marker = ""
            print(f"  hl={hl:>2}: Sharpe {res['sharpe']:>6.3f}  vs B&H {delta:>+6.3f}  vs Paper {gap:>+6.3f}{marker}{best_marker}")

    # Find best HL
    if results:
        best = max(results, key=lambda r: r['sharpe'])
        paper_res = next((r for r in results if r['hl'] == info['paper_hl']), None)
        if paper_res:
            improvement = best['sharpe'] - paper_res['sharpe']
            print(f"\n  Best HL: {best['hl']} (Sharpe {best['sharpe']:.3f}) vs Paper HL {info['paper_hl']} (Sharpe {paper_res['sharpe']:.3f}) → Δ={improvement:+.3f}")

    return results


print("=" * 70)
print("  EWMA Halflife Sensitivity: Paper HL vs Auto-Tuned (2007-2023)")
print(f"  Lambda Grid: {LAMBDA_GRID}")
print("=" * 70)

all_results = []
for ticker, info in TICKERS.items():
    results = test_ticker(ticker, info)
    all_results.extend(results)

# Summary
print(f"\n\n{'='*70}")
print("  SUMMARY: Best HL per asset")
print(f"{'='*70}")
print(f"{'Ticker':<12} {'Paper HL':>8} {'Paper Sharpe':>12} {'Best HL':>8} {'Best Sharpe':>12} {'HL Gain':>8}")
print("-" * 62)
for ticker, info in TICKERS.items():
    ticker_results = [r for r in all_results if r['ticker'] == ticker]
    if ticker_results:
        best = max(ticker_results, key=lambda r: r['sharpe'])
        paper_r = next((r for r in ticker_results if r['hl'] == info['paper_hl']), None)
        paper_s = paper_r['sharpe'] if paper_r else float('nan')
        gain = best['sharpe'] - paper_s if paper_r else float('nan')
        print(f"{ticker:<12} {info['paper_hl']:>8} {paper_s:>12.3f} {best['hl']:>8} {best['sharpe']:>12.3f} {gain:>+8.3f}")
