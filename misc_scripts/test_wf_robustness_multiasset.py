"""
Multi-asset WF robustness comparison: tests different lambda selection strategies
across all 11 Long History assets. Compares WF Sharpe vs Oracle (best fixed lambda).

Tests: Baseline, Expanding Window, Median-Positive, Sub-Window Consensus, Expand+Smooth
Uses multiprocessing for speed.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'misc_scripts'))

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from config import StrategyConfig

# Must import after path setup
import benchmark_assets
from benchmark_assets import (
    fetch_etf_data, backtest_single_asset, calculate_metrics,
    LAMBDA_GRID, PAPER_EWMA_HL,
)

# Only test 2007-2023 period for paper comparison
TEST_PERIOD = ("Full OOS", "2007-01-01", "2024-01-01")

TICKERS = [
    '^SP500TR', 'VIMSX', 'NAESX', 'FDIVX', 'VEIEX',
    'FRESX', 'VBMFX', 'VUSTX', 'VWEHX', 'VWESX', 'GC=F'
]
ASSET_NAMES = {
    '^SP500TR': 'LargeCap', 'VIMSX': 'MidCap', 'NAESX': 'SmallCap',
    'FDIVX': 'EAFE', 'VEIEX': 'EM', 'FRESX': 'REIT',
    'VBMFX': 'AggBond', 'VUSTX': 'Treasury', 'VWEHX': 'HighYield',
    'VWESX': 'Corporate', 'GC=F': 'Gold',
}

PAPER_BH_SHARPE = {
    '^SP500TR': 0.50, 'VIMSX': 0.45, 'NAESX': 0.36, 'FDIVX': 0.20,
    'VEIEX': 0.20, 'FRESX': 0.27, 'VBMFX': 0.46, 'VUSTX': 0.26,
    'VWEHX': 0.67, 'VWESX': 0.54, 'GC=F': 0.43,
}

PAPER_JMXGB_SHARPE = {
    '^SP500TR': 0.79, 'VIMSX': 0.63, 'NAESX': 0.51, 'FDIVX': 0.73,
    'VEIEX': 0.85, 'FRESX': 0.43, 'VBMFX': 1.14, 'VUSTX': 0.48,
    'VWEHX': 1.88, 'VWESX': 1.53, 'GC=F': 0.08,
}

# Strategies to test
STRATEGIES = {
    'Baseline': StrategyConfig(name="Baseline"),
    'Expanding': StrategyConfig(name="Expanding", validation_window_type="expanding"),
    'MedianPos': StrategyConfig(name="MedianPos", lambda_selection="median_positive"),
    'SubWindow': StrategyConfig(name="SubWindow", lambda_subwindow_consensus=True),
    'Exp+Smooth': StrategyConfig(name="Exp+Smooth", validation_window_type="expanding", lambda_smoothing=True),
}

DATA_START = '1975-01-01'


def run_one(args):
    """Worker: run one (ticker, strategy) pair."""
    ticker, strat_name, config = args
    # Override TIME_PERIODS in this worker
    benchmark_assets.TIME_PERIODS = [TEST_PERIOD]
    _, df = fetch_etf_data(ticker, DATA_START)
    if df is None:
        return (ticker, strat_name, None, None)
    result = backtest_single_asset((ticker, df, config, DATA_START))
    jmxgb = [r for r in result if r['Strategy'] == 'JM-XGB' and r['Period'] == TEST_PERIOD[0]]
    bh = [r for r in result if r['Strategy'] == 'B&H' and r['Period'] == TEST_PERIOD[0]]
    sharpe = jmxgb[0]['Sharpe'] if jmxgb and not np.isnan(jmxgb[0]['Sharpe']) else None
    bh_sharpe = bh[0]['Sharpe'] if bh and not np.isnan(bh[0]['Sharpe']) else None
    return (ticker, strat_name, sharpe, bh_sharpe)


def main():
    print("=" * 100)
    print("  MULTI-ASSET WF ROBUSTNESS: Comparing Lambda Selection Strategies (Parallel)")
    print(f"  Lambda Grid: {LAMBDA_GRID}")
    print(f"  Period: {TEST_PERIOD[1]} to {TEST_PERIOD[2]}")
    print(f"  Workers: {cpu_count()}")
    print("=" * 100)

    # Build all (ticker, strategy) jobs
    jobs = []
    for ticker in TICKERS:
        for strat_name, config in STRATEGIES.items():
            jobs.append((ticker, strat_name, config))

    print(f"\n  Running {len(jobs)} jobs ({len(TICKERS)} tickers × {len(STRATEGIES)} strategies)...\n")

    # Run in parallel
    n_workers = min(cpu_count(), 8)
    with Pool(n_workers) as pool:
        raw_results = pool.map(run_one, jobs)

    # Organize results
    all_results = {}  # {ticker: {strat_name: {sharpe, bh_sharpe}}}
    for ticker, strat_name, sharpe, bh_sharpe in raw_results:
        if ticker not in all_results:
            all_results[ticker] = {}
        if sharpe is not None:
            all_results[ticker][strat_name] = {'sharpe': sharpe, 'bh_sharpe': bh_sharpe}

    # Summary table
    print(f"\n{'='*110}")
    print("  SUMMARY: WF Sharpe by Strategy (2007-2023)")
    print(f"{'='*110}")

    header = f"{'Ticker':<10} {'Asset':<10}"
    for sn in STRATEGIES:
        header += f" {sn:>12}"
    header += f" {'Paper':>8} {'B&H':>6}"
    print(header)
    print("-" * 110)

    wins = {sn: 0 for sn in STRATEGIES}
    totals = {sn: 0 for sn in STRATEGIES}
    sharpe_sums = {sn: 0.0 for sn in STRATEGIES}
    delta_sums = {sn: 0.0 for sn in STRATEGIES}

    for ticker in TICKERS:
        if ticker not in all_results:
            continue
        asset = ASSET_NAMES[ticker]
        paper = PAPER_JMXGB_SHARPE.get(ticker, np.nan)
        bh = PAPER_BH_SHARPE.get(ticker, np.nan)

        line = f"{ticker:<10} {asset:<10}"
        for sn in STRATEGIES:
            if sn in all_results[ticker]:
                s = all_results[ticker][sn]['sharpe']
                bh_our = all_results[ticker][sn].get('bh_sharpe', bh)
                line += f" {s:>12.3f}"
                if bh_our is not None and not np.isnan(bh_our) and s > bh_our:
                    wins[sn] += 1
                totals[sn] += 1
                sharpe_sums[sn] += s
                if bh_our is not None and not np.isnan(bh_our):
                    delta_sums[sn] += (s - bh_our)
            else:
                line += f" {'N/A':>12}"
        line += f" {paper:>8.2f} {bh:>6.2f}"
        print(line)

    print("-" * 110)
    line = f"{'Win rate':<10} {'':10}"
    for sn in STRATEGIES:
        if totals[sn] > 0:
            wr = wins[sn] / totals[sn]
            line += f" {wr:>11.0%} "
        else:
            line += f" {'N/A':>12}"
    print(line)

    line = f"{'Avg Sharpe':<10} {'':10}"
    for sn in STRATEGIES:
        if totals[sn] > 0:
            avg = sharpe_sums[sn] / totals[sn]
            line += f" {avg:>12.3f}"
        else:
            line += f" {'N/A':>12}"
    print(line)

    line = f"{'Avg Delta':<10} {'vs B&H':10}"
    for sn in STRATEGIES:
        if totals[sn] > 0:
            avg = delta_sums[sn] / totals[sn]
            line += f" {avg:>+12.3f}"
        else:
            line += f" {'N/A':>12}"
    print(line)

    print(f"\n  Best by avg Sharpe:    {max(sharpe_sums, key=lambda k: sharpe_sums[k]/max(totals[k],1))}")
    print(f"  Best by win rate:      {max(wins, key=lambda k: wins[k]/max(totals[k],1))}")
    print(f"  Best by avg delta B&H: {max(delta_sums, key=lambda k: delta_sums[k]/max(totals[k],1))}")


if __name__ == '__main__':
    main()
