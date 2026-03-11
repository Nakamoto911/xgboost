"""
Multi-Asset Benchmark: 2007-2023 period matching Paper Table 4.
Monkey-patches TIME_PERIODS in benchmark_assets to run only the paper period,
then prints a comparison table vs paper JM-XGB Sharpe values.
"""

import sys
import os
import time

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'misc_scripts'))

# Monkey-patch TIME_PERIODS before importing benchmark_assets
import benchmark_assets
benchmark_assets.TIME_PERIODS = [
    ('Full (2007-2023)', '2007-01-01', '2024-01-01'),
]

from benchmark_assets import (
    load_asset_lists, fetch_etf_data, backtest_single_asset
)
from config import StrategyConfig

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

# Paper Table 4 JM-XGB Sharpe values (2007-2023)
PAPER_SHARPE = {
    '^SP500TR': 0.79,
    'VIMSX': 0.59,
    'NAESX': 0.51,
    'FDIVX': 0.56,
    'VEIEX': 0.85,
    'FRESX': 0.56,
    'VBMFX': 0.67,
    'VUSTX': 0.38,
    'VWEHX': 1.88,
    'VWESX': 0.76,
    'PCASX': 0.23,
    'GC=F': 0.31,
}

def main():
    all_lists = load_asset_lists()
    selected = all_lists['Long History']
    tickers = selected['tickers']
    asset_classes = selected['asset_classes']
    data_start = selected['data_start']

    t0 = time.time()
    print("=" * 80)
    print("  MULTI-ASSET JM-XGB BENCHMARK — Paper Period (2007-2023)")
    print("=" * 80)
    print(f"\nFetching data for {len(tickers)} assets (data_start={data_start})...")

    asset_data = {}
    for ticker in tickers:
        print(f"  Fetching {ticker}...", end=" ", flush=True)
        _, df = fetch_etf_data(ticker, data_start=data_start)
        if df is not None:
            asset_data[ticker] = df
            print(f"OK ({len(df)} rows, {df.index[0].date()} to {df.index[-1].date()})")
        else:
            print("FAILED")

    print(f"\nSuccessfully loaded {len(asset_data)}/{len(tickers)} assets")
    if not asset_data:
        print("No data available. Exiting.")
        return

    print(f"\nRunning walk-forward backtests (2007-2023)...")
    config = StrategyConfig()
    args_list = [(ticker, df, config, data_start) for ticker, df in asset_data.items()]

    all_results = []
    n_workers = min(cpu_count(), len(args_list))
    print(f"Using {n_workers} parallel workers\n")

    if n_workers > 1:
        with Pool(n_workers) as pool:
            for i, result_list in enumerate(pool.imap_unordered(backtest_single_asset, args_list)):
                ticker = result_list[0]['Ticker'] if result_list else '?'
                print(f"  Completed {ticker} ({i+1}/{len(args_list)})")
                all_results.extend(result_list)
    else:
        for args in args_list:
            result_list = backtest_single_asset(args)
            ticker = result_list[0]['Ticker'] if result_list else '?'
            print(f"  Completed {ticker}")
            all_results.extend(result_list)

    elapsed = time.time() - t0
    results_df = pd.DataFrame(all_results)

    # Extract JM-XGB and B&H results for the full period
    full_period = 'Full (2007-2023)'
    jm = results_df[(results_df['Period'] == full_period) & (results_df['Strategy'] == 'JM-XGB')]
    bh = results_df[(results_df['Period'] == full_period) & (results_df['Strategy'] == 'B&H')]

    # Print comparison table
    print(f"\n{'=' * 100}")
    print(f"  RESULTS: Our JM-XGB vs Paper Table 4 (2007-2023)")
    print(f"{'=' * 100}")
    print(f"\n{'Ticker':<12} {'Our JM-XGB':>12} {'Our B&H':>10} {'Paper JM-XGB':>14} {'Gap (Ours-Paper)':>18} {'Our vs B&H':>12}")
    print(f"{'-'*12} {'-'*12} {'-'*10} {'-'*14} {'-'*18} {'-'*12}")

    our_sharpes = []
    paper_sharpes = []
    gaps = []
    wins_vs_paper = 0
    wins_vs_bh = 0
    total = 0

    for ticker in tickers:
        j = jm[jm['Ticker'] == ticker]
        b = bh[bh['Ticker'] == ticker]
        paper_val = PAPER_SHARPE.get(ticker, np.nan)

        if j.empty or np.isnan(j.iloc[0]['Sharpe']):
            print(f"{ticker:<12} {'N/A':>12} {'N/A':>10} {paper_val:>14.2f} {'N/A':>18} {'N/A':>12}")
            continue

        our_val = j.iloc[0]['Sharpe']
        bh_val = b.iloc[0]['Sharpe'] if not b.empty else np.nan
        gap = our_val - paper_val
        delta_bh = our_val - bh_val if not np.isnan(bh_val) else np.nan

        total += 1
        our_sharpes.append(our_val)
        paper_sharpes.append(paper_val)
        gaps.append(gap)
        if our_val > paper_val:
            wins_vs_paper += 1
        if not np.isnan(bh_val) and our_val > bh_val:
            wins_vs_bh += 1

        print(f"{ticker:<12} {our_val:>12.2f} {bh_val:>10.2f} {paper_val:>14.2f} {gap:>+18.2f} {delta_bh:>+12.2f}")

    print(f"{'-'*12} {'-'*12} {'-'*10} {'-'*14} {'-'*18} {'-'*12}")
    if our_sharpes:
        avg_ours = np.mean(our_sharpes)
        avg_paper = np.mean(paper_sharpes)
        avg_gap = np.mean(gaps)
        print(f"{'AVG':<12} {avg_ours:>12.2f} {'':>10} {avg_paper:>14.2f} {avg_gap:>+18.2f}")
        print(f"\nWins vs Paper: {wins_vs_paper}/{total} | Wins vs B&H: {wins_vs_bh}/{total}")
        print(f"Mean absolute gap: {np.mean(np.abs(gaps)):.2f}")

    print(f"\nCompleted in {elapsed:.1f}s")

    # Save CSV
    csv_path = os.path.join(PROJECT_ROOT, 'benchmarks', f'benchmark_2007_2023_{time.strftime("%Y%m%d_%H%M%S")}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")


if __name__ == '__main__':
    main()
