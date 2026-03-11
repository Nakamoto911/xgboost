"""
Diagnose multi-asset benchmark performance gaps vs paper.
Tests: (1) time period effect, (2) per-asset lambda sensitivity, (3) walk-forward lambda choices.
"""
import sys, os, types

try:
    import distutils, distutils.version
except ImportError:
    d = types.ModuleType('distutils')
    dv = types.ModuleType('distutils.version')
    class LooseVersion:
        def __init__(self, vstring=None): self.vstring = vstring
        def __str__(self): return self.vstring
        def __lt__(self, other): return self.vstring < (other.vstring if hasattr(other, 'vstring') else other)
        def __le__(self, other): return self.vstring <= (other.vstring if hasattr(other, 'vstring') else other)
        def __gt__(self, other): return self.vstring > (other.vstring if hasattr(other, 'vstring') else other)
        def __ge__(self, other): return self.vstring >= (other.vstring if hasattr(other, 'vstring') else other)
        def __eq__(self, other): return self.vstring == (other.vstring if hasattr(other, 'vstring') else other)
    dv.LooseVersion = LooseVersion
    d.version = dv
    sys.modules['distutils'] = d
    sys.modules['distutils.version'] = dv

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'misc_scripts'))

from benchmark_assets import (
    fetch_etf_data, backtest_single_asset, StatisticalJumpModel,
    run_period_forecast_fast, simulate_strategy_fast, calculate_metrics,
    PAPER_EWMA_HL, LAMBDA_GRID, DD_EXCLUDE_TICKERS, EWMA_HL_GRID,
    VALIDATION_WINDOW_YRS
)
from config import StrategyConfig

# Assets to diagnose (all Long History assets)
ASSETS = {
    '^SP500TR': 'LargeCap',
    'VIMSX': 'MidCap',
    'NAESX': 'SmallCap',
    'FDIVX': 'EAFE',
    'VEIEX': 'EM',
    'VBMFX': 'AggBond',
    'VUSTX': 'Treasury',
    'VWEHX': 'HighYield',
    'VWESX': 'Corporate',
    'FRESX': 'REIT',
    'GC=F': 'Gold',
}

PAPER_SHARPE = {
    '^SP500TR': (0.79, 0.50), 'VIMSX': (0.59, 0.45), 'NAESX': (0.51, 0.36),
    'FDIVX': (0.56, 0.20), 'VEIEX': (0.85, 0.20), 'VBMFX': (0.67, 0.46),
    'VUSTX': (0.38, 0.26), 'VWEHX': (1.88, 0.67), 'VWESX': (0.76, 0.54),
    'FRESX': (0.56, 0.27), 'GC=F': (0.31, 0.43),
}

DATA_START = '1975-01-01'


def run_single_asset_backtest(ticker, df, oos_start, oos_end, lambda_grid=None, ewma_hl=None):
    """Run walk-forward backtest for a single asset with custom parameters."""
    config = StrategyConfig()
    cache = {}

    if lambda_grid is None:
        lambda_grid = LAMBDA_GRID

    oos_start_dt = pd.to_datetime(oos_start)
    oos_end_dt = pd.to_datetime(oos_end)

    # Phase 1: EWMA halflife
    if ewma_hl is not None:
        best_ewma_hl = ewma_hl
    elif ticker in PAPER_EWMA_HL:
        best_ewma_hl = PAPER_EWMA_HL[ticker]
    else:
        best_ewma_hl = 8

    # Phase 2: Walk-forward
    current_date = oos_start_dt
    jm_xgb_chunks = []
    lambda_history = []

    while current_date < oos_end_dt:
        chunk_end = min(current_date + pd.DateOffset(months=6), oos_end_dt)
        val_start = current_date - pd.DateOffset(years=VALIDATION_WINDOW_YRS)

        lambda_scores = []
        for lmbda in lambda_grid:
            val_res = simulate_strategy_fast(df, val_start, current_date, lmbda, cache, config,
                                            include_xgboost=True, ewma_halflife=best_ewma_hl)
            if not val_res.empty:
                _, _, sharpe, _, _ = calculate_metrics(val_res['Strat_Return'], val_res['RF_Rate'])
                if not np.isnan(sharpe):
                    lambda_scores.append((sharpe, lmbda))

        lambda_scores.sort(key=lambda x: x[0], reverse=True)
        if not lambda_scores:
            lambda_scores = [(0.0, lambda_grid[len(lambda_grid)//2])]

        best_lambda = lambda_scores[0][1]
        lambda_history.append(best_lambda)

        oos_chunk = run_period_forecast_fast(df, current_date, best_lambda, cache, config, include_xgboost=True)
        if oos_chunk is not None:
            jm_xgb_chunks.append(oos_chunk)

        current_date = chunk_end

    if not jm_xgb_chunks:
        return None, None, lambda_history

    jm_xgb_df = pd.concat(jm_xgb_chunks)

    if best_ewma_hl == 0:
        jm_xgb_df['State_Prob'] = jm_xgb_df['Raw_Prob']
    else:
        jm_xgb_df['State_Prob'] = jm_xgb_df['Raw_Prob'].ewm(halflife=best_ewma_hl).mean()

    jm_xgb_df['Forecast_State'] = (jm_xgb_df['State_Prob'] > 0.5).astype(int)
    trading_signals = jm_xgb_df['Forecast_State'].shift(1).fillna(0)
    alloc_target = 1.0 - trading_signals
    strat_rets = (alloc_target * jm_xgb_df['Target_Return']) + ((1.0 - alloc_target) * jm_xgb_df['RF_Rate'])
    trades = alloc_target.diff().abs().fillna(0)
    jm_xgb_df['Strat_Return'] = strat_rets - (trades * 0.0005)

    mask = (jm_xgb_df.index >= oos_start_dt) & (jm_xgb_df.index < oos_end_dt)
    jm_xgb_df = jm_xgb_df[mask]

    strat_metrics = calculate_metrics(jm_xgb_df['Strat_Return'], jm_xgb_df['RF_Rate'])
    bh_metrics = calculate_metrics(jm_xgb_df['Target_Return'], jm_xgb_df['RF_Rate'])

    return strat_metrics, bh_metrics, lambda_history


def run_fixed_lambda(ticker, df, oos_start, oos_end, fixed_lambda, ewma_hl=None):
    """Run backtest with a single fixed lambda (no walk-forward tuning)."""
    config = StrategyConfig()
    cache = {}

    oos_start_dt = pd.to_datetime(oos_start)
    oos_end_dt = pd.to_datetime(oos_end)

    if ewma_hl is None:
        ewma_hl = PAPER_EWMA_HL.get(ticker, 8)

    current_date = oos_start_dt
    chunks = []

    while current_date < oos_end_dt:
        chunk = run_period_forecast_fast(df, current_date, fixed_lambda, cache, config, include_xgboost=True)
        if chunk is not None:
            chunks.append(chunk)
        current_date += pd.DateOffset(months=6)

    if not chunks:
        return None, None

    result = pd.concat(chunks)

    if ewma_hl == 0:
        result['State_Prob'] = result['Raw_Prob']
    else:
        result['State_Prob'] = result['Raw_Prob'].ewm(halflife=ewma_hl).mean()

    result['Forecast_State'] = (result['State_Prob'] > 0.5).astype(int)
    signals = result['Forecast_State'].shift(1).fillna(0)
    alloc = 1.0 - signals
    strat_rets = (alloc * result['Target_Return']) + ((1.0 - alloc) * result['RF_Rate'])
    trades = alloc.diff().abs().fillna(0)
    result['Strat_Return'] = strat_rets - (trades * 0.0005)

    mask = (result.index >= oos_start_dt) & (result.index < oos_end_dt)
    result = result[mask]

    strat_metrics = calculate_metrics(result['Strat_Return'], result['RF_Rate'])
    bh_metrics = calculate_metrics(result['Target_Return'], result['RF_Rate'])

    return strat_metrics, bh_metrics


def main():
    print("=" * 90)
    print("  MULTI-ASSET DIAGNOSTIC: Why do some assets LOSE to B&H?")
    print("=" * 90)

    # Fetch all data
    print("\nFetching data...")
    asset_data = {}
    for ticker in ASSETS:
        _, df = fetch_etf_data(ticker, data_start=DATA_START)
        if df is not None:
            asset_data[ticker] = df
            print(f"  {ticker}: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 1: Time period effect (2007-2023 vs 2007-2025)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  TEST 1: Time Period Effect (2007-2023 vs 2007-2025)")
    print("=" * 90)

    print(f"\n{'Ticker':<10} {'Name':<12} {'2007-23 JM':>10} {'2007-23 BH':>10} {'Δ23':>8} {'V23':>6} "
          f"{'2007-25 JM':>10} {'2007-25 BH':>10} {'Δ25':>8} {'V25':>6} {'Paper Δ':>8}")
    print("-" * 110)

    period_results = {}
    for ticker, name in ASSETS.items():
        if ticker not in asset_data:
            continue
        df = asset_data[ticker]

        # 2007-2023 (paper period)
        s23, b23, lh23 = run_single_asset_backtest(ticker, df, '2007-01-01', '2024-01-01')
        # 2007-2025 (our period)
        s25, b25, lh25 = run_single_asset_backtest(ticker, df, '2007-01-01', '2026-01-01')

        if s23 and s25:
            delta23 = s23[2] - b23[2]
            delta25 = s25[2] - b25[2]
            v23 = 'WIN' if delta23 > 0 else 'LOSE'
            v25 = 'WIN' if delta25 > 0 else 'LOSE'
            paper_jm, paper_bh = PAPER_SHARPE.get(ticker, (0, 0))
            paper_delta = paper_jm - paper_bh

            period_results[ticker] = {
                's23': s23[2], 'b23': b23[2], 'delta23': delta23,
                's25': s25[2], 'b25': b25[2], 'delta25': delta25,
                'lambda23': lh23, 'lambda25': lh25,
            }

            print(f"{ticker:<10} {name:<12} {s23[2]:>10.3f} {b23[2]:>10.3f} {delta23:>+8.3f} {v23:>6} "
                  f"{s25[2]:>10.3f} {b25[2]:>10.3f} {delta25:>+8.3f} {v25:>6} {paper_delta:>+8.3f}")

    # Count wins
    wins_23 = sum(1 for r in period_results.values() if r['delta23'] > 0)
    wins_25 = sum(1 for r in period_results.values() if r['delta25'] > 0)
    total = len(period_results)
    print(f"\nWins: 2007-2023={wins_23}/{total}, 2007-2025={wins_25}/{total}, Paper=11/12")

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 2: Per-asset fixed lambda sweep (2007-2023)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  TEST 2: Fixed Lambda Sweep (2007-2023, no walk-forward)")
    print("=" * 90)

    lambda_sweep = [1.0, 2.15, 4.64, 10.0, 15.0, 21.54, 30.0, 46.42, 70.0, 100.0]

    for ticker, name in ASSETS.items():
        if ticker not in asset_data:
            continue
        df = asset_data[ticker]
        hl = PAPER_EWMA_HL.get(ticker, 8)

        print(f"\n  {ticker} ({name}, hl={hl}):")
        print(f"  {'λ':>8} {'JM-XGB':>10} {'B&H':>10} {'Delta':>8} {'Verdict':>8}")

        best_delta = -np.inf
        best_lambda = None
        for lmbda in lambda_sweep:
            sm, bm = run_fixed_lambda(ticker, df, '2007-01-01', '2024-01-01', lmbda, ewma_hl=hl)
            if sm and bm:
                delta = sm[2] - bm[2]
                v = 'WIN' if delta > 0 else 'LOSE'
                marker = ' <--' if delta > best_delta else ''
                print(f"  {lmbda:>8.2f} {sm[2]:>10.3f} {bm[2]:>10.3f} {delta:>+8.3f} {v:>8}{marker}")
                if delta > best_delta:
                    best_delta = delta
                    best_lambda = lmbda

        if best_lambda:
            paper_jm, _ = PAPER_SHARPE.get(ticker, (0, 0))
            wf_s = period_results.get(ticker, {}).get('s23', 0)
            print(f"  → Best fixed λ={best_lambda:.2f} (Δ={best_delta:+.3f}) vs WF Sharpe={wf_s:.3f}")

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 3: Walk-forward lambda history per asset (2007-2023)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  TEST 3: Walk-Forward Lambda History (2007-2023)")
    print("=" * 90)

    for ticker, name in ASSETS.items():
        if ticker not in period_results:
            continue
        lh = period_results[ticker]['lambda23']
        if lh:
            mean_l = np.mean(lh)
            std_l = np.std(lh)
            cv = std_l / mean_l if mean_l > 0 else 0
            unique_lambdas = sorted(set(lh))
            print(f"  {ticker:<10} ({name:<12}): mean={mean_l:.1f}, std={std_l:.1f}, CV={cv:.2f}, "
                  f"picks={[f'{l:.1f}' for l in lh]}")

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 4: EWMA halflife sensitivity for losing assets (2007-2023)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  TEST 4: EWMA Halflife Sensitivity for Losing Assets (2007-2023)")
    print("=" * 90)

    losing_assets = [t for t, r in period_results.items() if r['delta23'] <= 0.05]

    for ticker in losing_assets:
        if ticker not in asset_data:
            continue
        name = ASSETS[ticker]
        df = asset_data[ticker]
        paper_hl = PAPER_EWMA_HL.get(ticker, -1)

        print(f"\n  {ticker} ({name}), paper hl={paper_hl}:")
        print(f"  {'HL':>4} {'JM-XGB':>10} {'B&H':>10} {'Delta':>8}")

        for hl in [0, 2, 4, 8, 12, 16]:
            sm, bm, _ = run_single_asset_backtest(ticker, df, '2007-01-01', '2024-01-01', ewma_hl=hl)
            if sm and bm:
                delta = sm[2] - bm[2]
                marker = ' <-- paper' if hl == paper_hl else ''
                print(f"  {hl:>4} {sm[2]:>10.3f} {bm[2]:>10.3f} {delta:>+8.3f}{marker}")

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 5: Lambda grid variants for losing assets (2007-2023)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  TEST 5: Lambda Grid Variants for Losing Assets (2007-2023)")
    print("=" * 90)

    grid_variants = {
        'Current [4.6,10,21.5,46.4,100]': [4.64, 10.0, 21.54, 46.42, 100.0],
        'No-100 [4.6,10,21.5,46.4]': [4.64, 10.0, 21.54, 46.42],
        'Low [1,2.15,4.64,10,21.54]': [1.0, 2.15, 4.64, 10.0, 21.54],
        'Mid [10,15,21.5,30,46.4]': [10.0, 15.0, 21.54, 30.0, 46.42],
        'Wide [2.15,4.64,10,21.54,46.42,100]': [2.15, 4.64, 10.0, 21.54, 46.42, 100.0],
        'High [21.5,46.4,70,100]': [21.54, 46.42, 70.0, 100.0],
        'Tight [10,15,21.5,30]': [10.0, 15.0, 21.54, 30.0],
    }

    for ticker in losing_assets:
        if ticker not in asset_data:
            continue
        name = ASSETS[ticker]
        df = asset_data[ticker]

        print(f"\n  {ticker} ({name}):")
        print(f"  {'Grid':<40} {'JM-XGB':>10} {'B&H':>10} {'Delta':>8} {'Lambdas picked':>20}")

        for grid_name, grid in grid_variants.items():
            sm, bm, lh = run_single_asset_backtest(ticker, df, '2007-01-01', '2024-01-01', lambda_grid=grid)
            if sm and bm:
                delta = sm[2] - bm[2]
                v = 'WIN' if delta > 0 else 'LOSE'
                # Summarize lambda picks
                from collections import Counter
                counts = Counter([f'{l:.1f}' for l in lh])
                top_picks = ', '.join(f'{l}×{c}' for l, c in counts.most_common(3))
                print(f"  {grid_name:<40} {sm[2]:>10.3f} {bm[2]:>10.3f} {delta:>+8.3f} {top_picks:>20}")


if __name__ == '__main__':
    main()
