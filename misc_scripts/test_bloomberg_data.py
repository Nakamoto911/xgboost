#!/usr/bin/env python3
"""
Test Bloomberg SPTR data against paper results.

Paper target for LargeCap (SPTR) 2007-2023:
  - JM-XGB Sharpe: 0.79
  - JM-XGB MDD: -17.69%
  - JM-only Sharpe: 0.59
  - JM-only MDD: -24.78%
  - B&H Sharpe: 0.50
  - B&H MDD: -55.25%
  - % Bear (Fig 2): 20.9%
  - Regime shifts (Fig 2): 46

This script replaces Yahoo ^SP500TR with Bloomberg SPTR Index data
and Bloomberg LBUSTRUU for Stock-Bond Correlation (instead of VBMFX).
VIX and FRED yields still come from Yahoo/FRED.

Usage:
  python misc_scripts/test_bloomberg_data.py           # Grid sweep (default)
  python misc_scripts/test_bloomberg_data.py --full    # Grid sweep + JM-only + bear periods
  python misc_scripts/test_bloomberg_data.py --regime  # Regime stats only (current 8pt grid)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

# --- Override main.py module-level constants BEFORE importing ---
os.environ['XGB_END_DATE'] = '2023-12-31'
os.environ['XGB_OOS_START_DATE'] = '2007-01-01'
os.environ['XGB_START_DATE_DATA'] = '1987-01-01'

import main
from config import StrategyConfig

# Force paper date range
main.END_DATE = '2023-12-31'
main.OOS_START_DATE = '2007-01-01'

# Paper targets (Table 4 + Figure 2)
PAPER_SHARPE = 0.790
PAPER_MDD    = -0.1769
PAPER_BH_SHARPE = 0.500
PAPER_BH_MDD = -0.5525
PAPER_BEAR_PCT = 20.9   # % from Figure 2
PAPER_SHIFTS   = 46     # regime shifts from Figure 2

# Paper-matching StrategyConfig (ewma_mode="paper" → hl=8 for LargeCap)
PAPER_CONFIG = StrategyConfig(
    name="Paper Baseline (Bloomberg)",
    ewma_mode="paper",
    tuning_metric="sharpe",
    lambda_selection="best",
    lambda_subwindow_consensus=False,
    allocation_style="binary",
    prob_threshold=0.50,
)


def load_bloomberg_sptr():
    """Load Bloomberg SPTR and LBUSTRUU (AggBond) from Excel."""
    xlsx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'cache', 'DATA PAUL.xlsx')
    df = pd.read_excel(xlsx_path, header=None, skiprows=6)
    df.columns = ['Date', 'SPTR', 'SPTRMDCP', 'RU20INTR', 'NDDUEAFE', 'NDUEEGF',
                  'LBUSTRUU', 'IBOXHY', 'LUACTRUU', 'DJUSRET', 'DBLCDBCE',
                  'GOLDLNPM', 'LUTLTRUU']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    df = df[['SPTR', 'LBUSTRUU']].dropna()
    return df


def prepare_bloomberg_features():
    """Build feature DataFrame using Bloomberg SPTR data + Yahoo VIX + FRED yields."""
    print("Loading Bloomberg data...")
    bbg = load_bloomberg_sptr()
    print(f"  SPTR: {bbg.index.min().date()} to {bbg.index.max().date()}, {len(bbg)} rows")

    fred_data = main._fetch_fred_data()
    fred_data = fred_data.ffill().dropna()

    import yfinance as yf
    print("Fetching VIX and IRX from Yahoo...")
    vix = yf.download('^VIX', start='1987-01-01', end='2024-01-01', auto_adjust=False)
    irx = yf.download('^IRX', start='1987-01-01', end='2024-01-01', auto_adjust=False)

    if isinstance(vix.columns, pd.MultiIndex):
        vix_series = vix['Adj Close'].iloc[:, 0]
    else:
        vix_series = vix['Adj Close'] if 'Adj Close' in vix.columns else vix['Close']

    if isinstance(irx.columns, pd.MultiIndex):
        irx_series = irx['Adj Close'].iloc[:, 0]
    else:
        irx_series = irx['Adj Close'] if 'Adj Close' in irx.columns else irx['Close']

    vix_series = vix_series.rename('VIX')
    irx_series = irx_series.rename('IRX')

    df = bbg.join(fred_data, how='inner')
    df = df.join(vix_series, how='inner')
    df = df.join(irx_series, how='inner')
    df = df.ffill().dropna()

    print(f"  Combined data: {df.index.min().date()} to {df.index.max().date()}, {len(df)} rows")

    print("Calculating features...")
    features = pd.DataFrame(index=df.index)

    target_returns = df['SPTR'].pct_change().fillna(0)
    features['Target_Return'] = target_returns
    features['Target_Intraday_Ret'] = target_returns
    features['Target_Overnight_Ret'] = 0.0
    features['RF_Rate'] = (df['IRX'] / 100) / 252
    features['Excess_Return'] = target_returns - features['RF_Rate']

    downside_returns = np.minimum(features['Excess_Return'], 0)

    for hl in [5, 21]:
        ewm_var = (downside_returns ** 2).ewm(halflife=hl).mean()
        ewm_dd = np.sqrt(ewm_var).fillna(0)
        features[f'DD_log_{hl}'] = np.log(ewm_dd + 1e-8)

    for hl in [5, 10, 21]:
        features[f'Avg_Ret_{hl}'] = features['Excess_Return'].ewm(halflife=hl).mean()

    for hl in [5, 10, 21]:
        ewm_var = (downside_returns ** 2).ewm(halflife=hl).mean()
        ewm_dd_raw = np.sqrt(ewm_var).fillna(1e-8)
        ewm_dd_raw = np.maximum(ewm_dd_raw, 1e-8)
        features[f'Sortino_{hl}'] = (features[f'Avg_Ret_{hl}'] / ewm_dd_raw).clip(-10, 10)

    features['Yield_2Y_diff'] = df['DGS2'].diff().fillna(0)
    features['Yield_2Y_EWMA_diff'] = features['Yield_2Y_diff'].ewm(halflife=21).mean()

    slope = df['DGS10'] - df['DGS2']
    features['Yield_Slope_EWMA_10'] = slope.ewm(halflife=10).mean()
    slope_diff = slope.diff().fillna(0)
    features['Yield_Slope_EWMA_diff_21'] = slope_diff.ewm(halflife=21).mean()

    vix_log_diff = np.log(df['VIX'] / df['VIX'].shift(1)).fillna(0)
    features['VIX_EWMA_log_diff'] = vix_log_diff.ewm(halflife=63).mean()

    largecap_returns = df['SPTR'].pct_change().fillna(0)
    bond_returns = df['LBUSTRUU'].pct_change().fillna(0)
    features['Stock_Bond_Corr'] = largecap_returns.rolling(window=252).corr(bond_returns).fillna(0)

    final_df = features.dropna()
    print(f"  Features: {final_df.index.min().date()} to {final_df.index.max().date()}, {len(final_df)} rows")
    return final_df


def compute_regime_stats(result):
    """
    Compute bear regime statistics from walk_forward_backtest result.
    Returns dict with: bear_pct, n_shifts, bear_periods list.
    Prints comparison to paper Figure 2 targets.
    """
    oos = result[(result.index >= '2007-01-01') & (result.index <= '2023-12-31')]

    if 'Forecast_State' not in oos.columns:
        print("  [no Forecast_State column — regime stats unavailable]")
        return {}

    states = oos['Forecast_State']

    # % bear days
    bear_pct = (states == 1).mean() * 100

    # Number of regime shifts (transitions between state 0 and state 1)
    shifted = states.shift(1)
    n_shifts = int((states != shifted).sum()) - 1  # subtract 1 for the NaN first row
    n_shifts = max(n_shifts, 0)

    # Bear periods (consecutive runs of state == 1)
    bear_periods = []
    in_bear = False
    start = None
    for date, s in states.items():
        if s == 1 and not in_bear:
            in_bear = True
            start = date
        elif s == 0 and in_bear:
            in_bear = False
            bear_periods.append((start, date))
    if in_bear:
        bear_periods.append((start, states.index[-1]))

    return {
        'bear_pct': bear_pct,
        'n_shifts': n_shifts,
        'bear_periods': bear_periods,
    }


def print_regime_stats(stats, show_periods=False):
    """Print regime stats vs paper Figure 2 targets."""
    if not stats:
        return
    bear_pct = stats['bear_pct']
    n_shifts = stats['n_shifts']
    bear_periods = stats['bear_periods']

    print(f"  Regime stats:")
    print(f"    % Bear days: {bear_pct:.1f}%  (paper Fig 2: {PAPER_BEAR_PCT}%,  delta: {bear_pct - PAPER_BEAR_PCT:+.1f}pp)")
    print(f"    # Shifts:    {n_shifts}       (paper Fig 2: {PAPER_SHIFTS},     delta: {n_shifts - PAPER_SHIFTS:+d})")

    if show_periods:
        print(f"    Bear periods ({len(bear_periods)} total):")
        for s, e in bear_periods:
            dur = (e - s).days
            print(f"      {s.strftime('%Y-%m-%d')} → {e.strftime('%Y-%m-%d')}  ({dur}d)")


def run_single_grid(df, grid, label, config=None, show_periods=False):
    """Run walk-forward backtest with a specific lambda grid, return (sharpe, mdd, stats).

    NOTE: does NOT clear the forecast cache — forecasts at specific (date, lambda) pairs
    are shared across grid runs, so keeping the cache makes multi-grid sweeps much faster.
    """
    if config is None:
        config = PAPER_CONFIG
    main.LAMBDA_GRID = grid

    result = main.walk_forward_backtest(df, config)
    if result.empty:
        print(f"  {label}: ERROR — empty result")
        return None, None, None

    oos = result[(result.index >= '2007-01-01') & (result.index <= '2023-12-31')]
    _, _, sharpe, _, mdd = main.calculate_metrics(oos['Strat_Return'], oos['RF_Rate'])

    lambda_history = result.attrs.get('lambda_history', [])
    ewma_hl = result.attrs.get('ewma_halflife', '?')
    stats = compute_regime_stats(result)

    sharpe_gap = sharpe - PAPER_SHARPE
    mdd_gap = mdd - PAPER_MDD

    print(f"  {label:<35} Sharpe={sharpe:.3f} ({sharpe_gap:+.3f})  MDD={mdd:.1%} ({mdd_gap:+.1%})  hl={ewma_hl}")
    if lambda_history:
        lh = np.array(lambda_history)
        print(f"    λ history: mean={lh.mean():.1f} std={lh.std():.1f}  picks={[f'{l:.2f}' for l in lh]}")
    print_regime_stats(stats, show_periods=show_periods)

    return sharpe, mdd, stats


def run_jm_only(df):
    """Run JM-only across all grid lambdas and report best."""
    print(f"\n{'='*70}")
    print("JM-ONLY BASELINE (Bloomberg SPTR, 2007-2023)")
    print(f"{'='*70}")
    print(f"Grid: {main.LAMBDA_GRID}")

    best_sharpe = -np.inf
    best_lambda = None
    best_res = None

    for lmbda in main.LAMBDA_GRID:
        res = main.simulate_strategy(df, '2007-01-01', '2023-12-31', lmbda,
                                     PAPER_CONFIG, include_xgboost=False)
        if not res.empty:
            _, _, s, _, m = main.calculate_metrics(res['Strat_Return'], res['RF_Rate'])
            states = res.get('Forecast_State', pd.Series(dtype=int))
            bear_pct = (states == 1).mean() * 100 if not states.empty else float('nan')
            shifted = states.shift(1)
            n_shifts = int((states != shifted).sum()) - 1
            print(f"  λ={lmbda:6.2f}  Sharpe={s:.3f}  MDD={m:.1%}  Bear={bear_pct:.1f}%  Shifts={n_shifts}")
            if s > best_sharpe:
                best_sharpe = s
                best_lambda = lmbda
                best_res = res

    if best_res is not None:
        _, _, _, _, best_mdd = main.calculate_metrics(best_res['Strat_Return'], best_res['RF_Rate'])
        print(f"\n  Best JM-only: λ={best_lambda}  Sharpe={best_sharpe:.3f} (paper: 0.59)  MDD={best_mdd:.1%} (paper: -24.78%)")


def run_grid_sweep(df):
    """
    Systematic lambda grid sweep including λ=0 variants.
    Tests the paper's claim of 'log-uniform grid from 0.0 to 100.0'.
    """
    print(f"\n{'='*70}")
    print("LAMBDA GRID SWEEP — Bloomberg SPTR 2007-2023")
    print(f"Paper targets: Sharpe={PAPER_SHARPE:.3f}  MDD={PAPER_MDD:.2%}  Bear={PAPER_BEAR_PCT}%  Shifts={PAPER_SHIFTS}")
    print(f"{'='*70}\n")

    # ── Group 1: Reference grids (no λ=0) ────────────────────────────────────
    print("── Group 1: Reference grids (λ>0 only) ──")

    grids_ref = [
        ("8pt dense [4.64-100]",        [4.64, 10.0, 15.0, 21.54, 30.0, 46.42, 70.0, 100.0]),
        ("Log 18pt [1-100]",             list(np.logspace(0, 2, 18))),
        ("Log 19pt [1-100] (best Sharpe from S10)", list(np.logspace(0, 2, 19))),
        ("Log 20pt [1-100] (best MDD from S10)",    list(np.logspace(0, 2, 20))),
        ("Log 21pt [1-100]",             list(np.logspace(0, 2, 21))),
    ]
    ref_results = {}
    for label, grid in grids_ref:
        s, m, stats = run_single_grid(df, grid, label)
        ref_results[label] = (s, m, stats)

    # ── Group 2: Grids with λ=0 prepended ────────────────────────────────────
    print("\n── Group 2: Grids with λ=0 included (paper: 'log-uniform 0.0 to 100.0') ──")

    # [0] + logspace(0, 2, N) = [0, 1, ..., 100]
    grids_with_zero = [
        ("[0]+Log17pt → 18 total [0,1-100]",  [0.0] + list(np.logspace(0, 2, 17))),
        ("[0]+Log18pt → 19 total [0,1-100]",  [0.0] + list(np.logspace(0, 2, 18))),
        ("[0]+Log19pt → 20 total [0,1-100]",  [0.0] + list(np.logspace(0, 2, 19))),
        ("[0]+Log20pt → 21 total [0,1-100]",  [0.0] + list(np.logspace(0, 2, 20))),
    ]
    zero_results = {}
    for label, grid in grids_with_zero:
        s, m, stats = run_single_grid(df, grid, label)
        zero_results[label] = (s, m, stats)

    # ── Group 3: Grids with λ=0 + sub-1 range ────────────────────────────────
    print("\n── Group 3: Grids with λ=0 + fine sub-1 range ──")

    grids_sub1 = [
        ("[0]+Log18pt [0.1-100]",   [0.0] + list(np.logspace(-1, 2, 18))),
        ("[0]+Log19pt [0.1-100]",   [0.0] + list(np.logspace(-1, 2, 19))),
        ("[0]+Log18pt [0.01-100]",  [0.0] + list(np.logspace(-2, 2, 18))),
    ]
    sub1_results = {}
    for label, grid in grids_sub1:
        s, m, stats = run_single_grid(df, grid, label)
        sub1_results[label] = (s, m, stats)

    # ── Summary ──────────────────────────────────────────────────────────────
    all_results = list(ref_results.items()) + list(zero_results.items()) + list(sub1_results.items())
    all_results = [(lbl, s, m, st) for (lbl, (s, m, st)) in all_results if s is not None]

    print(f"\n{'='*70}")
    print("SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"{'Grid':<45} {'Sharpe':>8} {'ΔSharpe':>8} {'MDD':>8} {'ΔMDD':>8} {'Bear%':>7} {'Shifts':>7}")
    print(f"{'─'*45} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*7} {'─'*7}")
    print(f"{'Paper target':<45} {'0.790':>8} {'':>8} {'-17.69%':>8} {'':>8} {'20.9%':>7} {'46':>7}")
    print(f"{'─'*45} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*7} {'─'*7}")

    for lbl, s, m, st in sorted(all_results, key=lambda x: -x[1]):
        ds = s - PAPER_SHARPE
        dm = m - PAPER_MDD
        bp = f"{st['bear_pct']:.1f}%" if st else "?"
        ns = str(st['n_shifts']) if st else "?"
        print(f"  {lbl:<43} {s:>8.3f} {ds:>+8.3f} {m:>8.1%} {dm:>+8.1%} {bp:>7} {ns:>7}")

    # Best by Sharpe
    best_sharpe_row = max(all_results, key=lambda x: x[1])
    # Best combined (closest to both Sharpe 0.79 AND MDD -17.69%)
    best_combined = min(all_results, key=lambda x: abs(x[1] - PAPER_SHARPE) + abs(x[2] - PAPER_MDD) * 5)

    print(f"\n  Best Sharpe:   {best_sharpe_row[0]}  → {best_sharpe_row[1]:.3f}")
    print(f"  Best combined: {best_combined[0]}  → Sharpe={best_combined[1]:.3f}  MDD={best_combined[2]:.1%}")

    return all_results


def run_regime_detail(df, grid, label):
    """Run a single grid and print full bear period detail."""
    print(f"\n{'='*70}")
    print(f"REGIME DETAIL: {label}")
    print(f"Grid: {[round(x, 3) for x in grid]}")
    print(f"{'='*70}")
    run_single_grid(df, grid, label, show_periods=True)


def run_bh_baseline(df):
    """Compute and print B&H stats for reference."""
    oos_data = df[(df.index >= '2007-01-01') & (df.index <= '2023-12-31')]
    bh_ret, bh_vol, bh_sharpe, bh_sortino, bh_mdd = main.calculate_metrics(
        oos_data['Target_Return'], oos_data['RF_Rate']
    )
    print(f"\n  B&H: Sharpe={bh_sharpe:.3f} (paper: {PAPER_BH_SHARPE})  MDD={bh_mdd:.1%} (paper: {PAPER_BH_MDD:.2%})")


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else '--sweep'

    df = prepare_bloomberg_features()
    # Clear cache once at startup — shared across all grid runs for speed
    main._forecast_cache.clear()
    run_bh_baseline(df)

    if mode == '--regime':
        # Just compute regime stats for the current default grid
        grid = [4.64, 10.0, 15.0, 21.54, 30.0, 46.42, 70.0, 100.0]
        run_regime_detail(df, grid, "Current 8pt dense")

    elif mode == '--full':
        all_results = run_grid_sweep(df)
        # Show bear periods for the best combined match
        best_combined = min(
            [(lbl, s, m, st) for (lbl, s, m, st) in all_results if s is not None],
            key=lambda x: abs(x[1] - PAPER_SHARPE) + abs(x[2] - PAPER_MDD) * 5
        )
        best_label = best_combined[0]
        # Find the matching grid
        all_grids = {
            "8pt dense [4.64-100]":        [4.64, 10.0, 15.0, 21.54, 30.0, 46.42, 70.0, 100.0],
            "[0]+Log18pt → 19 total [0,1-100]":  [0.0] + list(np.logspace(0, 2, 18)),
            "[0]+Log19pt → 20 total [0,1-100]":  [0.0] + list(np.logspace(0, 2, 19)),
            "Log 19pt [1-100] (best Sharpe from S10)": list(np.logspace(0, 2, 19)),
            "Log 20pt [1-100] (best MDD from S10)":    list(np.logspace(0, 2, 20)),
        }
        if best_label in all_grids:
            run_regime_detail(df, all_grids[best_label], f"Best combined: {best_label}")
        run_jm_only(df)

    else:  # --sweep (default)
        run_grid_sweep(df)
