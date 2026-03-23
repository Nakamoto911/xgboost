#!/usr/bin/env python3
"""
Test Bloomberg SPTR data against paper results.

Paper target for LargeCap (SPTR) 2007-2023:
  - JM-XGB Sharpe: 0.79
  - JM-XGB MDD: -17.69%
  - JM-only Sharpe: 0.59
  - B&H Sharpe: 0.50
  - B&H MDD: -55.25%

This script replaces Yahoo ^SP500TR with Bloomberg SPTR Index data
and Bloomberg LBUSTRUU for Stock-Bond Correlation (instead of VBMFX).
VIX and FRED yields still come from Yahoo/FRED.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

# --- Override main.py module-level constants BEFORE importing ---
# We need to set paper-matching parameters
os.environ['XGB_END_DATE'] = '2023-12-31'
os.environ['XGB_OOS_START_DATE'] = '2007-01-01'
os.environ['XGB_START_DATE_DATA'] = '1987-01-01'

import main
from config import StrategyConfig

# Force paper date range
main.END_DATE = '2023-12-31'
main.OOS_START_DATE = '2007-01-01'


def load_bloomberg_sptr():
    """Load Bloomberg SPTR and LBUSTRUU (AggBond) from Excel."""
    xlsx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'cache', 'datab.xlsx')
    df = pd.read_excel(xlsx_path, header=None, skiprows=6)
    df.columns = ['Date', 'SPTR', 'SPTRMDCP', 'RU20INTR', 'NDDUEAFE', 'NDUEEGF',
                  'LBUSTRUU', 'IBOXHY', 'LUACTRUU', 'DJUSRET', 'DBLCDBCE',
                  'GOLDLNPM', 'LUTLTRUU']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    # Keep only SPTR and LBUSTRUU (AggBond for Stock-Bond Corr)
    df = df[['SPTR', 'LBUSTRUU']].dropna()
    return df


def prepare_bloomberg_features():
    """Build feature DataFrame using Bloomberg SPTR data + Yahoo VIX + FRED yields."""
    print("Loading Bloomberg data...")
    bbg = load_bloomberg_sptr()
    print(f"  SPTR: {bbg.index.min().date()} to {bbg.index.max().date()}, {len(bbg)} rows")

    # Fetch FRED data (DGS2, DGS10) - uses cached if available
    fred_data = main._fetch_fred_data()
    fred_data = fred_data.ffill().dropna()

    # Fetch VIX and IRX from Yahoo
    import yfinance as yf
    print("Fetching VIX and IRX from Yahoo...")
    vix = yf.download('^VIX', start='1987-01-01', end='2024-01-01', auto_adjust=False)
    irx = yf.download('^IRX', start='1987-01-01', end='2024-01-01', auto_adjust=False)

    # Extract Adj Close
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

    # Combine all data
    df = bbg.join(fred_data, how='inner')
    df = df.join(vix_series, how='inner')
    df = df.join(irx_series, how='inner')
    df = df.ffill().dropna()

    print(f"  Combined data: {df.index.min().date()} to {df.index.max().date()}, {len(df)} rows")

    # --- Feature Engineering (mirrors main.fetch_and_prepare_data) ---
    print("Calculating features...")
    features = pd.DataFrame(index=df.index)

    # Target returns from Bloomberg SPTR
    target_returns = df['SPTR'].pct_change().fillna(0)
    features['Target_Return'] = target_returns

    # No intraday/overnight split for Bloomberg data
    features['Target_Intraday_Ret'] = target_returns
    features['Target_Overnight_Ret'] = 0.0

    # Risk-free daily rate
    features['RF_Rate'] = (df['IRX'] / 100) / 252
    features['Excess_Return'] = target_returns - features['RF_Rate']

    # A. Return Features
    downside_returns = np.minimum(features['Excess_Return'], 0)

    # DD features (LargeCap is NOT in DD_EXCLUDE_TICKERS)
    for hl in [5, 21]:
        ewm_var = (downside_returns ** 2).ewm(halflife=hl).mean()
        ewm_dd = np.sqrt(ewm_var).fillna(0)
        features[f'DD_log_{hl}'] = np.log(ewm_dd + 1e-8)

    # EWM Average Return
    for hl in [5, 10, 21]:
        features[f'Avg_Ret_{hl}'] = features['Excess_Return'].ewm(halflife=hl).mean()

    # EWM Sortino Ratio
    for hl in [5, 10, 21]:
        ewm_var = (downside_returns ** 2).ewm(halflife=hl).mean()
        ewm_dd_raw = np.sqrt(ewm_var).fillna(1e-8)
        ewm_dd_raw = np.maximum(ewm_dd_raw, 1e-8)
        features[f'Sortino_{hl}'] = (features[f'Avg_Ret_{hl}'] / ewm_dd_raw).clip(-10, 10)

    # B. Macro Features
    features['Yield_2Y_diff'] = df['DGS2'].diff().fillna(0)
    features['Yield_2Y_EWMA_diff'] = features['Yield_2Y_diff'].ewm(halflife=21).mean()

    slope = df['DGS10'] - df['DGS2']
    features['Yield_Slope_EWMA_10'] = slope.ewm(halflife=10).mean()
    slope_diff = slope.diff().fillna(0)
    features['Yield_Slope_EWMA_diff_21'] = slope_diff.ewm(halflife=21).mean()

    vix_log_diff = np.log(df['VIX'] / df['VIX'].shift(1)).fillna(0)
    features['VIX_EWMA_log_diff'] = vix_log_diff.ewm(halflife=63).mean()

    # Stock-Bond Correlation: corr(SPTR returns, LBUSTRUU returns) over 252 days
    largecap_returns = df['SPTR'].pct_change().fillna(0)
    bond_returns = df['LBUSTRUU'].pct_change().fillna(0)
    features['Stock_Bond_Corr'] = largecap_returns.rolling(window=252).corr(bond_returns).fillna(0)

    final_df = features.dropna()
    print(f"  Features: {final_df.index.min().date()} to {final_df.index.max().date()}, {len(final_df)} rows")
    return final_df


def run_test():
    """Run backtest with Bloomberg data and paper parameters."""
    df = prepare_bloomberg_features()

    # Paper Baseline config: ewma_mode="paper", no sub-window consensus
    paper_config = StrategyConfig(
        name="Paper Baseline (Bloomberg)",
        ewma_mode="paper",
        tuning_metric="sharpe",
        lambda_selection="best",
        lambda_subwindow_consensus=False,
        allocation_style="binary",
        prob_threshold=0.50,
    )

    # Also test our optimized config
    optimized_config = StrategyConfig(
        name="Optimized (Bloomberg)",
        ewma_mode="auto",
        tuning_metric="sharpe",
        lambda_selection="best",
        lambda_subwindow_consensus=True,
        allocation_style="binary",
        prob_threshold=0.50,
    )

    # Clear forecast cache
    main._forecast_cache.clear()

    print(f"\n{'='*70}")
    print(f"PAPER BASELINE (Bloomberg SPTR, 2007-2023)")
    print(f"{'='*70}")
    print(f"Lambda grid: {main.LAMBDA_GRID}")
    print(f"OOS: {main.OOS_START_DATE} to {main.END_DATE}")

    result = main.walk_forward_backtest(df, paper_config)

    if result.empty:
        print("ERROR: Empty result!")
        return

    # Filter to paper period
    result = result[(result.index >= '2007-01-01') & (result.index <= '2023-12-31')]

    ann_ret, ann_vol, sharpe, sortino, mdd = main.calculate_metrics(
        result['Strat_Return'], result['RF_Rate']
    )
    bh_ret, bh_vol, bh_sharpe, bh_sortino, bh_mdd = main.calculate_metrics(
        result['Target_Return'], result['RF_Rate']
    )

    # Get lambda history from result attrs
    lambda_history = result.attrs.get('lambda_history', [])
    lambda_dates = result.attrs.get('lambda_dates', [])
    ewma_hl = result.attrs.get('ewma_halflife', '?')

    print(f"\nEWMA halflife used: {ewma_hl}")
    if lambda_history:
        print(f"Lambda history ({len(lambda_history)} periods):")
        for i, (d, l) in enumerate(zip(lambda_dates, lambda_history)):
            d_str = d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)
            print(f"  {d_str}: λ={l:.2f}")
        print(f"  Mean λ: {np.mean(lambda_history):.2f}, Std: {np.std(lambda_history):.2f}")

    print(f"\n{'─'*50}")
    print(f"{'Metric':<25} {'JM-XGB':>10} {'B&H':>10} {'Paper JM-XGB':>14} {'Paper B&H':>12}")
    print(f"{'─'*50}")
    print(f"{'Sharpe':<25} {sharpe:>10.3f} {bh_sharpe:>10.3f} {'0.790':>14} {'0.500':>12}")
    print(f"{'Sortino':<25} {sortino:>10.3f} {bh_sortino:>10.3f} {'':>14} {'':>12}")
    print(f"{'Ann Return':<25} {ann_ret:>10.1%} {bh_ret:>10.1%} {'':>14} {'':>12}")
    print(f"{'Ann Vol':<25} {ann_vol:>10.1%} {bh_vol:>10.1%} {'':>14} {'':>12}")
    print(f"{'Max DD':<25} {mdd:>10.1%} {bh_mdd:>10.1%} {'-17.69%':>14} {'-55.25%':>12}")
    print(f"{'─'*50}")

    sharpe_gap = sharpe - 0.79
    print(f"\nSharpe gap vs paper: {sharpe_gap:+.3f}")
    print(f"MDD gap vs paper:   {(mdd - (-0.1769)):+.1%}")

    # --- Also run JM-only baseline for comparison ---
    print(f"\n{'='*70}")
    print(f"JM-ONLY BASELINE (Bloomberg SPTR, 2007-2023)")
    print(f"{'='*70}")

    main._forecast_cache.clear()

    # For JM-only, we need to use simulate_strategy directly with a fixed lambda
    # Paper reports JM Sharpe 0.59 at some lambda. Try λ=100 (our best JM-only)
    jm_results = []
    best_jm_sharpe = -np.inf
    best_jm_lambda = None

    for lmbda in main.LAMBDA_GRID:
        res = main.simulate_strategy(df, '2007-01-01', '2023-12-31', lmbda,
                                     paper_config, include_xgboost=False)
        if not res.empty:
            _, _, s, _, _ = main.calculate_metrics(res['Strat_Return'], res['RF_Rate'])
            if s > best_jm_sharpe:
                best_jm_sharpe = s
                best_jm_lambda = lmbda
                best_jm_res = res

    print(f"Best JM-only: λ={best_jm_lambda}, Sharpe={best_jm_sharpe:.3f} (paper: 0.59)")
    _, _, _, _, jm_mdd = main.calculate_metrics(best_jm_res['Strat_Return'], best_jm_res['RF_Rate'])
    print(f"JM-only MDD: {jm_mdd:.1%} (paper: -24.78%)")

    # --- Now run optimized config ---
    print(f"\n{'='*70}")
    print(f"OPTIMIZED (Bloomberg SPTR, 2007-2023)")
    print(f"{'='*70}")

    main._forecast_cache.clear()
    result2 = main.walk_forward_backtest(df, optimized_config)
    if not result2.empty:
        result2 = result2[(result2.index >= '2007-01-01') & (result2.index <= '2023-12-31')]
        ann_ret2, ann_vol2, sharpe2, sortino2, mdd2 = main.calculate_metrics(
            result2['Strat_Return'], result2['RF_Rate']
        )
        ewma_hl2 = result2.attrs.get('ewma_halflife', '?')
        print(f"EWMA halflife: {ewma_hl2}")
        print(f"Sharpe: {sharpe2:.3f}, Sortino: {sortino2:.3f}, MDD: {mdd2:.1%}")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<30} {'Sharpe':>8} {'MDD':>10}")
    print(f"{'Paper JM-XGB':<30} {'0.790':>8} {'-17.69%':>10}")
    print(f"{'Paper Baseline (Bloomberg)':<30} {sharpe:>8.3f} {mdd:>10.1%}")
    if not result2.empty:
        print(f"{'Optimized (Bloomberg)':<30} {sharpe2:>8.3f} {mdd2:>10.1%}")
    print(f"{'Paper B&H':<30} {'0.500':>8} {'-55.25%':>10}")
    print(f"{'Our B&H (Bloomberg)':<30} {bh_sharpe:>8.3f} {bh_mdd:>10.1%}")


if __name__ == '__main__':
    run_test()
