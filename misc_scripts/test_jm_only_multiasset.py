"""
Test: JM-only strategy across all 12 assets vs Paper Table 4 JM row.
Uses the fixed StatisticalJumpModel (k-means++, n_init=10, matching paper's jumpmodels library).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'misc_scripts'))
from benchmark_assets import (
    fetch_etf_data, StatisticalJumpModel,
    LAMBDA_GRID, PAPER_EWMA_HL, TRANSACTION_COST,
    VALIDATION_WINDOW_YRS, DD_EXCLUDE_TICKERS
)
from config import StrategyConfig

# Paper Table 4 JM-only Sharpe ratios (2007-2023)
PAPER_JM_SHARPE = {
    '^SP500TR': 0.59,   # LargeCap
    'VIMSX':    0.49,   # MidCap
    'NAESX':    0.28,   # SmallCap
    'FDIVX':    0.28,   # EAFE
    'VEIEX':    0.65,   # EM
    'FRESX':    0.39,   # REIT
    'VBMFX':    0.43,   # AggBond
    'VUSTX':    0.21,   # Treasury
    'VWEHX':    1.49,   # HighYield
    'VWESX':    0.83,   # Corporate
    'GC=F':     0.12,   # Gold  (paper: Commodity=0.08, Gold=0.12)
}

PAPER_BH_SHARPE = {
    '^SP500TR': 0.50, 'VIMSX': 0.45, 'NAESX': 0.36, 'FDIVX': 0.20,
    'VEIEX': 0.20, 'FRESX': 0.27, 'VBMFX': 0.46, 'VUSTX': 0.26,
    'VWEHX': 0.67, 'VWESX': 0.54, 'GC=F': 0.43,
}

ASSET_NAMES = {
    '^SP500TR': 'LargeCap', 'VIMSX': 'MidCap', 'NAESX': 'SmallCap',
    'FDIVX': 'EAFE', 'VEIEX': 'EM', 'FRESX': 'REIT',
    'VBMFX': 'AggBond', 'VUSTX': 'Treasury', 'VWEHX': 'HighYield',
    'VWESX': 'Corporate', 'GC=F': 'Gold',
}

OOS_START = '2007-01-01'
OOS_END   = '2024-01-01'

config = StrategyConfig()


def calculate_metrics(returns, rf_rate):
    """Calculate annualized Sharpe, Sortino, MDD."""
    excess = returns - rf_rate
    ann_ret = excess.mean() * 252
    ann_vol = excess.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    neg = excess[excess < 0]
    down_vol = neg.std() * np.sqrt(252) if len(neg) > 0 else 1e-6
    sortino = ann_ret / down_vol
    cum = (1 + returns).cumprod()
    drawdown = cum / cum.cummax() - 1
    mdd = drawdown.min()
    return ann_ret, ann_vol, sharpe, sortino, mdd


def run_jm_only_backtest(ticker, df, lmbda, ewma_hl=0):
    """Run JM-only walk-forward for a single lambda."""
    oos_start_dt = pd.to_datetime(OOS_START)
    oos_end_dt   = pd.to_datetime(OOS_END)
    current      = oos_start_dt

    exclude_dd = ticker in DD_EXCLUDE_TICKERS
    if exclude_dd:
        ret_features = [c for c in df.columns if c.startswith(('Avg_Ret_', 'Sortino_'))]
    else:
        ret_features = [c for c in df.columns if c.startswith(('DD_', 'Avg_Ret_', 'Sortino_'))]

    all_dates  = []
    all_states = []

    while current < oos_end_dt:
        chunk_end   = min(current + pd.DateOffset(months=6), oos_end_dt)
        train_start = current - pd.DateOffset(years=11)
        train_df    = df[(df.index >= train_start) & (df.index < current)]
        oos_df      = df[(df.index >= current) & (df.index < chunk_end)]

        if len(train_df) < 252 * 5 or len(oos_df) == 0:
            current = chunk_end
            continue

        X_train = train_df[ret_features]
        m, s = X_train.mean(), X_train.std()
        s[s == 0] = 1.0
        X_train_std = (X_train - m) / s
        X_oos_std   = (oos_df[ret_features] - m) / s

        jm = StatisticalJumpModel(n_states=2, lambda_penalty=lmbda)
        states = jm.fit_predict(X_train_std.values)

        # Align: State 0 = Bullish
        c0 = train_df['Excess_Return'].values[states == 0].sum()
        c1 = train_df['Excess_Return'].values[states == 1].sum()
        if c1 > c0:
            states = 1 - states
            jm.means = jm.means[::-1].copy()

        oos_states = jm.predict_online(X_oos_std.values)

        all_dates.extend(oos_df.index.tolist())
        all_states.extend(oos_states.tolist())
        current = chunk_end

    if not all_dates:
        return None

    result = df.loc[all_dates, ['Target_Return', 'RF_Rate']].copy()
    result['Forecast_State'] = all_states

    # Apply strategy: state 0 = invest, state 1 = risk-free
    signals = result['Forecast_State'].shift(1).fillna(0)
    alloc   = 1.0 - signals
    strat   = alloc * result['Target_Return'] + (1 - alloc) * result['RF_Rate']
    trades  = alloc.diff().abs().fillna(0)
    result['Strat_Return'] = strat - trades * TRANSACTION_COST

    mask = (result.index >= oos_start_dt) & (result.index < oos_end_dt)
    result = result[mask]
    if result.empty:
        return None

    _, _, sharpe, sortino, mdd = calculate_metrics(result['Strat_Return'], result['RF_Rate'])
    _, _, bh_sharpe, _, bh_mdd = calculate_metrics(result['Target_Return'], result['RF_Rate'])
    bear_pct = (result['Forecast_State'] == 1).mean()
    shifts = (result['Forecast_State'].diff().abs() > 0).sum()

    return {
        'sharpe': sharpe, 'bh_sharpe': bh_sharpe, 'mdd': mdd, 'bh_mdd': bh_mdd,
        'bear_pct': bear_pct, 'shifts': shifts, 'sortino': sortino,
    }


def run_wf_jm_only(ticker, df, ewma_hl=0):
    """Walk-forward: tune lambda on validation, then run OOS JM-only."""
    oos_start_dt = pd.to_datetime(OOS_START)
    oos_end_dt   = pd.to_datetime(OOS_END)
    current      = oos_start_dt
    lambda_hist  = []

    exclude_dd = ticker in DD_EXCLUDE_TICKERS
    if exclude_dd:
        ret_features = [c for c in df.columns if c.startswith(('Avg_Ret_', 'Sortino_'))]
    else:
        ret_features = [c for c in df.columns if c.startswith(('DD_', 'Avg_Ret_', 'Sortino_'))]

    all_dates  = []
    all_states = []

    while current < oos_end_dt:
        chunk_end = min(current + pd.DateOffset(months=6), oos_end_dt)
        val_start = current - pd.DateOffset(years=VALIDATION_WINDOW_YRS)

        # Lambda tuning on validation window
        best_sharpe = -np.inf
        best_lambda = LAMBDA_GRID[len(LAMBDA_GRID) // 2]
        for lmbda in LAMBDA_GRID:
            val_res = _run_jm_chunk(df, val_start, current, lmbda, ret_features, ticker)
            if val_res is not None and len(val_res) > 0:
                signals = val_res['Forecast_State'].shift(1).fillna(0)
                alloc = 1.0 - signals
                strat = alloc * val_res['Target_Return'] + (1 - alloc) * val_res['RF_Rate']
                trades = alloc.diff().abs().fillna(0)
                strat -= trades * TRANSACTION_COST
                excess = strat - val_res['RF_Rate']
                ann_ret = excess.mean() * 252
                ann_vol = excess.std() * np.sqrt(252)
                s = ann_ret / ann_vol if ann_vol > 0 else 0.0
                if not np.isnan(s) and s > best_sharpe:
                    best_sharpe = s
                    best_lambda = lmbda

        lambda_hist.append(best_lambda)

        # OOS chunk with best lambda
        oos_res = _run_jm_chunk(df, current, chunk_end, best_lambda, ret_features, ticker)
        if oos_res is not None:
            all_dates.extend(oos_res.index.tolist())
            all_states.extend(oos_res['Forecast_State'].tolist())

        current = chunk_end

    if not all_dates:
        return None, lambda_hist

    result = df.loc[all_dates, ['Target_Return', 'RF_Rate']].copy()
    result['Forecast_State'] = all_states

    signals = result['Forecast_State'].shift(1).fillna(0)
    alloc = 1.0 - signals
    strat = alloc * result['Target_Return'] + (1 - alloc) * result['RF_Rate']
    trades = alloc.diff().abs().fillna(0)
    result['Strat_Return'] = strat - trades * TRANSACTION_COST

    mask = (result.index >= oos_start_dt) & (result.index < oos_end_dt)
    result = result[mask]
    if result.empty:
        return None, lambda_hist

    _, _, sharpe, sortino, mdd = calculate_metrics(result['Strat_Return'], result['RF_Rate'])
    _, _, bh_sharpe, _, _ = calculate_metrics(result['Target_Return'], result['RF_Rate'])
    bear_pct = (result['Forecast_State'] == 1).mean()
    shifts = (result['Forecast_State'].diff().abs() > 0).sum()

    return {
        'sharpe': sharpe, 'bh_sharpe': bh_sharpe, 'mdd': mdd,
        'bear_pct': bear_pct, 'shifts': shifts, 'sortino': sortino,
    }, lambda_hist


def _run_jm_chunk(df, start, end, lmbda, ret_features, ticker):
    """Run JM predict_online on a single chunk."""
    start_dt = pd.to_datetime(start)
    end_dt   = pd.to_datetime(end)
    train_start = start_dt - pd.DateOffset(years=11)
    train_df = df[(df.index >= train_start) & (df.index < start_dt)]
    oos_df   = df[(df.index >= start_dt) & (df.index < end_dt)]

    if len(train_df) < 252 * 5 or len(oos_df) == 0:
        return None

    X_train = train_df[ret_features]
    m, s = X_train.mean(), X_train.std()
    s[s == 0] = 1.0
    X_train_std = (X_train - m) / s
    X_oos_std   = (oos_df[ret_features] - m) / s

    jm = StatisticalJumpModel(n_states=2, lambda_penalty=lmbda)
    states = jm.fit_predict(X_train_std.values)

    c0 = train_df['Excess_Return'].values[states == 0].sum()
    c1 = train_df['Excess_Return'].values[states == 1].sum()
    if c1 > c0:
        jm.means = jm.means[::-1].copy()

    oos_states = jm.predict_online(X_oos_std.values)

    result = oos_df[['Target_Return', 'RF_Rate']].copy()
    result['Forecast_State'] = oos_states
    return result


# ============================================================================
print("=" * 80)
print("  MULTI-ASSET JM-ONLY BASELINE: Fixed JM vs Paper Table 4 (2007-2023)")
print(f"  Lambda Grid: {LAMBDA_GRID}")
print("=" * 80)

all_results = {}

for ticker in PAPER_JM_SHARPE.keys():
    asset = ASSET_NAMES[ticker]
    paper_jm = PAPER_JM_SHARPE[ticker]

    print(f"\n{'='*70}")
    print(f"  {ticker} ({asset}) — Paper JM Sharpe: {paper_jm}")
    print(f"{'='*70}")

    _, df_asset = fetch_etf_data(ticker, '1975-01-01')
    if df_asset is None:
        print(f"  FAILED to fetch data")
        continue

    # Test each lambda individually
    best_res = None
    print(f"  {'Lambda':>8} {'Sharpe':>8} {'B&H':>6} {'Bear%':>6} {'Shifts':>7} {'vs Paper':>9}")
    print(f"  " + "-" * 50)
    for lmbda in LAMBDA_GRID:
        res = run_jm_only_backtest(ticker, df_asset, lmbda)
        if res:
            gap = res['sharpe'] - paper_jm
            print(f"  {lmbda:>8.2f} {res['sharpe']:>8.3f} {res['bh_sharpe']:>6.3f} {res['bear_pct']:>5.1%} {res['shifts']:>7} {gap:>+9.3f}")
            if best_res is None or res['sharpe'] > best_res['sharpe']:
                best_res = {'lambda': lmbda, **res}

    # Walk-forward tuned
    wf_res, wf_lambdas = run_wf_jm_only(ticker, df_asset)
    if wf_res:
        gap = wf_res['sharpe'] - paper_jm
        print(f"  {'WF-tuned':>8} {wf_res['sharpe']:>8.3f} {wf_res['bh_sharpe']:>6.3f} {wf_res['bear_pct']:>5.1%} {wf_res['shifts']:>7} {gap:>+9.3f}")
        all_results[ticker] = {'wf': wf_res, 'best_fixed': best_res, 'wf_lambdas': wf_lambdas}
    elif best_res:
        all_results[ticker] = {'wf': best_res, 'best_fixed': best_res, 'wf_lambdas': []}


# ============================================================================
print(f"\n\n{'='*80}")
print("  SUMMARY: JM-Only Walk-Forward vs Paper Table 4 JM")
print(f"{'='*80}")
print(f"{'Ticker':<10} {'Asset':<10} {'Our JM':>7} {'Paper JM':>9} {'Gap':>7} {'Our B&H':>8} {'Paper B&H':>10} {'Beat Paper?':>12}")
print("-" * 80)

wins = 0
total = 0
for ticker in PAPER_JM_SHARPE.keys():
    if ticker not in all_results:
        continue
    asset = ASSET_NAMES[ticker]
    paper_jm = PAPER_JM_SHARPE[ticker]
    paper_bh = PAPER_BH_SHARPE[ticker]
    wf = all_results[ticker]['wf']
    gap = wf['sharpe'] - paper_jm
    beat = "YES" if gap >= -0.05 else "no"
    if gap >= -0.05:
        wins += 1
    total += 1
    print(f"{ticker:<10} {asset:<10} {wf['sharpe']:>7.3f} {paper_jm:>9.2f} {gap:>+7.3f} {wf['bh_sharpe']:>8.3f} {paper_bh:>10.2f} {beat:>12}")

print(f"\nMatch rate (within 0.05): {wins}/{total} ({wins/total:.0%})")

print(f"\n{'='*80}")
print("  BEST-LAMBDA COMPARISON (oracle)")
print(f"{'='*80}")
print(f"{'Ticker':<10} {'Asset':<10} {'Best λ':>7} {'Sharpe':>7} {'Paper JM':>9} {'Gap':>7}")
print("-" * 55)
for ticker in PAPER_JM_SHARPE.keys():
    if ticker not in all_results:
        continue
    asset = ASSET_NAMES[ticker]
    paper_jm = PAPER_JM_SHARPE[ticker]
    bf = all_results[ticker]['best_fixed']
    if bf:
        gap = bf['sharpe'] - paper_jm
        print(f"{ticker:<10} {asset:<10} {bf['lambda']:>7.2f} {bf['sharpe']:>7.3f} {paper_jm:>9.2f} {gap:>+7.3f}")
