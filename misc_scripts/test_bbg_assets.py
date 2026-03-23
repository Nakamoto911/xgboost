#!/usr/bin/env python3
"""
Test n_estimators=200 vs 100 on Bloomberg data for REIT and AggBond.
Compares walk-forward Sharpe/MDD against paper Table 4 targets.

Usage:
  python misc_scripts/test_bbg_assets.py            # REIT + AggBond
  python misc_scripts/test_bbg_assets.py REIT        # single asset
  python misc_scripts/test_bbg_assets.py AggBond
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

os.environ['XGB_END_DATE'] = '2023-12-31'
os.environ['XGB_OOS_START_DATE'] = '2007-01-01'
os.environ['XGB_START_DATE_DATA'] = '1987-01-01'

import main
from config import StrategyConfig

main.END_DATE = '2023-12-31'
main.OOS_START_DATE = '2007-01-01'

# Paper Table 4 targets (2007-2023)
PAPER_TARGETS = {
    'REIT':    {'sharpe': 0.56, 'mdd': -0.3270, 'bh_sharpe': 0.27, 'bear_pct': 18.4, 'shifts': 46},
    'AggBond': {'sharpe': 0.67, 'mdd': -0.0630, 'bh_sharpe': 0.46, 'bear_pct': 41.5, 'shifts': 97},
}

GRID_8PT = [4.64, 10.0, 15.0, 21.54, 30.0, 46.42, 70.0, 100.0]

# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_bbg_raw():
    """Load Bloomberg Excel, FRED, VIX, IRX. Returns (raw_df, fred, vix, irx)."""
    import yfinance as yf
    xlsx = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'cache', 'DATA PAUL.xlsx')
    print("Loading Bloomberg data...")
    raw = pd.read_excel(xlsx, header=None, skiprows=6)
    raw.columns = ['Date', 'SPTR', 'SPTRMDCP', 'RU20INTR', 'NDDUEAFE', 'NDUEEGF',
                   'LBUSTRUU', 'IBOXHY', 'LUACTRUU', 'DJUSRET', 'DBLCDBCE', 'GOLDLNPM', 'LUTLTRUU']
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw = raw.set_index('Date').sort_index()

    fred = main._fetch_fred_data().ffill().dropna()

    def _s(d):
        return (d['Adj Close'].iloc[:, 0] if isinstance(d.columns, pd.MultiIndex)
                else d.get('Adj Close', d['Close']))

    vix = _s(yf.download('^VIX', start='1987-01-01', end='2024-01-01',
                         auto_adjust=False, progress=False)).rename('VIX')
    irx = _s(yf.download('^IRX', start='1987-01-01', end='2024-01-01',
                         auto_adjust=False, progress=False)).rename('IRX')
    return raw, fred, vix, irx


def build_features(raw, fred, vix, irx, target_col, include_dd=True):
    """
    Build feature DataFrame for a given Bloomberg target column.
    include_dd=False for AggBond/Treasury/Gold per paper Table 2.
    Stock_Bond_Corr always uses SPTR vs LBUSTRUU (market feature, same for all assets).
    """
    cols = list(dict.fromkeys([target_col, 'SPTR', 'LBUSTRUU']))  # dedup, preserve order
    df = raw[cols].join(fred, how='inner') \
             .join(vix, how='inner').join(irx, how='inner').ffill().dropna()

    f = pd.DataFrame(index=df.index)
    tr = df[target_col].pct_change().fillna(0)
    f['Target_Return']       = tr
    f['Target_Intraday_Ret'] = tr
    f['Target_Overnight_Ret'] = 0.0
    f['RF_Rate']      = (df['IRX'] / 100) / 252
    f['Excess_Return'] = tr - f['RF_Rate']

    dn = np.minimum(f['Excess_Return'], 0)

    if include_dd:
        for hl in [5, 21]:
            ewm_dd = np.sqrt((dn ** 2).ewm(halflife=hl).mean()).fillna(0)
            f[f'DD_log_{hl}'] = np.log(ewm_dd + 1e-8)

    for hl in [5, 10, 21]:
        f[f'Avg_Ret_{hl}'] = f['Excess_Return'].ewm(halflife=hl).mean()

    for hl in [5, 10, 21]:
        dd_r = np.maximum(np.sqrt((dn ** 2).ewm(halflife=hl).mean()).fillna(1e-8), 1e-8)
        f[f'Sortino_{hl}'] = (f[f'Avg_Ret_{hl}'] / dd_r).clip(-10, 10)

    f['Yield_2Y_EWMA_diff']      = df['DGS2'].diff().fillna(0).ewm(halflife=21).mean()
    sl = df['DGS10'] - df['DGS2']
    f['Yield_Slope_EWMA_10']     = sl.ewm(halflife=10).mean()
    f['Yield_Slope_EWMA_diff_21'] = sl.diff().fillna(0).ewm(halflife=21).mean()
    f['VIX_EWMA_log_diff'] = (np.log(df['VIX'] / df['VIX'].shift(1))
                               .fillna(0).ewm(halflife=63).mean())
    # Stock-Bond Corr: always SPTR vs LBUSTRUU
    sptr_ret = df['SPTR'].pct_change().fillna(0)
    bond_ret = df['LBUSTRUU'].pct_change().fillna(0)
    f['Stock_Bond_Corr'] = sptr_ret.rolling(252).corr(bond_ret).fillna(0)

    return f.dropna()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def oos(r):
    return r[(r.index >= '2007-01-01') & (r.index <= '2023-12-31')]


def mets(r):
    o = oos(r)
    _, _, s, _, m = main.calculate_metrics(o['Strat_Return'], o['RF_Rate'])
    return s, m


def reg_stats(r):
    o = oos(r)
    if 'Forecast_State' not in o.columns:
        return None, None
    st = o['Forecast_State']
    return (st == 1).mean() * 100, max(int((st != st.shift(1)).sum()) - 1, 0)


def bh_sharpe(df_feat):
    """B&H Sharpe on OOS window."""
    o = oos(df_feat)
    _, _, s, _, _ = main.calculate_metrics(o['Target_Return'], o['RF_Rate'])
    return s


def run_asset(asset_name, df_feat, paper):
    print(f"\n{'='*70}")
    print(f"  {asset_name}")
    print(f"  Paper: S={paper['sharpe']}  MDD={paper['mdd']:.2%}  "
          f"B&H={paper['bh_sharpe']}  Bear={paper['bear_pct']}%  Shifts={paper['shifts']}")
    print(f"{'='*70}")
    n_feats = [c for c in df_feat.columns if c.startswith(('DD_', 'Avg_Ret_', 'Sortino_'))]
    print(f"  Return features ({len(n_feats)}): {n_feats}")
    bh = bh_sharpe(df_feat)
    print(f"  B&H Sharpe (OOS): {bh:.3f}  (paper: {paper['bh_sharpe']})\n")

    configs = [
        ("n_est=100 [baseline]", {}),
        ("n_est=200",            {'n_estimators': 200}),
        ("n_est=300",            {'n_estimators': 300}),
        ("exact + n_est=200",    {'tree_method': 'exact', 'n_estimators': 200}),
    ]

    fmt = "  {:<40} S={:.3f}({:+.3f}) MDD={:.1%} Bear={:.1f}% Shft={} λ̄={:.1f}"
    for label, xp in configs:
        cfg = StrategyConfig(name=f'{asset_name}_{label}', ewma_mode='paper', xgb_params=xp)
        main.LAMBDA_GRID = GRID_8PT
        main._forecast_cache.clear()
        r = main.walk_forward_backtest(df_feat, cfg)
        if r is None or r.empty:
            print(f"  {label:<40} ERROR")
            continue
        s, m = mets(r)
        bp, ns = reg_stats(r)
        lh = np.array(r.attrs.get('lambda_history', [0]))
        print(fmt.format(label, s, s - paper['sharpe'], m, bp or 0, ns or 0, lh.mean()))

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = sys.argv[1:] if len(sys.argv) > 1 else ['REIT', 'AggBond']
    args = [a.upper() for a in args]

    raw, fred, vix, irx = load_bbg_raw()

    if 'REIT' in args:
        print("\nBuilding REIT (DJUSRET) features  [DD included, hl=8]...")
        df_reit = build_features(raw, fred, vix, irx, target_col='DJUSRET', include_dd=True)
        print(f"  {df_reit.index.min().date()} → {df_reit.index.max().date()}, {len(df_reit)} rows")
        run_asset('REIT', df_reit, PAPER_TARGETS['REIT'])

    if 'AGGBOND' in args:
        print("\nBuilding AggBond (LBUSTRUU) features  [DD excluded, hl=8]...")
        df_agg = build_features(raw, fred, vix, irx, target_col='LBUSTRUU', include_dd=False)
        print(f"  {df_agg.index.min().date()} → {df_agg.index.max().date()}, {len(df_agg)} rows")
        run_asset('AggBond', df_agg, PAPER_TARGETS['AggBond'])

    print("\n" + "="*70)
    print("DONE")
    print("="*70)
