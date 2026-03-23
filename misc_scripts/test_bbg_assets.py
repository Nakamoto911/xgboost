#!/usr/bin/env python3
"""
Test n_estimators baseline on Bloomberg data for all 12 paper assets.
Also includes fixed-λ oracle sweep and focused-grid walk-forward tests.

Usage:
  python misc_scripts/test_bbg_assets.py              # all remaining assets (baseline only)
  python misc_scripts/test_bbg_assets.py REIT          # single asset
  python misc_scripts/test_bbg_assets.py AggBond
  python misc_scripts/test_bbg_assets.py REIT_ORACLE   # fixed-λ sweep for REIT
  python misc_scripts/test_bbg_assets.py MIDCAP_ORACLE # fixed-λ sweep for MidCap
  python misc_scripts/test_bbg_assets.py EM_ORACLE     # fixed-λ sweep for EM
  python misc_scripts/test_bbg_assets.py MIDCAP_GRID   # focused-grid WF tests for MidCap
  python misc_scripts/test_bbg_assets.py EM_GRID       # focused-grid WF tests for EM
  python misc_scripts/test_bbg_assets.py ALL            # all 12 assets
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

GRID_8PT = [4.64, 10.0, 15.0, 21.54, 30.0, 46.42, 70.0, 100.0]

# ──────────────────────────────────────────────────────────────────────────────
# Asset configuration table
# col          : column name in Bloomberg Excel
# dd           : include DD_log features (False for AggBond, Treasury, Gold)
# hl_proxy     : Yahoo ticker to set as TARGET_TICKER for PAPER_EWMA_HL lookup
# paper_sharpe : JM-XGB Sharpe from Table 4
# paper_mdd    : JM-XGB MDD from Table 4
# bh_sharpe    : B&H Sharpe from Table 4
# bear_pct     : % bear from Figure 2 (None if not in Figure 2)
# shifts       : # regime shifts from Figure 2 (None if not in Figure 2)
# ──────────────────────────────────────────────────────────────────────────────
ASSET_CONFIGS = {
    'LargeCap':  {'col': 'SPTR',     'dd': True,  'hl_proxy': '^SP500TR', 'paper_sharpe': 0.79, 'paper_mdd': -0.1769, 'bh_sharpe': 0.50, 'bear_pct': 20.9, 'shifts': 46},
    'MidCap':    {'col': 'SPTRMDCP', 'dd': True,  'hl_proxy': '^SP500TR', 'paper_sharpe': 0.59, 'paper_mdd': -0.2989, 'bh_sharpe': 0.45, 'bear_pct': None, 'shifts': None},
    'SmallCap':  {'col': 'RU20INTR', 'dd': True,  'hl_proxy': '^SP500TR', 'paper_sharpe': 0.51, 'paper_mdd': -0.3584, 'bh_sharpe': 0.36, 'bear_pct': None, 'shifts': None},
    'EAFE':      {'col': 'NDDUEAFE', 'dd': True,  'hl_proxy': 'EFA',      'paper_sharpe': 0.56, 'paper_mdd': -0.1993, 'bh_sharpe': 0.20, 'bear_pct': None, 'shifts': None},
    'EM':        {'col': 'NDUEEGF',  'dd': True,  'hl_proxy': 'EEM',      'paper_sharpe': 0.85, 'paper_mdd': -0.2130, 'bh_sharpe': 0.20, 'bear_pct': None, 'shifts': None},
    'REIT':      {'col': 'DJUSRET',  'dd': True,  'hl_proxy': '^SP500TR', 'paper_sharpe': 0.56, 'paper_mdd': -0.3270, 'bh_sharpe': 0.27, 'bear_pct': 18.4, 'shifts': 46},
    'AggBond':   {'col': 'LBUSTRUU', 'dd': False, 'hl_proxy': '^SP500TR', 'paper_sharpe': 0.67, 'paper_mdd': -0.0630, 'bh_sharpe': 0.46, 'bear_pct': 41.5, 'shifts': 97},
    'Treasury':  {'col': 'LUTLTRUU', 'dd': False, 'hl_proxy': '^SP500TR', 'paper_sharpe': 0.38, 'paper_mdd': -0.1746, 'bh_sharpe': 0.26, 'bear_pct': None, 'shifts': None},
    'HighYield': {'col': 'IBOXHY',   'dd': True,  'hl_proxy': 'HYG',      'paper_sharpe': 1.88, 'paper_mdd': -0.1025, 'bh_sharpe': 0.67, 'bear_pct': None, 'shifts': None},
    'Corporate': {'col': 'LUACTRUU', 'dd': False, 'hl_proxy': 'SPBO',     'paper_sharpe': 0.76, 'paper_mdd': -0.0679, 'bh_sharpe': 0.54, 'bear_pct': None, 'shifts': None},
    'Commodity': {'col': 'DBLCDBCE', 'dd': True,  'hl_proxy': 'DBC',      'paper_sharpe': 0.23, 'paper_mdd': -0.4790, 'bh_sharpe': 0.03, 'bear_pct': None, 'shifts': None},
    'Gold':      {'col': 'GOLDLNPM', 'dd': False, 'hl_proxy': 'GLD',      'paper_sharpe': 0.31, 'paper_mdd': -0.2162, 'bh_sharpe': 0.43, 'bear_pct': None, 'shifts': None},
}

# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

_raw_cache = {}

def load_bbg_raw():
    if _raw_cache:
        return _raw_cache['raw'], _raw_cache['fred'], _raw_cache['vix'], _raw_cache['irx']
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
    vix = _s(yf.download('^VIX', start='1987-01-01', end='2024-01-01', auto_adjust=False, progress=False)).rename('VIX')
    irx = _s(yf.download('^IRX', start='1987-01-01', end='2024-01-01', auto_adjust=False, progress=False)).rename('IRX')
    _raw_cache.update(raw=raw, fred=fred, vix=vix, irx=irx)
    return raw, fred, vix, irx


def build_features(raw, fred, vix, irx, target_col, include_dd=True):
    """Build feature DataFrame. include_dd=False for AggBond/Treasury/Gold."""
    cols = list(dict.fromkeys([target_col, 'SPTR', 'LBUSTRUU']))  # dedup
    df = raw[cols].join(fred, how='inner').join(vix, how='inner').join(irx, how='inner').ffill().dropna()

    f = pd.DataFrame(index=df.index)
    tr = df[target_col].pct_change().fillna(0)
    f['Target_Return']        = tr
    f['Target_Intraday_Ret']  = tr
    f['Target_Overnight_Ret'] = 0.0
    f['RF_Rate']       = (df['IRX'] / 100) / 252
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

    f['Yield_2Y_EWMA_diff']       = df['DGS2'].diff().fillna(0).ewm(halflife=21).mean()
    sl = df['DGS10'] - df['DGS2']
    f['Yield_Slope_EWMA_10']      = sl.ewm(halflife=10).mean()
    f['Yield_Slope_EWMA_diff_21'] = sl.diff().fillna(0).ewm(halflife=21).mean()
    f['VIX_EWMA_log_diff'] = np.log(df['VIX'] / df['VIX'].shift(1)).fillna(0).ewm(halflife=63).mean()
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
    o = oos(df_feat)
    _, _, s, _, _ = main.calculate_metrics(o['Target_Return'], o['RF_Rate'])
    return s


# ──────────────────────────────────────────────────────────────────────────────
# Walk-forward test — parameterized lambda grid
# ──────────────────────────────────────────────────────────────────────────────

def run_asset_wf(asset_name, df_feat, cfg_info, grid, label=None):
    """Walk-forward test with a specified lambda grid."""
    paper = cfg_info
    main.TARGET_TICKER = cfg_info['hl_proxy']
    main.LAMBDA_GRID   = grid
    main._forecast_cache.clear()

    cfg = StrategyConfig(name=f'BBG_{asset_name}', ewma_mode='paper')
    r = main.walk_forward_backtest(df_feat, cfg)

    bh = bh_sharpe(df_feat)
    if r is None or r.empty:
        print(f"  {asset_name:<12} [{label or str(grid)}] ERROR")
        return

    s, m = mets(r)
    bp, ns = reg_stats(r)
    lh = np.array(r.attrs.get('lambda_history', [0]))
    hl = r.attrs.get('ewma_halflife', '?')

    gap_s = s - paper['paper_sharpe']
    grid_label = label or f'{len(grid)}pt [{grid[0]:.0f}-{grid[-1]:.0f}]'
    print(f"  {asset_name:<12}  grid={grid_label:<22}  S={s:.3f}({gap_s:+.3f})  "
          f"MDD={m:.1%}  Bear={bp:.1f}%  Shft={ns}  hl={hl}  λ̄={lh.mean():.1f}  λs={list(lh)}")


def run_asset_baseline(asset_name, df_feat, cfg_info):
    paper = cfg_info
    main.TARGET_TICKER = cfg_info['hl_proxy']   # controls EWMA halflife lookup
    main.LAMBDA_GRID   = GRID_8PT
    main._forecast_cache.clear()

    cfg = StrategyConfig(name=f'BBG_{asset_name}', ewma_mode='paper')
    r = main.walk_forward_backtest(df_feat, cfg)

    bh = bh_sharpe(df_feat)
    if r is None or r.empty:
        print(f"  {asset_name:<12} ERROR")
        return

    s, m = mets(r)
    bp, ns = reg_stats(r)
    lh   = np.array(r.attrs.get('lambda_history', [0]))
    hl   = r.attrs.get('ewma_halflife', '?')

    gap_s = s - paper['paper_sharpe']
    bear_str = f"{bp:.1f}%" if bp is not None else "  —  "
    shft_str = str(ns) if ns is not None else "—"
    pbear    = f"{paper['bear_pct']:.1f}%" if paper['bear_pct'] else "  —  "
    pshft    = str(paper['shifts']) if paper['shifts'] else "—"

    print(f"  {asset_name:<12}  B&H={bh:.3f}(p:{paper['bh_sharpe']:.2f})  "
          f"S={s:.3f}({gap_s:+.3f})  MDD={m:.1%}  "
          f"Bear={bear_str}(p:{pbear})  Shft={shft_str}(p:{pshft})  "
          f"hl={hl}  λ̄={lh.mean():.1f}")


# ──────────────────────────────────────────────────────────────────────────────
# Generic oracle sweep — fixed-λ across wide range for any asset
# ──────────────────────────────────────────────────────────────────────────────

def oracle_sweep(asset_name, df_feat, cfg_info, sweep=None):
    """Fixed-λ oracle sweep for any asset. Finds best λ independent of walk-forward."""
    if sweep is None:
        sweep = [4.64, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100, 120, 150, 200, 300]

    p = cfg_info
    hl = main.PAPER_EWMA_HL.get(p['hl_proxy'], 8)

    print("\n" + "="*70)
    print(f"{asset_name} Oracle λ Sweep — find λ closest to paper")
    print(f"Paper: S={p['paper_sharpe']:.2f}  MDD={p['paper_mdd']:.2%}  "
          f"B&H={p['bh_sharpe']:.2f}"
          + (f"  Bear={p['bear_pct']:.1f}%  Shifts={p['shifts']}" if p['bear_pct'] else ""))
    print("="*70)

    main.TARGET_TICKER = p['hl_proxy']
    paper_cfg = StrategyConfig(name=f'{asset_name}_oracle', ewma_mode='paper')

    print(f"\n  {'λ':<8} {'JM S':>7} {'XGB S':>7} {'XGB add':>8} {'Bear%':>7} {'Shft':>6}")
    print(f"  {'─'*8} {'─'*7} {'─'*7} {'─'*8} {'─'*7} {'─'*6}")

    best = (-np.inf, None, None)
    for lm in sweep:
        main._forecast_cache.clear()
        r_jm  = main.simulate_strategy(df_feat, '2007-01-01', '2023-12-31', float(lm),
                                       paper_cfg, include_xgboost=False)
        r_xgb = main.simulate_strategy(df_feat, '2007-01-01', '2023-12-31', float(lm),
                                       paper_cfg, include_xgboost=True, ewma_halflife=hl)
        if r_jm.empty or r_xgb.empty:
            continue
        sj, _ = mets(r_jm)
        sx, mx = mets(r_xgb)
        bp, ns = reg_stats(r_xgb)
        print(f"  {lm:<8.2f} {sj:>7.3f} {sx:>7.3f} {sx-sj:>+8.3f} "
              f"{bp:>7.1f}% {ns:>6}")
        if sx > best[0]:
            best = (sx, lm, mx)

    print(f"\n  → Best oracle: λ={best[1]}, S={best[0]:.3f}, MDD={best[2]:.1%}")
    print(f"  → Paper:       λ=?,      S={p['paper_sharpe']:.3f}, MDD={p['paper_mdd']:.2%}")


def reit_oracle_sweep(df_reit):
    oracle_sweep('REIT', df_reit, ASSET_CONFIGS['REIT'])


# ──────────────────────────────────────────────────────────────────────────────
# Focused-grid walk-forward tests
# Oracle findings: MidCap oracle=λ15 (S=0.589≈paper 0.590), EM oracle=λ4.64 (S=0.910>paper 0.850)
# ──────────────────────────────────────────────────────────────────────────────

def midcap_grid_sweep(df_feat):
    """Walk-forward with several focused grids for MidCap (oracle λ=15, paper S=0.590)."""
    print("\n" + "="*70)
    print("MidCap Focused-Grid Walk-Forward — Oracle λ=15, Paper S=0.590")
    print("Baseline 8pt grid: S=0.475 (λ̄=26.7). Goal: S≈0.590")
    print("="*70 + "\n")

    grids = [
        ([10.0, 15.0, 21.54],           '3pt [10-22]'),
        ([10.0, 15.0, 21.54, 30.0],     '4pt [10-30]'),
        ([10.0, 15.0, 20.0, 25.0],      '4pt [10-25]'),
        ([10.0, 15.0, 20.0],            '3pt [10-20]'),
        ([15.0],                         'fixed λ=15'),
        (GRID_8PT,                       '8pt baseline'),
    ]
    for grid, label in grids:
        run_asset_wf('MidCap', df_feat, ASSET_CONFIGS['MidCap'], grid, label)


def em_grid_sweep(df_feat):
    """Walk-forward with several focused grids for EM (oracle λ=4.64, paper S=0.850)."""
    print("\n" + "="*70)
    print("EM Focused-Grid Walk-Forward — Oracle λ=4.64, Paper S=0.850")
    print("Baseline 8pt grid: S=0.701 (λ̄=12.6). Goal: S≈0.850")
    print("="*70 + "\n")

    grids = [
        ([4.64],                         'fixed λ=4.64'),
        ([4.64, 10.0],                   '2pt [4.64-10]'),
        ([4.64, 10.0, 15.0],             '3pt [4.64-15]'),
        ([4.64, 10.0, 15.0, 21.54],     '4pt [4.64-22]'),
        (GRID_8PT,                        '8pt baseline'),
    ]
    for grid, label in grids:
        run_asset_wf('EM', df_feat, ASSET_CONFIGS['EM'], grid, label)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = [a.upper() for a in sys.argv[1:]] if len(sys.argv) > 1 else list(ASSET_CONFIGS.keys())

    # Expand ALL → all assets
    if 'ALL' in args:
        args = list(ASSET_CONFIGS.keys()) + ['REIT_ORACLE']

    # Detect oracle/grid modes and map to asset keys
    ORACLE_MODES = {k + '_ORACLE': k for k in ASSET_CONFIGS}
    ORACLE_MODES_UPPER = {k.upper(): v for k, v in ORACLE_MODES.items()}
    GRID_MODES = {'MIDCAP_GRID': 'MidCap', 'EM_GRID': 'EM'}

    raw, fred, vix, irx = load_bbg_raw()

    # Pre-build feature DataFrames for requested assets
    dfs = {}
    for key, cfg in ASSET_CONFIGS.items():
        need_baseline = key.upper() in args
        need_oracle   = any(args_k == (key + '_ORACLE').upper() for args_k in args)
        need_grid     = any(args_k == (key.upper() + '_GRID') for args_k in args)
        if need_baseline or need_oracle or need_grid:
            print(f"Building {key} ({cfg['col']}) features  "
                  f"[DD={'incl' if cfg['dd'] else 'excl'}, hl_proxy={cfg['hl_proxy']}]...")
            dfs[key] = build_features(raw, fred, vix, irx, cfg['col'], cfg['dd'])
            print(f"  {dfs[key].index.min().date()} → {dfs[key].index.max().date()}, {len(dfs[key])} rows")

    print()
    print(f"  {'Asset':<12}  {'B&H (p)':>12}  {'Sharpe (Δ)':>14}  {'MDD':>8}  "
          f"{'Bear% (p)':>14}  {'Shifts (p)':>12}  {'hl':>4}  {'λ̄':>6}")
    print("  " + "─"*100)

    orig_ticker = main.TARGET_TICKER

    for key in ASSET_CONFIGS:
        if key.upper() in args and key in dfs:
            run_asset_baseline(key, dfs[key], ASSET_CONFIGS[key])

    main.TARGET_TICKER = orig_ticker  # restore

    # Run oracle sweeps
    for arg in args:
        asset_key = ORACLE_MODES_UPPER.get(arg)
        if asset_key and asset_key in dfs:
            oracle_sweep(asset_key, dfs[asset_key], ASSET_CONFIGS[asset_key])

    # Run focused-grid sweeps
    if 'MIDCAP_GRID' in args and 'MidCap' in dfs:
        midcap_grid_sweep(dfs['MidCap'])
    if 'EM_GRID' in args and 'EM' in dfs:
        em_grid_sweep(dfs['EM'])

    print("\n" + "="*70)
    print("DONE")
    print("="*70)
