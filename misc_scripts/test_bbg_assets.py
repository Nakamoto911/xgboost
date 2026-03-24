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
  python misc_scripts/test_bbg_assets.py JM_TC0_BATCH  # Session 20: JM-only + TC=0 (all 12 assets)
  python misc_scripts/test_bbg_assets.py LARGECAT_JM_TC0  # Session 20: single asset
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


def build_features(raw, fred, vix, irx, target_col, include_dd=True, ewm_adjust=True,
                   dd_formula='log', sortino_clip=10):
    """Build feature DataFrame.
    include_dd=False: exclude DD_log features (AggBond/Treasury/Gold).
    ewm_adjust=False: standard recursive EWMA vs pandas default weighted init.
    dd_formula='log' (default): DD_log_hl = log(sqrt(EWM(dn^2,hl))+1e-8)
    dd_formula='raw': DD_hl = sqrt(EWM(dn^2,hl)) — no log transformation
    sortino_clip=10 (default): clip Sortino at ±10; None = no clipping."""
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
            ewm_dd = np.sqrt((dn ** 2).ewm(halflife=hl, adjust=ewm_adjust).mean()).fillna(0)
            if dd_formula == 'log':
                f[f'DD_log_{hl}'] = np.log(ewm_dd + 1e-8)
            else:  # 'raw'
                f[f'DD_log_{hl}'] = ewm_dd   # raw sqrt, same column name for pipeline compatibility
    for hl in [5, 10, 21]:
        f[f'Avg_Ret_{hl}'] = f['Excess_Return'].ewm(halflife=hl, adjust=ewm_adjust).mean()
    for hl in [5, 10, 21]:
        dd_r = np.maximum(np.sqrt((dn ** 2).ewm(halflife=hl, adjust=ewm_adjust).mean()).fillna(1e-8), 1e-8)
        sortino = f[f'Avg_Ret_{hl}'] / dd_r
        f[f'Sortino_{hl}'] = sortino.clip(-sortino_clip, sortino_clip) if sortino_clip is not None else sortino

    f['Yield_2Y_EWMA_diff']       = df['DGS2'].diff().fillna(0).ewm(halflife=21, adjust=ewm_adjust).mean()
    sl = df['DGS10'] - df['DGS2']
    f['Yield_Slope_EWMA_10']      = sl.ewm(halflife=10, adjust=ewm_adjust).mean()
    f['Yield_Slope_EWMA_diff_21'] = sl.diff().fillna(0).ewm(halflife=21, adjust=ewm_adjust).mean()
    f['VIX_EWMA_log_diff'] = np.log(df['VIX'] / df['VIX'].shift(1)).fillna(0).ewm(halflife=63, adjust=ewm_adjust).mean()
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
# JM-only validation mode sweep
# Hypothesis: using JM-only sim for λ selection (not JM+XGB) gives a smoother,
# more robust Sharpe landscape — prevents walk-forward from picking low-λ during
# post-crisis validation windows. Global fix that should work across all assets.
# ──────────────────────────────────────────────────────────────────────────────

LOG19PT = list(np.logspace(0, 2, 19))   # [1.0, 1.38, ..., 100.0]
LOG25PT = list(np.logspace(0, 2, 25))   # [1.0, 1.21, ..., 100.0]


def run_asset_wf_jmval(asset_name, df_feat, cfg_info, grid, label=None):
    """Walk-forward with JM-only λ validation (lambda_validation_mode='jm_only')."""
    main.TARGET_TICKER = cfg_info['hl_proxy']
    main.LAMBDA_GRID   = grid
    main._forecast_cache.clear()

    cfg = StrategyConfig(name=f'BBG_{asset_name}_jmval', ewma_mode='paper',
                         lambda_validation_mode='jm_only')
    r = main.walk_forward_backtest(df_feat, cfg)

    bh = bh_sharpe(df_feat)
    if r is None or r.empty:
        print(f"  {asset_name:<12} [{label or str(grid)}] ERROR")
        return

    s, m = mets(r)
    bp, ns = reg_stats(r)
    lh = np.array(r.attrs.get('lambda_history', [0]))
    hl = r.attrs.get('ewma_halflife', '?')

    gap_s = s - cfg_info['paper_sharpe']
    grid_label = label or f'{len(grid)}pt [{grid[0]:.0f}-{grid[-1]:.0f}]'
    print(f"  {asset_name:<12}  grid={grid_label:<22}  S={s:.3f}({gap_s:+.3f})  "
          f"MDD={m:.1%}  Bear={bp:.1f}%  Shft={ns}  hl={hl}  λ̄={lh.mean():.1f}  λs={list(np.round(lh,1))}")


def jm_val_sweep(asset_name, df_feat, cfg_info):
    """
    Test JM-only λ validation across grids.
    Prints baseline (XGB val) vs JM-only val side-by-side for comparison.
    """
    p = cfg_info
    print("\n" + "="*70)
    print(f"{asset_name} — JM-only λ Validation Sweep")
    print(f"Paper: S={p['paper_sharpe']:.2f}  MDD={p['paper_mdd']:.2%}  B&H={p['bh_sharpe']:.2f}")
    print("="*70)
    print("\n  [XGB validation — baseline for comparison]")
    run_asset_wf(asset_name, df_feat, cfg_info, GRID_8PT,  '8pt  [xgb-val]')
    run_asset_wf(asset_name, df_feat, cfg_info, LOG19PT,   'Log19 [xgb-val]')

    print("\n  [JM-only validation — hypothesis]")
    run_asset_wf_jmval(asset_name, df_feat, cfg_info, GRID_8PT,  '8pt  [jm-val]')
    run_asset_wf_jmval(asset_name, df_feat, cfg_info, LOG19PT,   'Log19 [jm-val]')
    run_asset_wf_jmval(asset_name, df_feat, cfg_info, LOG25PT,   'Log25 [jm-val]')


def jmval_batch(dfs):
    """Batch 1 + 2: LargeCap + MidCap + EM + AggBond with JM-only validation."""
    for name in ['LargeCap', 'MidCap', 'EM', 'AggBond']:
        if name in dfs:
            jm_val_sweep(name, dfs[name], ASSET_CONFIGS[name])


# ──────────────────────────────────────────────────────────────────────────────
# n_estimators=200 global test
# Hypothesis: n_est=200 amplifies validation Sharpe difference between clean
# labels (high λ) and noisy labels (low λ), making walk-forward more reliable.
# Already confirmed for LargeCap (+0.079, S=0.770). Test across all assets.
# ──────────────────────────────────────────────────────────────────────────────

def run_asset_n200(asset_name, df_feat, cfg_info):
    """Run asset with n_estimators=200. Prints vs n_est=100 baseline."""
    paper = cfg_info
    main.TARGET_TICKER = cfg_info['hl_proxy']
    main.LAMBDA_GRID   = GRID_8PT
    main._forecast_cache.clear()

    from config import _default_xgb_params
    xgb200 = _default_xgb_params()
    xgb200['n_estimators'] = 200

    cfg = StrategyConfig(name=f'BBG_{asset_name}_n200', ewma_mode='paper', xgb_params=xgb200)
    r = main.walk_forward_backtest(df_feat, cfg)

    if r is None or r.empty:
        print(f"  {asset_name:<12}  n_est=200  ERROR")
        return

    s, m = mets(r)
    bp, ns = reg_stats(r)
    lh = np.array(r.attrs.get('lambda_history', [0]))

    gap_s  = s - paper['paper_sharpe']
    print(f"  {asset_name:<12}  n_est=200  S={s:.3f}({gap_s:+.3f})  MDD={m:.1%}  "
          f"Bear={bp:.1f}%  Shft={ns}  λ̄={lh.mean():.1f}")
    # Also print λ trace to diagnose selection quality
    lh_rounded = [round(float(x), 1) for x in lh]
    print(f"    λ trace: {lh_rounded}")


def n200_batch(dfs):
    """
    Test n_est=200 on all assets not yet tested (all except LargeCap/AggBond/REIT).
    Prints both n_est=100 (baseline) and n_est=200 side-by-side for comparison.
    """
    # Assets already tested in Sessions 12-13: LargeCap (+0.079), AggBond (-0.046), REIT (-0.029)
    # Remaining assets to test:
    test_order = ['MidCap', 'SmallCap', 'EAFE', 'EM',
                  'Treasury', 'HighYield', 'Corporate', 'Commodity', 'Gold']
    print("\n" + "="*70)
    print("n_estimators=200 Global Test — all assets vs baseline n_est=100")
    print("Known: LargeCap +0.079, AggBond −0.046, REIT −0.029")
    print("="*70)
    print(f"\n  {'Asset':<12}  {'n_est':>7}  {'Sharpe (Δ)':>14}  {'MDD':>8}  "
          f"{'Bear%':>7}  {'Shft':>6}  {'λ̄':>6}")
    print("  " + "─"*70)
    orig_ticker = main.TARGET_TICKER
    for name in test_order:
        if name in dfs:
            # Baseline n_est=100
            main.TARGET_TICKER = ASSET_CONFIGS[name]['hl_proxy']
            main.LAMBDA_GRID   = GRID_8PT
            main._forecast_cache.clear()
            cfg100 = StrategyConfig(name=f'BBG_{name}', ewma_mode='paper')
            r100 = main.walk_forward_backtest(dfs[name], cfg100)
            if r100 is not None and not r100.empty:
                s100, m100 = mets(r100)
                bp100, ns100 = reg_stats(r100)
                lh100 = np.array(r100.attrs.get('lambda_history', [0]))
                g100 = s100 - ASSET_CONFIGS[name]['paper_sharpe']
                print(f"  {name:<12}  n_est=100  S={s100:.3f}({g100:+.3f})  MDD={m100:.1%}  "
                      f"Bear={bp100:.1f}%  Shft={ns100}  λ̄={lh100.mean():.1f}")
            # n_est=200
            run_asset_n200(name, dfs[name], ASSET_CONFIGS[name])
    main.TARGET_TICKER = orig_ticker


# ──────────────────────────────────────────────────────────────────────────────
# JM-only walk-forward — Table 4 JM row replication
# Uses JM regime labels directly (no XGBoost). Walk-forward λ selection uses
# JM-only validation Sharpe. Compares against paper Table 4 JM column.
# ──────────────────────────────────────────────────────────────────────────────

# Paper Table 4: JM-only Sharpe and MDD
PAPER_JM_SHARPE = {
    'LargeCap': 0.59, 'MidCap': 0.49, 'SmallCap': 0.28, 'EAFE': 0.28,
    'EM': 0.65, 'REIT': 0.39, 'AggBond': 0.43, 'Treasury': 0.21,
    'HighYield': 1.49, 'Corporate': 0.83, 'Commodity': 0.08, 'Gold': 0.12,
}
PAPER_JM_MDD = {
    'LargeCap': -0.2478, 'MidCap': -0.3324, 'SmallCap': -0.3835, 'EAFE': -0.2972,
    'EM': -0.2622, 'REIT': -0.5471, 'AggBond': -0.0609, 'Treasury': -0.2285,
    'HighYield': -0.1388, 'Corporate': -0.0826, 'Commodity': -0.5848, 'Gold': -0.3178,
}


def run_asset_jm_only(asset_name, df_feat, cfg_info, grid=None):
    """Walk-forward JM-only strategy. Validates with JM-only Sharpe, OOS with JM signals."""
    paper = cfg_info
    main.TARGET_TICKER = cfg_info['hl_proxy']
    main.LAMBDA_GRID   = grid if grid is not None else GRID_8PT
    main._forecast_cache.clear()

    cfg = StrategyConfig(name=f'BBG_{asset_name}_jmonly', ewma_mode='paper',
                         include_xgboost=False)
    r = main.walk_forward_backtest(df_feat, cfg)

    if r is None or r.empty:
        print(f"  {asset_name:<12}  JM-only  ERROR")
        return None

    s, m = mets(r)
    bp, ns = reg_stats(r)
    lh = np.array(r.attrs.get('lambda_history', [0]))

    ps = PAPER_JM_SHARPE.get(asset_name, cfg_info['paper_sharpe'])
    pm = PAPER_JM_MDD.get(asset_name, cfg_info['paper_mdd'])
    gap_s = s - ps
    gap_m = m - pm
    bh = bh_sharpe(df_feat)
    print(f"  {asset_name:<12}  JM-only  S={s:.3f}(p:{ps:.2f},{gap_s:+.3f})  "
          f"MDD={m:.1%}(p:{pm:.1%},{gap_m:+.1%})  Bear={bp:.1f}%  Shft={ns}  "
          f"λ̄={lh.mean():.1f}  λs={[round(float(x),1) for x in lh]}")
    return s


def jm_batch(dfs):
    """Run JM-only walk-forward on all 12 Bloomberg assets vs Table 4 JM row."""
    print("\n" + "="*70)
    print("JM-only Walk-Forward — Bloomberg Data vs Paper Table 4 JM Row")
    print("λ selection: JM-only validation Sharpe on 5yr rolling window")
    print("="*70)
    print(f"\n  {'Asset':<12}  {'S (p, Δ)':>26}  {'MDD (p, Δ)':>26}  {'Bear%':>7}  {'Shft':>6}  {'λ̄':>6}")
    print("  " + "─"*90)

    orig_ticker = main.TARGET_TICKER
    wins = 0
    for name in ASSET_CONFIGS:
        if name in dfs:
            result = run_asset_jm_only(name, dfs[name], ASSET_CONFIGS[name])
            if result is not None and result >= PAPER_JM_SHARPE.get(name, 0) - 0.05:
                wins += 1
    main.TARGET_TICKER = orig_ticker
    print(f"\n  → Results vs Paper JM row (Table 4)")


# ──────────────────────────────────────────────────────────────────────────────
# tree_method='exact' global test
# Hypothesis: paper may have used XGBoost 1.x where 'exact' was the default
# tree method (changed to 'hist' in XGBoost 2.0). 'exact' evaluates all split
# points and may produce different validation Sharpe landscapes.
# Session 12 showed: exact+n=100 → LargeCap S=0.713 (+0.022 vs hist 0.691).
# Cross-asset effect unknown — test globally here.
# ──────────────────────────────────────────────────────────────────────────────

def run_asset_exact(asset_name, df_feat, cfg_info):
    """Run asset with tree_method='exact'. Prints vs hist baseline."""
    paper = cfg_info
    main.TARGET_TICKER = cfg_info['hl_proxy']
    main.LAMBDA_GRID   = GRID_8PT
    main._forecast_cache.clear()

    from config import _default_xgb_params
    xgb_exact = _default_xgb_params()
    xgb_exact['tree_method'] = 'exact'

    cfg = StrategyConfig(name=f'BBG_{asset_name}_exact', ewma_mode='paper', xgb_params=xgb_exact)
    r = main.walk_forward_backtest(df_feat, cfg)

    if r is None or r.empty:
        print(f"  {asset_name:<12}  tree=exact  ERROR")
        return

    s, m = mets(r)
    bp, ns = reg_stats(r)
    lh = np.array(r.attrs.get('lambda_history', [0]))

    gap_s = s - paper['paper_sharpe']
    print(f"  {asset_name:<12}  tree=exact  S={s:.3f}({gap_s:+.3f})  MDD={m:.1%}  "
          f"Bear={bp:.1f}%  Shft={ns}  λ̄={lh.mean():.1f}")
    lh_rounded = [round(float(x), 1) for x in lh]
    print(f"    λ trace: {lh_rounded}")


def exact_batch(dfs):
    """
    Test tree_method='exact' on all 12 assets vs hist baseline.
    Session 12 confirmed LargeCap: exact S=0.713 vs hist S=0.691 (+0.022).
    """
    all_assets = list(ASSET_CONFIGS.keys())
    print("\n" + "="*70)
    print("tree_method='exact' Global Test — all 12 assets vs hist baseline")
    print("Known: LargeCap hist=0.691, exact=0.713 (+0.022)")
    print("="*70)

    orig_ticker = main.TARGET_TICKER
    for name in all_assets:
        if name not in dfs:
            continue
        # Baseline hist (n_est=100)
        main.TARGET_TICKER = ASSET_CONFIGS[name]['hl_proxy']
        main.LAMBDA_GRID   = GRID_8PT
        main._forecast_cache.clear()
        cfg_hist = StrategyConfig(name=f'BBG_{name}', ewma_mode='paper')
        r_hist = main.walk_forward_backtest(dfs[name], cfg_hist)
        if r_hist is not None and not r_hist.empty:
            s_h, m_h = mets(r_hist)
            bp_h, ns_h = reg_stats(r_hist)
            lh_h = np.array(r_hist.attrs.get('lambda_history', [0]))
            g_h = s_h - ASSET_CONFIGS[name]['paper_sharpe']
            print(f"  {name:<12}  tree=hist   S={s_h:.3f}({g_h:+.3f})  MDD={m_h:.1%}  "
                  f"Bear={bp_h:.1f}%  Shft={ns_h}  λ̄={lh_h.mean():.1f}")
        # exact
        run_asset_exact(name, dfs[name], ASSET_CONFIGS[name])
    main.TARGET_TICKER = orig_ticker


# ──────────────────────────────────────────────────────────────────────────────
# EWMA adjust=False test
# Hypothesis: paper uses standard recursive EWMA (adjust=False), not pandas
# default (adjust=True). Affects all feature EWMA computations (DD, Avg_Ret,
# Sortino, macro features) and probability smoothing. Global change.
#
# Large λ grid test
# Hypothesis: paper used a much larger log-uniform grid (50-100 pts from 1-100)
# without look-ahead bias in design. Denser coverage in 30-70 range may improve
# walk-forward λ selection even if more low-λ candidates exist.
# ──────────────────────────────────────────────────────────────────────────────

LOG50PT  = list(np.logspace(0, 2, 50))    # 50 pts: 1.0 → 100.0
LOG100PT = list(np.logspace(0, 2, 100))   # 100 pts: 1.0 → 100.0


def run_asset_adjust(asset_name, df_feat_adj, cfg_info, grid, label, ewma_adjust=False):
    """Walk-forward with adjust=False features and prob smoothing."""
    main.TARGET_TICKER = cfg_info['hl_proxy']
    main.LAMBDA_GRID   = grid
    main._forecast_cache.clear()

    cfg = StrategyConfig(name=f'BBG_{asset_name}_adj{label}', ewma_mode='paper',
                         ewma_adjust=ewma_adjust)
    r = main.walk_forward_backtest(df_feat_adj, cfg)

    if r is None or r.empty:
        print(f"  {asset_name:<12}  [{label}]  ERROR")
        return None

    s, m = mets(r)
    bp, ns = reg_stats(r)
    lh = np.array(r.attrs.get('lambda_history', [0]))
    gap_s = s - cfg_info['paper_sharpe']
    print(f"  {asset_name:<12}  {label:<30}  S={s:.3f}({gap_s:+.3f})  MDD={m:.1%}  "
          f"Bear={bp:.1f}%  Shft={ns}  λ̄={lh.mean():.1f}  λs={[round(float(x),1) for x in lh]}")
    return s


def adjust_false_sweep(asset_name, df_feat, cfg_info, raw, fred, vix, irx):
    """
    Test adjust=False on one asset across multiple grid sizes.
    Also tests large grids (50pt, 100pt) on both adjust=True and adjust=False.
    """
    p = cfg_info
    print("\n" + "="*70)
    print(f"{asset_name} — adjust=False + Large Grid Sweep")
    print(f"Paper JM-XGB: S={p['paper_sharpe']:.2f}  MDD={p['paper_mdd']:.2%}  B&H={p['bh_sharpe']:.2f}")
    print("="*70)

    df_adj = build_features(raw, fred, vix, irx, cfg_info['col'], cfg_info['dd'],
                            ewm_adjust=False)
    print(f"  Features built (adjust=False): {df_adj.index.min().date()} → {df_adj.index.max().date()}")

    print("\n  [adjust=True baseline]")
    run_asset_adjust(asset_name, df_feat, cfg_info, GRID_8PT,   'adj=True  8pt',   ewma_adjust=True)
    run_asset_adjust(asset_name, df_feat, cfg_info, LOG50PT,    'adj=True  50pt',  ewma_adjust=True)
    run_asset_adjust(asset_name, df_feat, cfg_info, LOG100PT,   'adj=True  100pt', ewma_adjust=True)

    print("\n  [adjust=False — hypothesis]")
    run_asset_adjust(asset_name, df_adj,  cfg_info, GRID_8PT,   'adj=False 8pt',   ewma_adjust=False)
    run_asset_adjust(asset_name, df_adj,  cfg_info, LOG50PT,    'adj=False 50pt',  ewma_adjust=False)
    run_asset_adjust(asset_name, df_adj,  cfg_info, LOG100PT,   'adj=False 100pt', ewma_adjust=False)


def adjust_false_batch(dfs, raw, fred, vix, irx):
    """Run adjust=False sweep on all 12 assets (adjust=True vs False, 8pt vs 50pt vs 100pt)."""
    print("\n" + "="*70)
    print("EWMA adjust=False + Large Grid — All 12 Assets")
    print("="*70)
    orig_ticker = main.TARGET_TICKER
    for name in ASSET_CONFIGS:
        if name not in dfs:
            continue
        cfg_info = ASSET_CONFIGS[name]
        df_adj = build_features(raw, fred, vix, irx, cfg_info['col'], cfg_info['dd'],
                                ewm_adjust=False)
        print(f"\n  {name} (paper S={cfg_info['paper_sharpe']:.2f})")
        run_asset_adjust(name, dfs[name], cfg_info, GRID_8PT,  'adj=True  8pt',   ewma_adjust=True)
        run_asset_adjust(name, df_adj,    cfg_info, GRID_8PT,  'adj=False 8pt',   ewma_adjust=False)
        run_asset_adjust(name, dfs[name], cfg_info, LOG50PT,   'adj=True  50pt',  ewma_adjust=True)
        run_asset_adjust(name, df_adj,    cfg_info, LOG50PT,   'adj=False 50pt',  ewma_adjust=False)
        run_asset_adjust(name, dfs[name], cfg_info, LOG100PT,  'adj=True  100pt', ewma_adjust=True)
        run_asset_adjust(name, df_adj,    cfg_info, LOG100PT,  'adj=False 100pt', ewma_adjust=False)
    main.TARGET_TICKER = orig_ticker


# ──────────────────────────────────────────────────────────────────────────────
# Shared λ hypothesis — does paper's Table 4 JM row use XGB-selected λ?
# Method: run XGB WF first, extract λ trace, then replay JM-only OOS with
# the same λ sequence (skip validation entirely — very fast).
# If this matches paper's JM Sharpe, the paper shares λ between rows.
# ──────────────────────────────────────────────────────────────────────────────

def run_jm_shared_lambda(asset_name, df_feat, cfg_info):
    """Run XGB WF, extract λ trace, replay JM-only OOS with same λs. Fast."""
    main.TARGET_TICKER = cfg_info['hl_proxy']
    main.LAMBDA_GRID   = GRID_8PT
    main._forecast_cache.clear()

    # Step 1: XGB walk-forward (standard)
    cfg_xgb = StrategyConfig(name=f'BBG_{asset_name}', ewma_mode='paper')
    r_xgb = main.walk_forward_backtest(df_feat, cfg_xgb)
    if r_xgb is None or r_xgb.empty:
        print(f"  {asset_name:<12}  XGB WF ERROR")
        return

    lambda_trace = r_xgb.attrs.get('lambda_history', [])
    s_xgb, m_xgb = mets(r_xgb)

    # Step 2: JM-only replay with same λ trace (no validation — just OOS)
    main._forecast_cache.clear()
    cfg_jm = StrategyConfig(name=f'BBG_{asset_name}_jmshared', ewma_mode='paper',
                            include_xgboost=False)
    r_jm = main.walk_forward_backtest(df_feat, cfg_jm, fixed_lambda_sequence=lambda_trace)
    if r_jm is None or r_jm.empty:
        print(f"  {asset_name:<12}  JM shared λ ERROR")
        return

    s_jm, m_jm = mets(r_jm)
    bp_jm, ns_jm = reg_stats(r_jm)
    pj = PAPER_JM_SHARPE.get(asset_name, cfg_info['paper_sharpe'])
    px = cfg_info['paper_sharpe']

    print(f"  {asset_name:<12}  "
          f"XGB  S={s_xgb:.3f}(p:{px:.2f},{s_xgb-px:+.3f})  "
          f"| JM-shared-λ  S={s_jm:.3f}(p:{pj:.2f},{s_jm-pj:+.3f})  "
          f"MDD={m_jm:.1%}  Bear={bp_jm:.1f}%  Shft={ns_jm}  "
          f"λ̄={np.mean(lambda_trace):.1f}")


def jm_shared_lambda_batch(dfs):
    """Shared λ hypothesis: all 12 assets. Fast since JM-only replay skips validation."""
    print("\n" + "="*70)
    print("Shared λ Hypothesis — Paper JM uses XGB-selected λ?")
    print("Method: Run XGB WF → apply JM-only signals with same λ trace")
    print(f"Paper JM targets: LC=0.59, MC=0.49, SC=0.28, EAFE=0.28, EM=0.65,")
    print(f"                  REIT=0.39, AB=0.43, TR=0.21, HY=1.49, Corp=0.83, Comm=0.08, Gold=0.12")
    print("="*70)
    orig_ticker = main.TARGET_TICKER
    for name in ASSET_CONFIGS:
        if name in dfs:
            run_jm_shared_lambda(name, dfs[name], ASSET_CONFIGS[name])
    main.TARGET_TICKER = orig_ticker


# ──────────────────────────────────────────────────────────────────────────────
# DD formula variants — tests whether paper uses log(DD) vs raw DD for JM features
# and whether Sortino clipping value differs.
# Current: DD_log_hl = log(sqrt(EWM(dn^2,hl))+1e-8), Sortino clipped ±10
# Variant: DD_hl = sqrt(EWM(dn^2,hl)) (raw), or different clip value
# ──────────────────────────────────────────────────────────────────────────────

def dd_formula_sweep(asset_name, df_feat, cfg_info, raw, fred, vix, irx):
    """Test DD formula variants on one asset. LargeCap first for fast feedback."""
    p = cfg_info
    print("\n" + "="*70)
    print(f"{asset_name} — DD Formula + Sortino Clip Variants")
    print(f"Paper JM-XGB: S={p['paper_sharpe']:.2f}  Paper JM: S={PAPER_JM_SHARPE.get(asset_name, '?')}")
    print("="*70)

    variants = [
        ('log', 10,   'log-DD  clip±10  (current baseline)'),
        ('raw', 10,   'raw-DD  clip±10'),
        ('log', 3,    'log-DD  clip±3'),
        ('raw', 3,    'raw-DD  clip±3'),
        ('log', None, 'log-DD  no-clip'),
        ('raw', None, 'raw-DD  no-clip'),
    ]

    orig_ticker = main.TARGET_TICKER
    main.TARGET_TICKER = cfg_info['hl_proxy']
    main.LAMBDA_GRID   = GRID_8PT

    for dd_f, s_clip, label in variants:
        df_v = build_features(raw, fred, vix, irx, cfg_info['col'], cfg_info['dd'],
                              dd_formula=dd_f, sortino_clip=s_clip)
        main._forecast_cache.clear()
        cfg = StrategyConfig(name=f'BBG_{asset_name}_dd{dd_f}c{s_clip}', ewma_mode='paper')
        r = main.walk_forward_backtest(df_v, cfg)
        if r is None or r.empty:
            print(f"  {label:<38}  ERROR")
            continue
        s, m = mets(r)
        bp, ns = reg_stats(r)
        lh = np.array(r.attrs.get('lambda_history', [0]))
        gap = s - p['paper_sharpe']
        print(f"  {label:<38}  S={s:.3f}({gap:+.3f})  MDD={m:.1%}  Bear={bp:.1f}%  λ̄={lh.mean():.1f}")

    main.TARGET_TICKER = orig_ticker


def dd_formula_batch(dfs, raw, fred, vix, irx):
    """Test 2 key DD variants (log vs raw) on all 12 assets."""
    print("\n" + "="*70)
    print("DD Formula Batch — log-DD vs raw-DD, all 12 assets")
    print("="*70)
    orig_ticker = main.TARGET_TICKER
    for name in ASSET_CONFIGS:
        if name not in dfs:
            continue
        cfg_info = ASSET_CONFIGS[name]
        main.TARGET_TICKER = cfg_info['hl_proxy']
        main.LAMBDA_GRID   = GRID_8PT
        p = cfg_info['paper_sharpe']

        # Baseline (log DD, clip±10) — use existing df_feat
        main._forecast_cache.clear()
        cfg_log = StrategyConfig(name=f'BBG_{name}', ewma_mode='paper')
        r_log = main.walk_forward_backtest(dfs[name], cfg_log)

        # Raw DD (no log, clip±10)
        df_raw = build_features(raw, fred, vix, irx, cfg_info['col'], cfg_info['dd'],
                                dd_formula='raw')
        main._forecast_cache.clear()
        cfg_raw = StrategyConfig(name=f'BBG_{name}_rawdd', ewma_mode='paper')
        r_raw = main.walk_forward_backtest(df_raw, cfg_raw)

        if r_log is not None and not r_log.empty:
            s_l, _ = mets(r_log)
            lh_l = np.array(r_log.attrs.get('lambda_history', [0]))
            print(f"  {name:<12}  log-DD  S={s_l:.3f}({s_l-p:+.3f})  λ̄={lh_l.mean():.1f}")
        if r_raw is not None and not r_raw.empty:
            s_r, _ = mets(r_raw)
            lh_r = np.array(r_raw.attrs.get('lambda_history', [0]))
            print(f"  {name:<12}  raw-DD  S={s_r:.3f}({s_r-p:+.3f})  λ̄={lh_r.mean():.1f}")

    main.TARGET_TICKER = orig_ticker


# ──────────────────────────────────────────────────────────────────────────────
# Session 20: JM-only + TC=0 — combine S18 (shared-λ) + S19 (TC=0)
# Hypothesis: Paper Table 4 JM row = JM signals + XGB-selected λ + TC=0
# The paper is gross-of-TC (S19) and reuses XGB λ for JM row (S18).
# This is the first time both conditions are tested simultaneously.
# ──────────────────────────────────────────────────────────────────────────────

def jm_tc0_comparison(asset_name, df_feat, cfg_info):
    """JM-only TC=0 comprehensive comparison vs paper Table 4 JM row.

    Runs 4 conditions per asset:
      TC=5bps + shared-λ  (S18 baseline — verify consistency)
      TC=0    + shared-λ  (KEY: paper-matching condition)
      TC=5bps + indep WF  (S17 independent JM WF — verify)
      TC=0    + indep WF  (for assets where shared-λ fails: MidCap, SmallCap, EM...)
    """
    main.TARGET_TICKER = cfg_info['hl_proxy']
    main.LAMBDA_GRID = GRID_8PT
    orig_tc = main.TRANSACTION_COST

    pj = PAPER_JM_SHARPE.get(asset_name, 0)
    pm = PAPER_JM_MDD.get(asset_name, 0)

    print(f"\n  {asset_name:<12}  Paper JM: S={pj:.2f}  MDD={pm:.1%}")

    for tc_val, tc_label in [(0.0005, 'TC=5bps'), (0.0, 'TC=0  ')]:
        main.TRANSACTION_COST = tc_val

        # XGB WF → extract λ trace (TC affects validation → λ trace changes with TC)
        main._forecast_cache.clear()
        cfg_xgb = StrategyConfig(name=f'BBG_{asset_name}', ewma_mode='paper')
        r_xgb = main.walk_forward_backtest(df_feat, cfg_xgb)
        if r_xgb is None or r_xgb.empty:
            print(f"    {tc_label}  XGB WF ERROR")
            continue
        lambda_trace = r_xgb.attrs.get('lambda_history', [])
        lm_xgb = np.mean(lambda_trace) if lambda_trace else 0

        # Condition A: shared-λ (JM replays same XGB λ trace, no validation)
        if lambda_trace:
            main._forecast_cache.clear()
            cfg_jm_sh = StrategyConfig(name=f'BBG_{asset_name}_jmsh', ewma_mode='paper',
                                       include_xgboost=False)
            r_jm_sh = main.walk_forward_backtest(df_feat, cfg_jm_sh,
                                                  fixed_lambda_sequence=lambda_trace)
            if r_jm_sh is not None and not r_jm_sh.empty:
                s_jmsh, m_jmsh = mets(r_jm_sh)
                bp_jmsh, ns_jmsh = reg_stats(r_jm_sh)
                gap_sh = s_jmsh - pj
                print(f"    {tc_label}  shared-λ  (λ̄={lm_xgb:.1f})  "
                      f"S={s_jmsh:.3f}({gap_sh:+.3f})  MDD={m_jmsh:.1%}  "
                      f"Bear={bp_jmsh:.1f}%  Shft={ns_jmsh}")

        # Condition B: independent JM WF (JM-only validation, no XGB)
        main._forecast_cache.clear()
        cfg_jm_wf = StrategyConfig(name=f'BBG_{asset_name}_jmwf', ewma_mode='paper',
                                   include_xgboost=False)
        r_jm_wf = main.walk_forward_backtest(df_feat, cfg_jm_wf)
        if r_jm_wf is not None and not r_jm_wf.empty:
            s_jmwf, m_jmwf = mets(r_jm_wf)
            bp_jmwf, ns_jmwf = reg_stats(r_jm_wf)
            lh_wf = np.array(r_jm_wf.attrs.get('lambda_history', [0]))
            gap_wf = s_jmwf - pj
            print(f"    {tc_label}  indep-WF  (λ̄={lh_wf.mean():.1f})  "
                  f"S={s_jmwf:.3f}({gap_wf:+.3f})  MDD={m_jmwf:.1%}  "
                  f"Bear={bp_jmwf:.1f}%  Shft={ns_jmwf}")

    main.TRANSACTION_COST = orig_tc


def jm_tc0_batch(dfs):
    """Run JM_TC0 comparison on all 12 Bloomberg assets."""
    print("\n" + "="*70)
    print("JM-only + TC=0 — Session 20 Key Experiment")
    print("Hypothesis: Table 4 JM row = JM signals + XGB-selected λ + TC=0 (gross)")
    print(f"Paper JM:  LC=0.59  MC=0.49  SC=0.28  EAFE=0.28  EM=0.65")
    print(f"           REIT=0.39  AB=0.43  TR=0.21  HY=1.49  Corp=0.83  Comm=0.08  Gold=0.12")
    print("="*70)
    orig_ticker = main.TARGET_TICKER
    for name in ASSET_CONFIGS:
        if name in dfs:
            jm_tc0_comparison(name, dfs[name], ASSET_CONFIGS[name])
    main.TARGET_TICKER = orig_ticker
    # Restore TC
    main.TRANSACTION_COST = 0.0005


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = [a.upper() for a in sys.argv[1:]] if len(sys.argv) > 1 else list(ASSET_CONFIGS.keys())

    # Expand ALL → all assets (uppercase keys to match key.upper() comparison below)
    if 'ALL' in args:
        args = [k.upper() for k in ASSET_CONFIGS.keys()] + ['REIT_ORACLE']

    # JM-only modes: JM_BATCH, or <ASSET>_JMONLY for a single asset
    JMONLY_SINGLE = {k.upper() + '_JMONLY': k for k in ASSET_CONFIGS}
    # Shared λ modes: JMSHARED_BATCH, or <ASSET>_JMSHARED
    JMSHARED_SINGLE = {k.upper() + '_JMSHARED': k for k in ASSET_CONFIGS}
    # JM + TC=0 combined modes: JM_TC0_BATCH, or <ASSET>_JM_TC0
    JM_TC0_SINGLE = {k.upper() + '_JM_TC0': k for k in ASSET_CONFIGS}
    # DD formula modes: DD_BATCH (all assets, log vs raw), or <ASSET>_DD (single asset full sweep)
    DD_SINGLE = {k.upper() + '_DD': k for k in ASSET_CONFIGS}
    # JM-val modes: JMVAL_BATCH, or <ASSET>_JMVAL for a single asset
    JMVAL_SINGLE = {k.upper() + '_JMVAL': k for k in ASSET_CONFIGS}
    # n200 modes: N200_BATCH, or <ASSET>_N200 for a single asset
    N200_SINGLE  = {k.upper() + '_N200':  k for k in ASSET_CONFIGS}
    # exact modes: EXACT_BATCH, or <ASSET>_EXACT for a single asset
    EXACT_SINGLE = {k.upper() + '_EXACT': k for k in ASSET_CONFIGS}

    # Detect oracle/grid modes and map to asset keys
    ORACLE_MODES = {k + '_ORACLE': k for k in ASSET_CONFIGS}
    ORACLE_MODES_UPPER = {k.upper(): v for k, v in ORACLE_MODES.items()}
    GRID_MODES = {'MIDCAP_GRID': 'MidCap', 'EM_GRID': 'EM'}

    raw, fred, vix, irx = load_bbg_raw()

    # Determine which assets need feature DataFrames
    jmonly_assets = set()
    if 'JM_BATCH' in args:
        jmonly_assets = set(ASSET_CONFIGS.keys())
    for arg in args:
        if arg in JMONLY_SINGLE:
            jmonly_assets.add(JMONLY_SINGLE[arg])

    jmshared_assets = set()
    if 'JMSHARED_BATCH' in args:
        jmshared_assets = set(ASSET_CONFIGS.keys())
    for arg in args:
        if arg in JMSHARED_SINGLE:
            jmshared_assets.add(JMSHARED_SINGLE[arg])

    jm_tc0_assets = set()
    if 'JM_TC0_BATCH' in args:
        jm_tc0_assets = set(ASSET_CONFIGS.keys())
    for arg in args:
        if arg in JM_TC0_SINGLE:
            jm_tc0_assets.add(JM_TC0_SINGLE[arg])

    dd_assets = set()
    if 'DD_BATCH' in args:
        dd_assets = set(ASSET_CONFIGS.keys())
    for arg in args:
        if arg in DD_SINGLE:
            dd_assets.add(DD_SINGLE[arg])

    jmval_assets = set()
    if 'JMVAL_BATCH' in args:
        jmval_assets = {'LargeCap', 'MidCap', 'EM', 'AggBond'}
    for arg in args:
        if arg in JMVAL_SINGLE:
            jmval_assets.add(JMVAL_SINGLE[arg])

    n200_assets = set()
    if 'N200_BATCH' in args:
        n200_assets = {'MidCap', 'SmallCap', 'EAFE', 'EM',
                       'Treasury', 'HighYield', 'Corporate', 'Commodity', 'Gold'}
    for arg in args:
        if arg in N200_SINGLE:
            n200_assets.add(N200_SINGLE[arg])

    exact_assets = set()
    if 'EXACT_BATCH' in args:
        exact_assets = set(ASSET_CONFIGS.keys())
    for arg in args:
        if arg in EXACT_SINGLE:
            exact_assets.add(EXACT_SINGLE[arg])

    # adjust=False + large grid modes: ADJUST_BATCH, or <ASSET>_ADJUST for single asset
    ADJUST_SINGLE = {k.upper() + '_ADJUST': k for k in ASSET_CONFIGS}
    adjust_assets = set()
    if 'ADJUST_BATCH' in args:
        adjust_assets = set(ASSET_CONFIGS.keys())
    for arg in args:
        if arg in ADJUST_SINGLE:
            adjust_assets.add(ADJUST_SINGLE[arg])

    # Pre-build feature DataFrames for requested assets
    dfs = {}
    for key, cfg in ASSET_CONFIGS.items():
        need_baseline = key.upper() in args
        need_oracle   = any(args_k == (key + '_ORACLE').upper() for args_k in args)
        need_grid     = any(args_k == (key.upper() + '_GRID') for args_k in args)
        need_jmonly   = key in jmonly_assets
        need_jmshared = key in jmshared_assets
        need_dd       = key in dd_assets
        need_jmval    = key in jmval_assets
        need_n200     = key in n200_assets
        need_exact    = key in exact_assets
        need_adjust   = key in adjust_assets
        need_jm_tc0   = key in jm_tc0_assets
        if need_baseline or need_oracle or need_grid or need_jmonly or need_jmshared or need_dd or need_jmval or need_n200 or need_exact or need_adjust or need_jm_tc0:
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

    # Run JM-only strategy (Table 4 JM row replication)
    if 'JM_BATCH' in args:
        jm_batch(dfs)
    else:
        for arg in args:
            asset_key = JMONLY_SINGLE.get(arg)
            if asset_key and asset_key in dfs:
                run_asset_jm_only(asset_key, dfs[asset_key], ASSET_CONFIGS[asset_key])

    # Run shared λ hypothesis tests
    if 'JMSHARED_BATCH' in args:
        jm_shared_lambda_batch(dfs)
    else:
        for arg in args:
            asset_key = JMSHARED_SINGLE.get(arg)
            if asset_key and asset_key in dfs:
                run_jm_shared_lambda(asset_key, dfs[asset_key], ASSET_CONFIGS[asset_key])

    # Run JM-only + TC=0 combined test (Session 20 key experiment)
    if 'JM_TC0_BATCH' in args:
        jm_tc0_batch(dfs)
    else:
        for arg in args:
            asset_key = JM_TC0_SINGLE.get(arg)
            if asset_key and asset_key in dfs:
                jm_tc0_comparison(asset_key, dfs[asset_key], ASSET_CONFIGS[asset_key])
                main.TRANSACTION_COST = 0.0005  # restore after single-asset run

    # Run DD formula tests
    if 'DD_BATCH' in args:
        dd_formula_batch(dfs, raw, fred, vix, irx)
    else:
        for arg in args:
            asset_key = DD_SINGLE.get(arg)
            if asset_key and asset_key in dfs:
                dd_formula_sweep(asset_key, dfs[asset_key], ASSET_CONFIGS[asset_key],
                                 raw, fred, vix, irx)

    # Run JM-only validation sweeps
    if 'JMVAL_BATCH' in args:
        jmval_batch(dfs)
    else:
        for arg in args:
            asset_key = JMVAL_SINGLE.get(arg)
            if asset_key and asset_key in dfs:
                jm_val_sweep(asset_key, dfs[asset_key], ASSET_CONFIGS[asset_key])

    # Run n_est=200 tests
    if 'N200_BATCH' in args:
        n200_batch(dfs)
    else:
        for arg in args:
            asset_key = N200_SINGLE.get(arg)
            if asset_key and asset_key in dfs:
                run_asset_n200(asset_key, dfs[asset_key], ASSET_CONFIGS[asset_key])

    # Run tree_method='exact' tests
    if 'EXACT_BATCH' in args:
        exact_batch(dfs)
    else:
        for arg in args:
            asset_key = EXACT_SINGLE.get(arg)
            if asset_key and asset_key in dfs:
                run_asset_exact(asset_key, dfs[asset_key], ASSET_CONFIGS[asset_key])

    # Run adjust=False + large grid tests
    if 'ADJUST_BATCH' in args:
        adjust_false_batch(dfs, raw, fred, vix, irx)
    else:
        for arg in args:
            asset_key = ADJUST_SINGLE.get(arg)
            if asset_key and asset_key in dfs:
                adjust_false_sweep(asset_key, dfs[asset_key], ASSET_CONFIGS[asset_key],
                                   raw, fred, vix, irx)

    print("\n" + "="*70)
    print("DONE")
    print("="*70)
