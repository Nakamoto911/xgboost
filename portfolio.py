"""
Portfolio construction module — reproduces Tables 6, 7 and Figure 3 from
Shu, Yu & Mulvey (2024) "Dynamic Asset Allocation with Asset-Specific Regime
Forecasts" (arXiv:2406.09578v2).

Builds 12-asset signal panels from the JM-XGB pipeline and runs MVO portfolios
under several specifications:

    60/40, MinVar, MinVar(JM-XGB), MV, MV(JM-XGB), EW, EW(JM-XGB)

Heavy work (per-asset walk-forward signals) is cached on disk; cheap work
(MVO sweep across 7 strategies) is recomputed each call.
"""
from __future__ import annotations

import os
import hashlib
import pickle
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Asset universes
# ──────────────────────────────────────────────────────────────────────────────

# Bloomberg (DATA PAUL.xlsx). Order matches paper Tables 4/6/7.
BBG_COLUMNS = ['SPTR', 'SPTRMDCP', 'RU20INTR', 'NDDUEAFE', 'NDUEEGF',
               'LBUSTRUU', 'IBOXHY', 'LUACTRUU', 'DJUSRET',
               'DBLCDBCE', 'GOLDLNPM', 'LUTLTRUU']

# (asset_name, bbg_col, hl_proxy_ticker_for_PAPER_EWMA_HL_lookup, include_dd_features)
BBG_ASSETS: List[Tuple[str, str, str, bool]] = [
    ('LargeCap',  'SPTR',     '^SP500TR', True),
    ('MidCap',    'SPTRMDCP', '^SP500TR', True),
    ('SmallCap',  'RU20INTR', '^SP500TR', True),
    ('EAFE',      'NDDUEAFE', 'EFA',      True),
    ('EM',        'NDUEEGF',  'EEM',      True),
    ('REIT',      'DJUSRET',  '^SP500TR', True),
    ('AggBond',   'LBUSTRUU', '^SP500TR', False),
    ('Treasury',  'LUTLTRUU', '^SP500TR', False),
    ('HighYield', 'IBOXHY',   'HYG',      True),
    ('Corporate', 'LUACTRUU', 'SPBO',     False),
    ('Commodity', 'DBLCDBCE', 'DBC',      True),
    ('Gold',      'GOLDLNPM', 'GLD',      False),
]

# Yahoo ETFs — paper Table 1 column "ETF". hl_proxy follows main.PAPER_EWMA_HL.
YAHOO_ASSETS: List[Tuple[str, str, str, bool]] = [
    ('LargeCap',  'IVV',  'IVV',  True),
    ('MidCap',    'IJH',  'IJH',  True),
    ('SmallCap',  'IWM',  'IWM',  True),
    ('EAFE',      'EFA',  'EFA',  True),
    ('EM',        'EEM',  'EEM',  True),
    ('REIT',      'IYR',  'IYR',  True),
    ('AggBond',   'AGG',  'AGG',  False),
    ('Treasury',  'SPTL', 'SPTL', False),
    ('HighYield', 'HYG',  'HYG',  True),
    ('Corporate', 'SPBO', 'SPBO', False),
    ('Commodity', 'DBC',  'DBC',  True),
    ('Gold',      'GLD',  'GLD',  False),
]

# Yahoo Mutual Funds — long-history proxies optimised for paper replication.
# ^SPGSCI replaces delisted PCASX/PCRAX (0.928 daily-return corr with paper's DBLCDBCE).
# GC=F and VBMFX/VUSTX excluded from DD features (same rule as benchmark_assets.py).
MUTUAL_FUNDS_ASSETS: List[Tuple[str, str, str, bool]] = [
    ('LargeCap',  '^SP500TR', '^SP500TR', True),
    ('MidCap',    'VIMSX',   'VIMSX',   True),
    ('SmallCap',  'NAESX',   'NAESX',   True),
    ('EAFE',      'FDIVX',   'FDIVX',   True),
    ('EM',        'VEIEX',   'VEIEX',   True),
    ('REIT',      'FRESX',   'FRESX',   True),
    ('AggBond',   'VBMFX',   'VBMFX',   False),
    ('Treasury',  'VUSTX',   'VUSTX',   False),
    ('HighYield', 'VWEHX',   'VWEHX',   True),
    ('Corporate', 'VWESX',   'VWESX',   True),
    ('Commodity', '^SPGSCI', '^SPGSCI', True),
    ('Gold',      'GC=F',    'GC=F',    False),
]

# BBG+Yahoo ETF Hybrid — BBG pre-ETF inception spliced with Yahoo ETF post-inception (return-space).
# hl_proxy = the ETF ticker so PAPER_EWMA_HL lookup matches each asset class.
# Spec format: (asset_name, bbg_col, etf_ticker, hl_proxy, include_dd)
HYBRID_ASSETS: List[Tuple[str, str, str, str, bool]] = [
    ('LargeCap',  'SPTR',     'IVV',  'IVV',  True),
    ('MidCap',    'SPTRMDCP', 'IJH',  'IJH',  True),
    ('SmallCap',  'RU20INTR', 'IWM',  'IWM',  True),
    ('EAFE',      'NDDUEAFE', 'EFA',  'EFA',  True),
    ('EM',        'NDUEEGF',  'EEM',  'EEM',  True),
    ('REIT',      'DJUSRET',  'IYR',  'IYR',  True),
    ('AggBond',   'LBUSTRUU', 'AGG',  'AGG',  False),
    ('Treasury',  'LUTLTRUU', 'SPTL', 'SPTL', False),
    ('HighYield', 'IBOXHY',   'HYG',  'HYG',  True),
    ('Corporate', 'LUACTRUU', 'SPBO', 'SPBO', False),
    ('Commodity', 'DBLCDBCE', 'DBC',  'DBC',  True),
    ('Gold',      'GOLDLNPM', 'GLD',  'GLD',  False),
]

PAPER_60_40 = {
    'LargeCap': 0.10, 'MidCap': 0.05, 'SmallCap': 0.05, 'EAFE': 0.05, 'EM': 0.05,
    'REIT': 0.10, 'HighYield': 0.10, 'Commodity': 0.05, 'Gold': 0.05,
    'Treasury': 0.10, 'Corporate': 0.10, 'AggBond': 0.20,
}

GRID_8PT = [4.64, 10.0, 15.0, 21.54, 30.0, 46.42, 70.0, 100.0]

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Bloomberg data loading + feature engineering
# ──────────────────────────────────────────────────────────────────────────────

_bbg_raw_cache: dict = {}


def _load_bbg_raw():
    """Load Bloomberg prices + FRED yields + ^VIX + ^IRX (Yahoo) once per process."""
    if _bbg_raw_cache:
        return _bbg_raw_cache['raw'], _bbg_raw_cache['fred'], _bbg_raw_cache['vix'], _bbg_raw_cache['irx']

    import yfinance as yf
    import main as _main

    xlsx = os.path.join(CACHE_DIR, 'DATA PAUL.xlsx')
    raw = pd.read_excel(xlsx, header=None, skiprows=6)
    raw.columns = ['Date'] + BBG_COLUMNS
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw = raw.set_index('Date').sort_index()

    fred = _main._fetch_fred_data().ffill().dropna()

    def _series(d):
        return (d['Adj Close'].iloc[:, 0] if isinstance(d.columns, pd.MultiIndex)
                else d.get('Adj Close', d['Close']))

    vix_cache = os.path.join(CACHE_DIR, 'portfolio_vix_irx.pkl')
    if os.path.exists(vix_cache):
        cached = pd.read_pickle(vix_cache)
        if cached.index.max() >= pd.Timestamp('2024-01-01'):
            vix, irx = cached['VIX'], cached['IRX']
        else:
            cached = None
    else:
        cached = None

    if cached is None:
        vix = _series(yf.download('^VIX', start='1987-01-01', end='2026-06-01',
                                  auto_adjust=False, progress=False)).rename('VIX')
        irx = _series(yf.download('^IRX', start='1987-01-01', end='2026-06-01',
                                  auto_adjust=False, progress=False)).rename('IRX')
        pd.DataFrame({'VIX': vix, 'IRX': irx}).to_pickle(vix_cache)

    _bbg_raw_cache.update(raw=raw, fred=fred, vix=vix, irx=irx)
    return raw, fred, vix, irx


def _build_bbg_features(target_col: str, include_dd: bool = True) -> pd.DataFrame:
    """Build the JM/XGB feature panel for one Bloomberg asset.

    Mirrors the implementation in misc_scripts/test_bbg_assets.py:build_features.
    """
    raw, fred, vix, irx = _load_bbg_raw()
    cols = list(dict.fromkeys([target_col, 'SPTR', 'LBUSTRUU']))
    df = (raw[cols].join(fred, how='inner').join(vix, how='inner')
          .join(irx, how='inner').ffill().dropna())

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
    f['VIX_EWMA_log_diff'] = (np.log(df['VIX'] / df['VIX'].shift(1)).fillna(0)
                              .ewm(halflife=63).mean())
    sptr_ret = df['SPTR'].pct_change().fillna(0)
    bond_ret = df['LBUSTRUU'].pct_change().fillna(0)
    f['Stock_Bond_Corr'] = sptr_ret.rolling(252).corr(bond_ret).fillna(0)
    return f.dropna()


def _build_yahoo_features(ticker: str, start_date: str, end_date: str,
                          include_dd: bool) -> pd.DataFrame:
    """Build a feature panel for one Yahoo ETF using main.fetch_and_prepare_data.

    Each call mutates main module-level constants (TARGET_TICKER, dates) and then
    returns the resulting feature DataFrame. The on-disk cache is per-ticker and
    auto-refreshes when stale.
    """
    import main as _main
    _main.TARGET_TICKER = ticker
    _main.START_DATE_DATA = start_date
    _main.END_DATE = end_date
    if not include_dd:
        _main.DD_EXCLUDE_TICKERS = set(_main.DD_EXCLUDE_TICKERS) | {ticker}
    df = _main.fetch_and_prepare_data()
    return df


def _build_hybrid_price_series(bbg_col: str, etf_ticker: str,
                               start_date: str, end_date: str) -> pd.Series:
    """Return a continuous synthetic price: BBG returns up to ETF inception
    (inclusive), then ETF returns afterwards. The result is rebased to BBG's
    first level so downstream feature engineering sees a single series."""
    import yfinance as yf
    raw, _, _, _ = _load_bbg_raw()
    bbg_series = raw[bbg_col].dropna()
    bbg_series = bbg_series[bbg_series.index >= pd.Timestamp(start_date)]

    fetch_end = (pd.Timestamp(end_date) + pd.Timedelta(days=2)).strftime('%Y-%m-%d')
    etf_df = yf.download(etf_ticker, start=start_date, end=fetch_end,
                         auto_adjust=False, progress=False)
    if etf_df.empty:
        raise ValueError(f'hybrid: Yahoo returned no data for {etf_ticker}')
    if isinstance(etf_df.columns, pd.MultiIndex):
        if 'Adj Close' in etf_df.columns.get_level_values(0):
            etf_px = etf_df['Adj Close'].iloc[:, 0]
        else:
            etf_px = etf_df.iloc[:, 0]
    else:
        etf_px = etf_df.get('Adj Close', etf_df.get('Close', etf_df.iloc[:, 0]))
    etf_px = etf_px.dropna().sort_index()

    splice_date = etf_px.index.min()
    bbg_rets = bbg_series.pct_change()
    etf_rets = etf_px.pct_change()
    combined = pd.concat([bbg_rets[bbg_rets.index <= splice_date],
                          etf_rets[etf_rets.index > splice_date]]).sort_index()
    combined = combined[~combined.index.duplicated(keep='last')]

    base = float(bbg_series.iloc[0])
    price = base * (1.0 + combined.fillna(0.0)).cumprod()
    price.name = f'{bbg_col}+{etf_ticker}'
    return price


def _build_hybrid_features(bbg_col: str, etf_ticker: str,
                           start_date: str, end_date: str,
                           include_dd: bool = True) -> pd.DataFrame:
    """Feature panel for a hybrid asset. Reuses the BBG feature pipeline but with
    the target price replaced by a spliced BBG-then-ETF series. The macro
    feature columns (Stock_Bond_Corr, yields, VIX) stay sourced from BBG/FRED/Yahoo
    exactly as in `_build_bbg_features` — only the target series is hybrid."""
    raw, fred, vix, irx = _load_bbg_raw()
    target = _build_hybrid_price_series(bbg_col, etf_ticker, start_date, end_date)

    # Build a raw frame with the synthetic target alongside the helper columns
    # used by the BBG feature path (SPTR for stock-bond corr, LBUSTRUU for bond ret).
    helper_cols = list(dict.fromkeys(['SPTR', 'LBUSTRUU']))
    helpers = raw[helper_cols]
    df = (pd.concat([target.rename('TARGET'), helpers], axis=1)
            .join(fred, how='inner').join(vix, how='inner')
            .join(irx, how='inner').ffill().dropna())

    f = pd.DataFrame(index=df.index)
    tr = df['TARGET'].pct_change().fillna(0)
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
    f['VIX_EWMA_log_diff'] = (np.log(df['VIX'] / df['VIX'].shift(1)).fillna(0)
                              .ewm(halflife=63).mean())
    sptr_ret = df['SPTR'].pct_change().fillna(0)
    bond_ret = df['LBUSTRUU'].pct_change().fillna(0)
    f['Stock_Bond_Corr'] = sptr_ret.rolling(252).corr(bond_ret).fillna(0)
    return f.dropna()


# ──────────────────────────────────────────────────────────────────────────────
# Per-asset signal computation (heavy step — cached to disk)
# ──────────────────────────────────────────────────────────────────────────────

UNIVERSES: List[str] = ['bloomberg', 'yahoo', 'yahoo_mutual', 'hybrid']

# Default MVO parameters — match the Portfolio Construction page sidebar defaults.
# Used by `build_full_cache_all_universes` to pre-build MVO results so the page
# loads instantly. If the user changes any MVO param in the sidebar, those
# values will not hit this cache (one fresh compute, then cached).
DEFAULT_MVO_PARAMS: Dict[str, object] = {
    'rebal_freq': 'daily',
    'gamma_risk_minvar': 10.0,
    'gamma_risk_mv_baseline': 5.0,
    'gamma_risk_mv_jmxgb': 10.0,
    'gamma_trade': 1.0,
    'w_ub': 0.40,
    'cov_hl_days': 252,
    'mu_baseline_hl_years': 5.0,
    'mu_jmxgb_lookback_years': 11.0,
}


def full_cache_window(universe: str) -> Tuple[str, str]:
    """Return (oos_start, oos_end) — the longest-window cache settings for this
    universe. Paper-aligned oos_start (2007-01-01 for BBG/Hybrid/Yahoo Mutual,
    2010-01-01 for Yahoo ETFs because SPBO inception is 2012-02-02 and the
    walk-forward handles partial coverage from 2010). oos_end is today, capped
    at the BBG file's last common date for the Bloomberg universe.
    """
    today = pd.Timestamp.today().normalize()
    if universe == 'bloomberg':
        oos_start = '2007-01-01'
        xlsx = os.path.join(CACHE_DIR, 'DATA PAUL.xlsx')
        if os.path.exists(xlsx):
            try:
                raw_bbg = pd.read_excel(xlsx, header=None, skiprows=6)
                raw_bbg.columns = ['Date'] + BBG_COLUMNS
                raw_bbg['Date'] = pd.to_datetime(raw_bbg['Date'])
                raw_bbg = raw_bbg.set_index('Date')
                last_dates = [raw_bbg[c].dropna().index.max() for c in BBG_COLUMNS]
                oos_end_ts = min(last_dates) if last_dates else today
            except Exception:
                oos_end_ts = today
        else:
            oos_end_ts = today
    elif universe == 'hybrid':
        oos_start = '2007-01-01'
        oos_end_ts = today
    elif universe == 'yahoo':
        oos_start = '2010-01-01'
        oos_end_ts = today
    elif universe == 'yahoo_mutual':
        oos_start = '2007-01-01'
        oos_end_ts = today
    else:
        raise ValueError(f'Unknown universe: {universe!r}')
    return oos_start, oos_end_ts.strftime('%Y-%m-%d')


def build_full_cache_all_universes(tc_mode: str = 'net',
                                   force_refresh: bool = False,
                                   n_jobs: Optional[int] = None,
                                   mvo_params: Optional[dict] = None,
                                   use_insample_mu: bool = True,
                                   progress_callback=None
                                   ) -> Dict[str, Tuple[str, str]]:
    """Build full-window signal + in-sample-μ + MVO-result caches for all 4
    universes. After this, opening the Portfolio Construction page hits all
    three caches and renders instantly.

    `mvo_params` overrides `DEFAULT_MVO_PARAMS` (e.g. when the user has tweaked
    sidebar values). `tc_oneway` is added automatically from `tc_mode`.

    `progress_callback(uni_idx, uni_total, universe, oos_start, oos_end, stage,
    asset_idx, asset_total, asset_name)` is called at each step. `stage` ∈
    {'signals', 'insample_mu', 'mvo', 'done'}.

    Returns the dict of (oos_start, oos_end) windows used per universe.
    """
    windows: Dict[str, Tuple[str, str]] = {}
    total = len(UNIVERSES)

    resolved_mvo = dict(DEFAULT_MVO_PARAMS)
    if mvo_params:
        resolved_mvo.update(mvo_params)
    resolved_mvo.setdefault('tc_oneway', 0.0 if tc_mode == 'gross' else 0.0005)

    for i, universe in enumerate(UNIVERSES):
        oos_start, oos_end = full_cache_window(universe)
        windows[universe] = (oos_start, oos_end)

        def _asset_cb_signals(a_idx, a_total, a_name, _i=i, _u=universe,
                              _s=oos_start, _e=oos_end):
            if progress_callback is not None:
                progress_callback(_i, total, _u, _s, _e, 'signals',
                                  a_idx, a_total, a_name)

        def _asset_cb_mu(a_idx, a_total, a_name, _i=i, _u=universe,
                         _s=oos_start, _e=oos_end):
            if progress_callback is not None:
                progress_callback(_i, total, _u, _s, _e, 'insample_mu',
                                  a_idx, a_total, a_name)

        if progress_callback is not None:
            progress_callback(i, total, universe, oos_start, oos_end, 'signals',
                              0, 0, '')
        compute_asset_signals(universe=universe, oos_start=oos_start, oos_end=oos_end,
                              tc_mode=tc_mode, force_refresh=force_refresh,
                              n_jobs=n_jobs,
                              progress_callback=_asset_cb_signals)
        if progress_callback is not None:
            progress_callback(i, total, universe, oos_start, oos_end, 'insample_mu',
                              0, 0, '')
        compute_insample_regime_means(universe=universe, oos_start=oos_start,
                                      oos_end=oos_end, tc_mode=tc_mode,
                                      force_refresh=force_refresh,
                                      n_jobs=n_jobs,
                                      progress_callback=_asset_cb_mu)

        # MVO: build the panel and run all 7 portfolios. Writes the disk cache
        # under the params_hash matching the resolved MVO params so the page
        # hits this cache on first load.
        if progress_callback is not None:
            progress_callback(i, total, universe, oos_start, oos_end, 'mvo',
                              0, 0, '')
        try:
            signals_for_panel = _load_extensible_signal_cache(
                universe, oos_start, oos_end, tc_mode) or {}
            insample_mu_for_panel = None
            if use_insample_mu:
                try:
                    insample_mu_for_panel = _load_extensible_insample_mu_cache(
                        universe, oos_start, oos_end, tc_mode)
                except Exception:
                    insample_mu_for_panel = None
            panel = build_asset_panel(signals_for_panel, oos_start, oos_end,
                                       insample_mu=insample_mu_for_panel)
            sig_mtime = signal_cache_mtime(universe, oos_start, oos_end, tc_mode)
            mu_mtime = (insample_mu_cache_mtime(universe, oos_start, oos_end, tc_mode)
                        if use_insample_mu else 0.0)
            run_all_portfolios_cached(
                panel,
                universe=universe, oos_start=oos_start, oos_end=oos_end,
                tc_mode=tc_mode, use_insample_mu=use_insample_mu,
                signal_mtime=sig_mtime, insample_mu_mtime=mu_mtime,
                **resolved_mvo,
            )
        except Exception as e:
            print(f'[portfolio] MVO pre-build failed for {universe}: {e}')

    if progress_callback is not None:
        progress_callback(total, total, '', '', '', 'done', 0, 0, '')
    return windows


def _signal_cache_path(universe: str, oos_start: str, oos_end: str,
                       tc_mode: str = 'net') -> str:
    # Backwards-compat: 'net' (5 bps, default) keeps the original filename.
    # 'gross' adds a suffix so paper-faithful (TC=0) signals cache separately.
    suffix = '' if tc_mode == 'net' else f'_{tc_mode}'
    fname = f'portfolio_signals_{universe}_{oos_start[:7]}_{oos_end[:7]}{suffix}.pkl'
    return os.path.join(CACHE_DIR, fname)


def _insample_mu_cache_path(universe: str, oos_start: str, oos_end: str,
                            tc_mode: str = 'net') -> str:
    # Mirrors signal cache naming so net/gross stay paired.
    suffix = '' if tc_mode == 'net' else f'_{tc_mode}'
    fname = f'portfolio_insample_mu_{universe}_{oos_start[:7]}_{oos_end[:7]}{suffix}.pkl'
    return os.path.join(CACHE_DIR, fname)


def _asset_specs_for(universe: str):
    if universe == 'bloomberg':
        return BBG_ASSETS
    if universe == 'yahoo':
        return YAHOO_ASSETS
    if universe == 'yahoo_mutual':
        return MUTUAL_FUNDS_ASSETS
    if universe == 'hybrid':
        return HYBRID_ASSETS
    raise ValueError(f'Unknown universe: {universe!r}')


def _start_date_for(universe: str) -> str:
    # Wide enough to give the 11-year JM lookback room
    if universe == 'bloomberg':
        return '1991-01-01'
    if universe == 'hybrid':
        return '1991-01-01'  # BBG drives the lookback; ETF takes over later
    if universe == 'yahoo_mutual':
        return '1990-01-01'  # ^VIX (1990-01-02) is the binding constraint
    return '1996-01-01'


def _features_for_asset(universe: str, spec, start_date: str, oos_end: str):
    """Build the feature DataFrame for one asset spec from the appropriate source."""
    if universe == 'bloomberg':
        asset_name, bbg_col, hl_proxy, include_dd = spec
        df_feat = _build_bbg_features(bbg_col, include_dd=include_dd)
    elif universe == 'hybrid':
        asset_name, bbg_col, etf_ticker, hl_proxy, include_dd = spec
        df_feat = _build_hybrid_features(bbg_col, etf_ticker, start_date,
                                         oos_end, include_dd=include_dd)
    else:
        asset_name, ticker, hl_proxy, include_dd = spec
        df_feat = _build_yahoo_features(ticker, start_date, oos_end, include_dd=include_dd)
    return asset_name, hl_proxy, include_dd, df_feat


def _run_walk_forward_for_asset(spec, universe, oos_start, oos_end,
                                tc_mode, start_date):
    """Run walk-forward for one asset between oos_start and oos_end.

    Returns the result DataFrame (with .attrs) or None on failure. Sets all
    needed `_main` globals internally (including TRANSACTION_COST per tc_mode)
    so the function is safe to call from a worker subprocess.
    """
    import main as _main
    from config import StrategyConfig

    try:
        asset_name, hl_proxy, include_dd, df_feat = _features_for_asset(
            universe, spec, start_date, oos_end)
    except Exception as e:
        spec_name = spec[0]
        print(f'[portfolio] {spec_name}: feature build failed — {e}')
        return None

    _main.TARGET_TICKER = hl_proxy
    _main._forecast_cache.clear()
    _main.OOS_START_DATE = oos_start
    _main.END_DATE = oos_end
    _main.START_DATE_DATA = start_date
    _main.LAMBDA_GRID = list(GRID_8PT)
    _main.TRANSACTION_COST = 0.0 if tc_mode == 'gross' else 0.0005

    cfg = StrategyConfig(name=f'Portfolio_{universe}_{asset_name}', ewma_mode='paper')
    try:
        r = _main.walk_forward_backtest(df_feat, cfg)
    except Exception as e:
        print(f'[portfolio] {asset_name}: walk_forward failed — {e}')
        return None
    if r is None or r.empty:
        return None

    r.attrs['hl_proxy'] = hl_proxy
    r.attrs['include_dd'] = include_dd
    r.attrs['tc_mode'] = tc_mode
    if universe == 'bloomberg':
        r.attrs['source_col'] = spec[1]
    elif universe == 'hybrid':
        r.attrs['source_col'] = spec[1]      # BBG column
        r.attrs['source_ticker'] = spec[2]   # ETF ticker
    else:
        r.attrs['source_ticker'] = spec[1]
    return r


def _limit_inner_threads():
    """Cap OpenMP / BLAS / MKL thread counts to 1 so that subprocess workers
    in a Parallel pool don't oversubscribe the CPU (XGBoost, OpenBLAS, numpy
    each default to all-cores). Must be called before heavy compute. Idempotent."""
    import os
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(k, '1')


def _process_one_asset_signals(spec, universe, oos_start, oos_end, tc_mode,
                               start_date, existing_df):
    """Picklable worker for parallel asset-level signal computation.

    Returns (asset_name, df_or_none). Self-contained: sets all required `_main`
    globals (TARGET_TICKER, dates, λ grid, TC) so subprocess workers don't need
    parent-side bootstrap. `existing_df` is the per-asset cached DataFrame
    (with .attrs) when extending, or None for a fresh run.
    """
    _limit_inner_threads()
    import main as _main
    from config import StrategyConfig

    asset_name = spec[0]

    if existing_df is not None and not existing_df.empty:
        next_anchor = _next_anchor_after(existing_df)
        if next_anchor is None or next_anchor >= pd.Timestamp(oos_end):
            return asset_name, None  # already covers target
        r_new = _run_walk_forward_for_asset(
            spec, universe, next_anchor.strftime('%Y-%m-%d'),
            oos_end, tc_mode, start_date)
        if r_new is None or r_new.empty:
            return asset_name, None
        r_new = r_new[r_new.index > existing_df.index.max()]
        if r_new.empty:
            return asset_name, None
        merged = pd.concat([existing_df, r_new])
        ewma_hl = existing_df.attrs.get(
            'ewma_halflife', r_new.attrs.get('ewma_halflife', 0))
        cfg = StrategyConfig(name=f'Portfolio_{universe}_{asset_name}',
                             ewma_mode='paper')
        merged = _main._finalize_walk_forward(merged, cfg, ewma_hl)
        merged.attrs['lambda_history'] = (
            list(existing_df.attrs.get('lambda_history', []))
            + list(r_new.attrs.get('lambda_history', [])))
        merged.attrs['lambda_dates'] = (
            list(existing_df.attrs.get('lambda_dates', []))
            + list(r_new.attrs.get('lambda_dates', [])))
        merged.attrs['ewma_halflife'] = ewma_hl
        merged.attrs['hl_proxy'] = existing_df.attrs.get(
            'hl_proxy', r_new.attrs.get('hl_proxy'))
        merged.attrs['include_dd'] = existing_df.attrs.get(
            'include_dd', r_new.attrs.get('include_dd', True))
        merged.attrs['tc_mode'] = tc_mode
        if universe == 'bloomberg':
            merged.attrs['source_col'] = existing_df.attrs.get(
                'source_col', spec[1])
        elif universe == 'hybrid':
            merged.attrs['source_col'] = existing_df.attrs.get(
                'source_col', spec[1])
            merged.attrs['source_ticker'] = existing_df.attrs.get(
                'source_ticker', spec[2])
        else:
            merged.attrs['source_ticker'] = existing_df.attrs.get(
                'source_ticker', spec[1])
        return asset_name, merged
    else:
        r = _run_walk_forward_for_asset(spec, universe, oos_start, oos_end,
                                         tc_mode, start_date)
        return asset_name, r


def _signals_max_end(signals: Dict[str, pd.DataFrame]) -> Optional[pd.Timestamp]:
    ends = [df.index.max() for df in signals.values() if not df.empty]
    return max(ends) if ends else None


def _next_anchor_after(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Find the next 6-month anchor after the last computed one (== start of the
    next uncomputed chunk). Returns None if no anchors exist."""
    lam_dates = pd.to_datetime(df.attrs.get('lambda_dates', []))
    if len(lam_dates) == 0:
        return None
    return lam_dates[-1] + pd.DateOffset(months=6)


def compute_asset_signals(universe: str = 'bloomberg',
                          oos_start: str = '2007-01-01',
                          oos_end: str = '2023-12-31',
                          force_refresh: bool = False,
                          tc_mode: str = 'net',
                          n_jobs: Optional[int] = None,
                          progress_callback=None) -> Dict[str, pd.DataFrame]:
    """Run walk-forward backtest for each of the 12 assets and return a dict
    keyed by asset name. Each value is a DataFrame containing at least:
        Target_Return, RF_Rate, Forecast_State, Raw_Prob (when JM-XGB),
        and lambda_history attribute under .attrs.

    Caching strategy:
      • Exact cache hit (same universe/start/end/tc_mode) → returned as-is.
      • Cache exists but is older than `oos_end` → incremental extension:
        for each asset, run walk-forward only from `last_anchor + 6mo` to
        `oos_end` and merge into the existing series. EWMA smoothing is then
        re-applied on the full merged Raw_Prob so the boundary is seamless.
      • No cache (or `force_refresh=True`) → full computation.

    tc_mode='net'  → walk-forward tunes λ on net-of-TC Sharpe (5 bps default).
    tc_mode='gross' → tunes on gross Sharpe (TC=0) — apples-to-apples with paper
                      Tables 4/6/7. Caches separately by filename suffix.
    """
    if tc_mode not in ('net', 'gross'):
        raise ValueError(f"tc_mode must be 'net' or 'gross', got {tc_mode!r}")
    target_start = pd.Timestamp(oos_start)
    target_end = pd.Timestamp(oos_end)

    # Try existing cache (exact, covering, or extension candidate). The picker
    # scans across both oos_start and oos_end prefixes so a wider-window cache
    # can serve a narrower request via a free trim.
    existing: Optional[Dict[str, pd.DataFrame]] = None
    existing_path: Optional[str] = None
    if not force_refresh:
        existing_path = _pick_signal_cache_path(universe, oos_start, oos_end, tc_mode)
        existing = _load_extensible_signal_cache(universe, oos_start, oos_end, tc_mode)

    if existing is not None:
        max_end = _signals_max_end(existing)
        if max_end is not None and max_end >= target_end - pd.Timedelta(days=1):
            # Covering cache — trim by both ends and return a view. We
            # deliberately do NOT persist under a new oos_end/oos_start path:
            # a fresh file would have a new mtime, invalidating downstream
            # MVO caches whose params_hash pins to the signal mtime.
            return {k: v[(v.index >= target_start) & (v.index <= target_end)]
                    for k, v in existing.items()}

    start_date = _start_date_for(universe)
    asset_specs = _asset_specs_for(universe)

    # Pre-warm shared on-disk caches in the parent so worker subprocesses don't
    # all race on the same Yahoo/FRED fetch (BBG raw load also caches VIX/IRX).
    if universe in ('bloomberg', 'hybrid'):
        try:
            _load_bbg_raw()
        except Exception:
            pass

    signals: Dict[str, pd.DataFrame] = dict(existing) if existing else {}
    did_work = False  # set True only when walk-forward actually produced new bars

    # Build the task list. Skip assets that are already fully covered so we
    # don't waste a worker on a no-op.
    pending: List[Tuple] = []
    for spec in asset_specs:
        asset_name = spec[0]
        df_existing = signals.get(asset_name)
        if df_existing is not None and not df_existing.empty:
            next_anchor = _next_anchor_after(df_existing)
            if next_anchor is None or next_anchor >= target_end:
                continue
        pending.append((spec, df_existing))

    if not pending:
        # Everything is already covered — nothing to compute or save.
        if progress_callback is not None:
            progress_callback(len(asset_specs), len(asset_specs), 'done')
        return signals

    # Resolve worker count: default to min(cpu_count, #tasks). n_jobs=1 keeps
    # the sequential code path (useful for debugging or low-memory runs).
    if n_jobs is None:
        import multiprocessing
        n_jobs = min(multiprocessing.cpu_count(), len(pending))
    n_jobs = max(1, int(n_jobs))

    if n_jobs == 1:
        # Sequential path — workers mutate `_main.TRANSACTION_COST` in the
        # parent process; save/restore so we don't leak the per-tc_mode value
        # into other code in the same Streamlit session.
        import main as _main
        original_tc = _main.TRANSACTION_COST
        try:
            for i, (spec, df_existing) in enumerate(pending):
                asset_name = spec[0]
                if progress_callback is not None:
                    progress_callback(i, len(pending), asset_name)
                try:
                    _, r = _process_one_asset_signals(
                        spec, universe, oos_start, oos_end, tc_mode,
                        start_date, df_existing)
                except Exception as e:
                    print(f'[portfolio] {asset_name}: {e}')
                    r = None
                if r is not None and not r.empty:
                    signals[asset_name] = r
                    did_work = True
        finally:
            _main.TRANSACTION_COST = original_tc
    else:
        # Parallel: process subprocess pool (loky) — each worker gets isolated
        # `_main` globals so no inter-asset race on TARGET_TICKER etc.
        from joblib import Parallel, delayed, parallel_config
        completed = 0
        with parallel_config(backend='loky', n_jobs=n_jobs,
                             inner_max_num_threads=1):
            parallel = Parallel(return_as='generator_unordered')
            gen = parallel(
                delayed(_process_one_asset_signals)(
                    spec, universe, oos_start, oos_end, tc_mode,
                    start_date, df_existing
                ) for spec, df_existing in pending
            )
            for asset_name, r in gen:
                if r is not None and not r.empty:
                    signals[asset_name] = r
                    did_work = True
                if progress_callback is not None:
                    progress_callback(completed, len(pending), asset_name)
                completed += 1

    # Persist only when new bars were computed. A no-op pass (e.g. requested
    # oos_end falls on a weekend so every asset's next anchor is past the
    # target) must NOT bump the file mtime — downstream caches' params_hash
    # depends on it. When extending an existing wider cache, preserve its
    # start token and update the end token to the data's true max — keeping
    # filenames honest, replacing the old file in place (no fragmentation).
    if did_work:
        new_max_end = _signals_max_end(signals)
        if existing_path is not None:
            existing_tokens = _parse_signal_cache_tokens(existing_path, tc_mode)
            start_token = existing_tokens[0] if existing_tokens else oos_start[:7]
        else:
            start_token = oos_start[:7]
        end_token = new_max_end.strftime('%Y-%m') if new_max_end is not None \
                                                  else oos_end[:7]
        suffix = '' if tc_mode == 'net' else f'_{tc_mode}'
        save_path = os.path.join(
            CACHE_DIR,
            f'portfolio_signals_{universe}_{start_token}_{end_token}{suffix}.pkl',
        )
        try:
            with open(save_path, 'wb') as fh:
                pickle.dump(signals, fh)
            if existing_path is not None and existing_path != save_path \
                    and os.path.exists(existing_path):
                try:
                    os.remove(existing_path)
                except OSError:
                    pass
        except Exception as e:
            print(f'[portfolio] failed to cache signals: {e}')

    if progress_callback is not None:
        progress_callback(len(asset_specs), len(asset_specs), 'done')
    return signals


def _parse_signal_cache_tokens(path: str, tc_mode: str) -> Optional[Tuple[str, str]]:
    """Return (start_token, end_token) parsed from a signal cache filename, or None."""
    suffix = '' if tc_mode == 'net' else f'_{tc_mode}'
    stem = os.path.basename(path)[:-4]
    if suffix and stem.endswith(suffix):
        stem = stem[:-len(suffix)]
    parts = stem.split('_')
    if len(parts) < 2:
        return None
    return parts[-2], parts[-1]


def _scan_signal_caches(universe: str, tc_mode: str
                        ) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Return all signal cache files matching the universe / tc_mode as
    (cache_start, cache_end, path) tuples, parsed from the filename."""
    import glob
    suffix = '' if tc_mode == 'net' else f'_{tc_mode}'
    prefix = f'portfolio_signals_{universe}_'
    pattern = f'{prefix}*_*{suffix}.pkl'
    out: List[Tuple[pd.Timestamp, pd.Timestamp, str]] = []
    for path in glob.glob(os.path.join(CACHE_DIR, pattern)):
        stem = os.path.basename(path)[:-4]
        if suffix and stem.endswith(suffix):
            stem = stem[:-len(suffix)]
        if not stem.startswith(prefix):
            continue
        body = stem[len(prefix):]
        parts = body.split('_')
        if len(parts) != 2:
            continue
        start_token, end_token = parts
        try:
            start_ts = pd.Timestamp(start_token + '-01')
            end_ts = pd.Timestamp(end_token + '-01') + pd.offsets.MonthEnd(0)
        except Exception:
            continue
        out.append((start_ts, end_ts, path))
    return out


def _pick_signal_cache_path(universe: str, oos_start: str, oos_end: str,
                            tc_mode: str) -> Optional[str]:
    """Return the path to the signal cache file that backs the requested window.

    Lookup precedence (so the same on-disk file backs as many windows as possible):
      1. Exact match (cache_start == request_start AND cache_end ≥ request_end-1d).
      2. Covering cache (cache_start ≤ request_start AND cache_end ≥ request_end-1d).
         Free trim — no recompute. Picks the cache closest to the request window.
      3. Same-start extension (cache_start == request_start, cache_end < request_end).
         Walk-forward extends forward from the cache's last anchor.
      4. Earlier-start extension (cache_start ≤ request_start, cache_end < request_end).
         Same as 3 but the cache started earlier — we extend its tail forward.
      None if no matching cache exists.
    """
    candidates = _scan_signal_caches(universe, tc_mode)
    if not candidates:
        return None
    request_start = pd.Timestamp(oos_start)
    request_end = pd.Timestamp(oos_end)
    cover_end = request_end - pd.Timedelta(days=1)

    exact_path = _signal_cache_path(universe, oos_start, oos_end, tc_mode=tc_mode)
    for s, e, p in candidates:
        if p == exact_path and e >= cover_end:
            return p

    covering = [(s, e, p) for s, e, p in candidates
                if s <= request_start and e >= cover_end]
    if covering:
        # Prefer the cache whose start is closest to (≤) request_start; tiebreak
        # by smallest cache_end (least excess at the tail).
        covering.sort(key=lambda c: (request_start - c[0], c[1] - request_end))
        return covering[0][2]

    same_start = [(s, e, p) for s, e, p in candidates if s == request_start]
    if same_start:
        same_start.sort(key=lambda c: c[1], reverse=True)
        return same_start[0][2]

    earlier = [(s, e, p) for s, e, p in candidates if s <= request_start]
    if earlier:
        earlier.sort(key=lambda c: c[1], reverse=True)
        return earlier[0][2]

    return None


def signal_cache_mtime(universe: str, oos_start: str, oos_end: str,
                       tc_mode: str) -> float:
    """mtime of the file that actually backs the requested window. Stable
    across different oos_end values served from the same underlying cache —
    so downstream caches (MVO results) don't invalidate just because the
    user changed the displayed window."""
    path = _pick_signal_cache_path(universe, oos_start, oos_end, tc_mode)
    return os.path.getmtime(path) if path else 0.0


def _load_extensible_signal_cache(universe: str, oos_start: str, oos_end: str,
                                  tc_mode: str) -> Optional[Dict[str, pd.DataFrame]]:
    """Load the signal cache picked by `_pick_signal_cache_path`."""
    chosen = _pick_signal_cache_path(universe, oos_start, oos_end, tc_mode)
    if chosen is None:
        return None
    try:
        with open(chosen, 'rb') as fh:
            signals = pickle.load(fh)
        if all(isinstance(v, pd.DataFrame) for v in signals.values()):
            return signals
    except Exception:
        pass
    return None


def clear_signal_cache(universe: Optional[str] = None) -> List[str]:
    """Remove on-disk signal caches. Returns list of files deleted.

    Globs match both net (no suffix) and gross (`_gross` suffix) variants,
    and the paired in-sample μ caches.
    """
    import glob
    patterns = []
    if universe:
        patterns.append(f'portfolio_signals_{universe}*.pkl')
        patterns.append(f'portfolio_insample_mu_{universe}*.pkl')
    else:
        patterns.append('portfolio_signals_*.pkl')
        patterns.append('portfolio_insample_mu_*.pkl')
    patterns.append('portfolio_results_*.pkl')

    deleted = []
    for pat in patterns:
        for f in glob.glob(os.path.join(CACHE_DIR, pat)):
            try:
                os.remove(f)
                deleted.append(os.path.basename(f))
            except OSError:
                pass
    return deleted


# ──────────────────────────────────────────────────────────────────────────────
# In-sample regime means — paper's exact spec for MV(JM-XGB) μ
# ──────────────────────────────────────────────────────────────────────────────

def _process_one_asset_insample_mu(spec, universe, oos_start, oos_end, start_date,
                                    lambda_history, lambda_dates, cached_df):
    """Picklable worker for parallel in-sample regime-mean computation per asset.

    Returns (asset_name, df_or_none). The caller passes the pre-extracted
    `lambda_history` / `lambda_dates` lists (small, cheap to pickle) rather
    than the full signal DataFrame.
    """
    _limit_inner_threads()
    import main as _main
    _main.START_DATE_DATA = start_date
    _main.OOS_START_DATE = oos_start
    _main.END_DATE = oos_end
    _main.LAMBDA_GRID = list(GRID_8PT)

    asset_name = spec[0]
    if len(lambda_history) == 0:
        return asset_name, None

    cached_anchors = (pd.to_datetime(cached_df.index)
                      if cached_df is not None and not cached_df.empty
                      else pd.DatetimeIndex([]))
    todo: List[Tuple[pd.Timestamp, float]] = []
    lam_dates_ts = pd.to_datetime(lambda_dates)
    for a, l in zip(lam_dates_ts, lambda_history):
        if a not in cached_anchors:
            todo.append((a, float(l)))
    if not todo:
        return asset_name, None

    try:
        _, _, _, df_feat = _features_for_asset(universe, spec, start_date, oos_end)
    except Exception as e:
        print(f'[portfolio] in-sample μ {asset_name}: feature build failed — {e}')
        return asset_name, None
    return_features = [c for c in df_feat.columns
                       if c.startswith(('DD_', 'Avg_Ret_', 'Sortino_'))]

    new_rows = []
    for anchor, lmbda in todo:
        train_start = anchor - pd.DateOffset(years=11)
        tr = df_feat[(df_feat.index >= train_start) & (df_feat.index < anchor)].copy()
        if len(tr) < 252 * 5:
            continue
        X = tr[return_features]
        X = (X - X.mean()) / X.std()
        jm = _main.StatisticalJumpModel(n_states=2, lambda_penalty=lmbda)
        states = jm.fit_predict(X.values)
        cum0 = tr['Excess_Return'][states == 0].sum()
        cum1 = tr['Excess_Return'][states == 1].sum()
        if cum1 > cum0:
            states = 1 - states
        mu_bull = float(tr['Excess_Return'][states == 0].mean()) if (states == 0).any() else 0.0
        mu_bear = float(tr['Excess_Return'][states == 1].mean()) if (states == 1).any() else 0.0
        new_rows.append({'anchor': anchor, 'mu_bull': mu_bull, 'mu_bear': mu_bear})

    if not new_rows:
        return asset_name, None
    new_df = pd.DataFrame(new_rows).set_index('anchor')
    if cached_df is None or cached_df.empty:
        return asset_name, new_df.sort_index()
    return asset_name, pd.concat([cached_df, new_df]).sort_index()


def _pick_insample_mu_cache_path(universe: str, oos_start: str,
                                 tc_mode: str) -> Optional[str]:
    """Return the in-sample μ cache file with the most cached anchors for this
    universe/tc_mode, restricted to caches whose oos_start ≤ the requested
    oos_start. None if no eligible cache exists."""
    import glob
    suffix = '' if tc_mode == 'net' else f'_{tc_mode}'
    prefix = f'portfolio_insample_mu_{universe}_'
    pattern = f'{prefix}*_*{suffix}.pkl'
    request_start = pd.Timestamp(oos_start)
    best_path: Optional[str] = None
    best_count = -1
    for p in glob.glob(os.path.join(CACHE_DIR, pattern)):
        stem = os.path.basename(p)[:-4]
        if suffix and stem.endswith(suffix):
            stem = stem[:-len(suffix)]
        if not stem.startswith(prefix):
            continue
        body = stem[len(prefix):]
        parts = body.split('_')
        if len(parts) != 2:
            continue
        try:
            cache_start = pd.Timestamp(parts[0] + '-01')
        except Exception:
            continue
        if cache_start > request_start:
            continue
        try:
            with open(p, 'rb') as fh:
                cached = pickle.load(fh)
            if not isinstance(cached, dict):
                continue
            if not all(isinstance(v, pd.DataFrame) for v in cached.values()):
                continue
            count = sum(len(v) for v in cached.values())
            if count > best_count:
                best_path = p
                best_count = count
        except Exception:
            continue
    return best_path


def insample_mu_cache_mtime(universe: str, oos_start: str, oos_end: str,
                            tc_mode: str) -> float:
    """mtime of the in-sample μ file actually backing the requested window.
    See `signal_cache_mtime` for the rationale."""
    path = _pick_insample_mu_cache_path(universe, oos_start, tc_mode)
    return os.path.getmtime(path) if path else 0.0


def _load_extensible_insample_mu_cache(universe: str, oos_start: str, oos_end: str,
                                       tc_mode: str) -> Optional[Dict[str, pd.DataFrame]]:
    """Find any prior in-sample μ cache for the same universe/oos_start/tc_mode
    so we can extend it with newly-computed anchors."""
    path = _pick_insample_mu_cache_path(universe, oos_start, tc_mode)
    if path is None:
        return None
    try:
        with open(path, 'rb') as fh:
            return pickle.load(fh)
    except Exception:
        return None


def compute_insample_regime_means(universe: str,
                                  oos_start: str = '2007-01-01',
                                  oos_end: str = '2023-12-31',
                                  force_refresh: bool = False,
                                  tc_mode: str = 'net',
                                  n_jobs: Optional[int] = None,
                                  progress_callback=None) -> Dict[str, pd.DataFrame]:
    """For each asset, at each biannual rebalance anchor, fit JM on the 11-year
    lookback window with the asset's selected λ and compute in-sample
    regime-conditional means.

    Returns:
        dict {asset_name: DataFrame(index=anchor_date,
                                   columns=['mu_bull', 'mu_bear'])}

    Caching is incremental: only anchors that are present in the signal cache
    but not yet in the μ cache are re-computed. Cache is keyed by
    (universe, oos_start, oos_end, tc_mode) so each window has its own file,
    but a prior cache file (same universe, earlier oos_end) is reused as the
    starting point for an extension.
    """
    if tc_mode not in ('net', 'gross'):
        raise ValueError(f"tc_mode must be 'net' or 'gross', got {tc_mode!r}")

    existing: Dict[str, pd.DataFrame] = {}
    existing_path: Optional[str] = None
    if not force_refresh:
        existing_path = _pick_insample_mu_cache_path(universe, oos_start, tc_mode)
        loaded = _load_extensible_insample_mu_cache(universe, oos_start, oos_end, tc_mode)
        if loaded is not None:
            existing = loaded

    import main as _main
    from config import StrategyConfig  # noqa: F401 (kept for symmetry / consistent imports)

    signals = _load_extensible_signal_cache(universe, oos_start, oos_end, tc_mode)
    if signals is None:
        raise FileNotFoundError(
            f'No signals cache covering {universe}/{oos_start}/{oos_end}/{tc_mode}. '
            f'Run compute_asset_signals(tc_mode={tc_mode!r}) first.')

    asset_specs = _asset_specs_for(universe)
    start_date = _start_date_for(universe)

    out: Dict[str, pd.DataFrame] = {k: v.copy() for k, v in existing.items()}
    # Quick check — fast-path if every asset's anchors are already cached.
    needs_work = False
    for spec in asset_specs:
        asset_name = spec[0]
        if asset_name not in signals:
            continue
        sig_anchors = pd.to_datetime(signals[asset_name].attrs.get('lambda_dates', []))
        cached_anchors = (pd.to_datetime(out[asset_name].index)
                          if asset_name in out else pd.DatetimeIndex([]))
        if sig_anchors.difference(cached_anchors).size > 0:
            needs_work = True
            break

    if not needs_work and out:
        # All anchors already cached on disk under some earlier cache file —
        # return as-is. We deliberately do NOT write a new file under the
        # current oos_end: a fresh mtime would invalidate downstream caches
        # (MVO results) whose params_hash pins to the in-sample-μ mtime,
        # forcing a recompute every time the user shifts oos_end.
        return out

    # Pre-warm shared data caches (same rationale as in compute_asset_signals).
    if universe in ('bloomberg', 'hybrid'):
        try:
            _load_bbg_raw()
        except Exception:
            pass

    # Build the per-asset task list, extracting only the metadata each worker
    # needs (no need to pickle the full signal DataFrame).
    pending: List[Tuple] = []
    for spec in asset_specs:
        asset_name = spec[0]
        if asset_name not in signals:
            continue
        sig_df = signals[asset_name]
        lambda_history = list(sig_df.attrs.get('lambda_history', []))
        lambda_dates = list(pd.to_datetime(sig_df.attrs.get('lambda_dates', [])))
        if not lambda_history:
            continue
        cached_df = out.get(asset_name)
        cached_anchors = (pd.to_datetime(cached_df.index)
                          if cached_df is not None and not cached_df.empty
                          else pd.DatetimeIndex([]))
        if pd.DatetimeIndex(lambda_dates).difference(cached_anchors).size == 0:
            continue
        pending.append((spec, lambda_history, lambda_dates, cached_df))

    if not pending:
        if progress_callback is not None:
            progress_callback(len(asset_specs), len(asset_specs), 'done')
        return out

    if n_jobs is None:
        import multiprocessing
        n_jobs = min(multiprocessing.cpu_count(), len(pending))
    n_jobs = max(1, int(n_jobs))

    did_work = False  # True only if at least one worker added new mu rows
    if n_jobs == 1:
        for i, (spec, lh, ld, cached_df) in enumerate(pending):
            asset_name = spec[0]
            if progress_callback is not None:
                progress_callback(i, len(pending), asset_name)
            try:
                _, df_out = _process_one_asset_insample_mu(
                    spec, universe, oos_start, oos_end, start_date,
                    lh, ld, cached_df)
            except Exception as e:
                print(f'[portfolio] in-sample μ {asset_name}: {e}')
                df_out = None
            if df_out is not None:
                out[asset_name] = df_out
                did_work = True
    else:
        from joblib import Parallel, delayed, parallel_config
        completed = 0
        with parallel_config(backend='loky', n_jobs=n_jobs,
                             inner_max_num_threads=1):
            parallel = Parallel(return_as='generator_unordered')
            gen = parallel(
                delayed(_process_one_asset_insample_mu)(
                    spec, universe, oos_start, oos_end, start_date,
                    lh, ld, cached_df
                ) for spec, lh, ld, cached_df in pending
            )
            for asset_name, df_out in gen:
                if df_out is not None:
                    out[asset_name] = df_out
                    did_work = True
                if progress_callback is not None:
                    progress_callback(completed, len(pending), asset_name)
                completed += 1

    if progress_callback is not None:
        progress_callback(len(asset_specs), len(asset_specs), 'done')

    # Don't save if no new μ rows were added. This is critical: several assets
    # have permanently-missing early anchors (insufficient 11-year lookback at
    # the start of the OOS window — e.g. SPBO inception 2012). Without this
    # guard, every call rewrites the file → mtime drift → params_hash changes →
    # MVO disk cache misses on subsequent calls.
    if not did_work:
        return out

    # Save with filename reflecting the actual data span. Preserve the existing
    # cache's start when extending; update the end to the latest anchor.
    if existing_path is not None:
        existing_tokens = _parse_signal_cache_tokens(existing_path, tc_mode)
        start_token = existing_tokens[0] if existing_tokens else oos_start[:7]
    else:
        start_token = oos_start[:7]
    all_anchors = [df.index.max() for df in out.values() if not df.empty]
    new_max_anchor = max(all_anchors) if all_anchors else None
    end_token = new_max_anchor.strftime('%Y-%m') if new_max_anchor is not None \
                                                 else oos_end[:7]
    suffix = '' if tc_mode == 'net' else f'_{tc_mode}'
    save_path = os.path.join(
        CACHE_DIR,
        f'portfolio_insample_mu_{universe}_{start_token}_{end_token}{suffix}.pkl',
    )
    try:
        with open(save_path, 'wb') as fh:
            pickle.dump(out, fh)
        if existing_path is not None and existing_path != save_path \
                and os.path.exists(existing_path):
            try:
                os.remove(existing_path)
            except OSError:
                pass
    except Exception as e:
        print(f'[portfolio] failed to cache in-sample μ: {e}')
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Panel assembly: align signals to a common date index
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AssetPanel:
    """A panel of aligned per-asset data over the OOS period."""
    returns:    pd.DataFrame   # daily total returns, columns = asset names
    forecast:   pd.DataFrame   # 0=bullish, 1=bearish, columns = asset names
    raw_prob:   pd.DataFrame   # P(bear) ∈ [0,1] (NaN for JM-only fallback)
    rf_daily:   pd.Series      # daily risk-free rate
    asset_order: List[str]
    # Optional: in-sample regime means per asset and anchor date. dict
    # keyed by asset name → DataFrame(index=anchor, columns=['mu_bull','mu_bear'])
    insample_mu: Optional[Dict[str, pd.DataFrame]] = None


def build_asset_panel(signals: Dict[str, pd.DataFrame],
                      oos_start: str, oos_end: str,
                      insample_mu: Optional[Dict[str, pd.DataFrame]] = None) -> AssetPanel:
    """Align per-asset Walk-Forward outputs into a single panel.

    NOTE: `Forecast_State[t]` in `walk_forward_backtest` is computed from features
    that include day-t's price action; main.py then applies `.shift(1)` before
    using it for trading. We replicate that shift here so that
    `panel.forecast.loc[t]` is the actionable forecast at the open of day t
    (i.e., made from data through t-1). Without this, applying weights on day t
    to day-t's return would be look-ahead.
    """
    ret_cols, fc_cols, prob_cols = {}, {}, {}
    rf_series = None

    # Use canonical asset order from whichever spec list the signals came from;
    # all three lists share the same 12 asset names so BBG_ASSETS ordering works universally.
    asset_order = [name for name, *_ in BBG_ASSETS if name in signals]
    if not asset_order:
        asset_order = list(signals.keys())

    for name in asset_order:
        df = signals[name]
        df = df[(df.index >= oos_start) & (df.index <= oos_end)].copy()
        ret_cols[name] = df['Target_Return']
        # Shift forecast +1 to align with the next-day execution convention used
        # in main.simulate_strategy / walk_forward_backtest.
        if 'Forecast_State' in df.columns:
            fc_cols[name] = df['Forecast_State'].shift(1).fillna(0).astype(int)
        else:
            fc_cols[name] = pd.Series(0, index=df.index)
        prob_cols[name] = (df['Raw_Prob'].shift(1) if 'Raw_Prob' in df.columns
                           else pd.Series(np.nan, index=df.index))
        if rf_series is None:
            rf_series = df['RF_Rate']

    returns = pd.DataFrame(ret_cols).dropna(how='all').fillna(0)
    forecast = pd.DataFrame(fc_cols).reindex(returns.index).ffill().fillna(0).astype(int)
    raw_prob = pd.DataFrame(prob_cols).reindex(returns.index).ffill()
    rf_daily = rf_series.reindex(returns.index).ffill().fillna(0)

    return AssetPanel(returns=returns, forecast=forecast, raw_prob=raw_prob,
                      rf_daily=rf_daily, asset_order=asset_order,
                      insample_mu=insample_mu)


# ──────────────────────────────────────────────────────────────────────────────
# Covariance and expected-return models
# ──────────────────────────────────────────────────────────────────────────────

def ewm_covariance(returns: pd.DataFrame, halflife: int = 252) -> pd.DataFrame:
    """EWM covariance of daily returns up to and including the last row.

    Returns: ndarray-of-floats (n × n), indexed by columns of `returns`.
    """
    if len(returns) < 30:
        # Fallback to sample cov when too few observations
        return returns.cov()
    cov = returns.ewm(halflife=halflife, min_periods=30).cov()
    last_date = cov.index.get_level_values(0).unique()[-1]
    return cov.loc[last_date]


def ewm_mean_returns(returns: pd.DataFrame, halflife_years: float = 5) -> pd.Series:
    """EWM mean of daily returns at the last date (used by MV baseline)."""
    hl_days = int(halflife_years * 252)
    return returns.ewm(halflife=hl_days, min_periods=30).mean().iloc[-1]


def regime_conditional_mu(returns: pd.DataFrame, forecast: pd.DataFrame,
                          forecast_today: pd.Series,
                          lookback_days: int = 252 * 5,
                          bear_cap: float = -0.001,
                          forecast_valid_from: Optional[pd.Timestamp] = None) -> pd.Series:
    """For MV(JM-XGB) — historical mean conditional on the asset's
    *forecasted* regime over the lookback window.

    For each asset j with current forecast f_j ∈ {0, 1}:
        μ_j = mean(returns[j, t]) over the lookback window where
              forecast[j, t] == f_j

    Bearish forecasts are capped at `bear_cap` (paper: -10 bps/day).

    `forecast_valid_from` excludes dates before that timestamp from the
    conditional-mean computation — preventing dilution by pre-OOS dates that
    have no real forecast (and would otherwise be filled with default bullish).

    This is a tractable approximation of the paper's spec — the paper conditions
    on the *in-sample JM regime* rather than the OOS forecasted regime.
    """
    mu = pd.Series(0.0, index=returns.columns)
    if len(returns) < 30:
        return mu
    window = returns.tail(lookback_days)
    if forecast_valid_from is not None:
        window = window.loc[window.index >= forecast_valid_from]
    fc_window = forecast.reindex(window.index).fillna(0).astype(int)
    for asset in returns.columns:
        f = int(forecast_today[asset])
        mask = fc_window[asset] == f
        if mask.sum() < 5:
            # Fallback to unconditional mean over the same valid window
            mu[asset] = float(window[asset].mean()) if len(window) else 0.0
            if f == 1:
                mu[asset] = max(mu[asset], bear_cap)
            continue
        m = window.loc[mask.values, asset].mean()
        if f == 1:  # bearish — cap downside
            m = max(float(m), bear_cap)
        mu[asset] = float(m)
    return mu


# ──────────────────────────────────────────────────────────────────────────────
# MVO solver (scipy SLSQP — small QP, n ≤ ~12 risky)
# ──────────────────────────────────────────────────────────────────────────────

_CVXPY_AVAILABLE = False
try:
    import cvxpy as _cp
    _CVXPY_AVAILABLE = True
except ImportError:
    pass


def _solve_mvo_cvxpy(mu, Sigma, w_pre, gamma_risk, gamma_trade,
                     tc_oneway, w_ub, L, active):
    """Convex QP via cvxpy — what Gurobi would do for the paper's formulation.
    Uses CLARABEL by default (interior-point, fast for small QPs)."""
    n = len(mu)
    eff_ub = np.where(active > 0, w_ub, 0.0)
    Sigma_sym = 0.5 * (Sigma + Sigma.T)

    w = _cp.Variable(n, nonneg=True)
    risk = _cp.quad_form(w, _cp.psd_wrap(Sigma_sym))
    obj  = (mu @ w
            - gamma_risk * risk
            - gamma_trade * tc_oneway * _cp.norm1(w - w_pre))
    constraints = [w <= eff_ub, _cp.sum(w) <= L]
    prob = _cp.Problem(_cp.Maximize(obj), constraints)
    try:
        prob.solve(solver=_cp.CLARABEL, verbose=False)
        if w.value is None or prob.status != 'optimal':
            return None
        return np.clip(w.value, 0, eff_ub)
    except Exception:
        return None


def solve_mvo(mu: np.ndarray, Sigma: np.ndarray, w_pre: np.ndarray,
              gamma_risk: float, gamma_trade: float,
              tc_oneway: float = 0.0005, w_ub: float = 0.40, L: float = 1.0,
              active: Optional[np.ndarray] = None,
              solver: str = 'slsqp') -> np.ndarray:
    """
    Solve:
        max_w  μ'w − γ_risk w'Σw − γ_trade · a · ‖w − w_pre‖_1
        s.t.  0 ≤ w ≤ w_ub,   sum(w) ≤ L
    Where `active` is an optional 0/1 mask: assets with active=0 are forced to 0.

    Reformulation: x = [w, t] of length 2n with t ≥ |w − w_pre|. Objective is
    convex quadratic in w plus linear in t. SLSQP is fast (~20 ms per solve);
    `trust-constr` finds slightly more-diversified interior solutions but is
    much slower. The math itself drives small allocations when daily μ ≈
    γ_trade · tc — see the dashboard for guidance.
    """
    n = len(mu)
    if active is None:
        active = np.ones(n)
    eff_ub = np.where(active > 0, w_ub, 0.0)

    # Try cvxpy first (proper convex QP, matches Gurobi's behavior)
    if solver == 'cvxpy' and _CVXPY_AVAILABLE:
        w_opt = _solve_mvo_cvxpy(mu, Sigma, w_pre, gamma_risk, gamma_trade,
                                  tc_oneway, w_ub, L, active)
        if w_opt is not None:
            s = w_opt.sum()
            if s > L + 1e-6:
                w_opt = w_opt * (L / s)
            return w_opt
        # else fall through to scipy

    Sigma_sym = 0.5 * (Sigma + Sigma.T)

    def obj(x):
        w = x[:n]; t = x[n:]
        return -float(mu @ w) + float(gamma_risk * w @ Sigma_sym @ w) + \
               float(gamma_trade * tc_oneway * t.sum())

    def grad(x):
        w = x[:n]
        gw = -mu + 2.0 * gamma_risk * (Sigma_sym @ w)
        gt = gamma_trade * tc_oneway * np.ones(n)
        return np.concatenate([gw, gt])

    x0 = np.concatenate([np.clip(w_pre, 0, eff_ub),
                         np.abs(w_pre - np.clip(w_pre, 0, eff_ub))])

    if solver == 'trust-constr':
        from scipy.optimize import LinearConstraint, Bounds

        def hess(x):
            H = np.zeros((2 * n, 2 * n))
            H[:n, :n] = 2.0 * gamma_risk * Sigma_sym
            return H

        lb_x = np.zeros(2 * n)
        ub_x = np.concatenate([eff_ub, np.full(n, np.inf)])
        A = np.zeros((1 + 2 * n, 2 * n))
        A[0, :n] = 1.0
        A[1:1+n, :n] = np.eye(n);    A[1:1+n, n:] = -np.eye(n)
        A[1+n:, :n] = -np.eye(n);   A[1+n:, n:] = -np.eye(n)
        ub_A = np.concatenate([[L], w_pre, -w_pre])
        linear = LinearConstraint(A, np.full_like(ub_A, -np.inf), ub_A)
        try:
            res = minimize(obj, x0, jac=grad, hess=hess,
                           bounds=Bounds(lb_x, ub_x), constraints=[linear],
                           method='trust-constr',
                           options={'maxiter': 200, 'xtol': 1e-8, 'verbose': 0})
            w_opt = np.clip(res.x[:n], 0, eff_ub)
        except Exception:
            return np.clip(w_pre, 0, eff_ub)
    else:
        bounds = [(0.0, float(eff_ub[i])) for i in range(n)] + [(0.0, None)] * n
        constraints = [
            {'type': 'ineq', 'fun': lambda x: float(L - x[:n].sum()),
             'jac':  lambda x: np.concatenate([-np.ones(n), np.zeros(n)])},
            {'type': 'ineq', 'fun': lambda x: x[n:] - (x[:n] - w_pre),
             'jac':  lambda x: np.hstack([-np.eye(n), np.eye(n)])},
            {'type': 'ineq', 'fun': lambda x: x[n:] + (x[:n] - w_pre),
             'jac':  lambda x: np.hstack([np.eye(n), np.eye(n)])},
        ]
        try:
            res = minimize(obj, x0, jac=grad, bounds=bounds, constraints=constraints,
                           method='SLSQP', options={'maxiter': 200, 'ftol': 1e-9})
            w_opt = np.clip(res.x[:n], 0, eff_ub)
        except Exception:
            return np.clip(w_pre, 0, eff_ub)

    s = w_opt.sum()
    if s > L + 1e-6:
        w_opt = w_opt * (L / s)
    return w_opt


# ──────────────────────────────────────────────────────────────────────────────
# Strategy weight functions — invoked at each rebalance
# ──────────────────────────────────────────────────────────────────────────────

def _bullish_active(panel: AssetPanel, date: pd.Timestamp) -> np.ndarray:
    """Return 0/1 array where 1 = bullish forecast for that date."""
    if date not in panel.forecast.index:
        # Use last available forecast on or before `date`
        idx = panel.forecast.index[panel.forecast.index <= date]
        if len(idx) == 0:
            return np.ones(len(panel.asset_order))
        date = idx[-1]
    f = panel.forecast.loc[date, panel.asset_order].values.astype(int)
    return (f == 0).astype(float)


def weights_60_40(panel: AssetPanel, date, w_pre, history) -> np.ndarray:
    """Fixed 60/40 mix per paper Table 1, rebalanced to target."""
    w = np.zeros(len(panel.asset_order))
    for i, name in enumerate(panel.asset_order):
        w[i] = PAPER_60_40.get(name, 0.0)
    return w


def _cov_window(history: pd.DataFrame, halflife_days: int) -> np.ndarray:
    """EWM covariance using all history (last value) — pandas weights via halflife."""
    n_assets = history.shape[1]
    if len(history) < 30:
        return np.eye(n_assets) * 1e-4
    ewm = history.ewm(halflife=halflife_days, min_periods=30).cov()
    last_date = ewm.index.get_level_values(0).unique()[-1]
    return ewm.loc[last_date].values


def weights_minvar_baseline(panel, date, w_pre, history, gamma_risk=10.0,
                            tc=0.0005, w_ub=0.40, L=1.0, cov_hl_days=252, **_):
    """MinVar baseline: equivalent to minimizing variance with constant μ."""
    Sigma = _cov_window(history, cov_hl_days)
    n = Sigma.shape[0]
    mu = np.ones(n)
    return solve_mvo(mu, Sigma, w_pre, gamma_risk, gamma_trade=0.0,
                     tc_oneway=tc, w_ub=w_ub, L=L)


def weights_minvar_jmxgb(panel, date, w_pre, history, gamma_risk=10.0,
                         gamma_trade=1.0, tc=0.0005, w_ub=0.40, L=1.0,
                         cov_hl_days=252, **_):
    """MinVar(JM-XGB): μ=10bps if bullish else 0; trade penalty on; ≤3 bull → cash.

    Paper Section 4.2: μ_j=10 bps for bullish, 0 for bearish. The optimizer
    chooses allocation freely (no hard active-mask restriction) — bearish
    assets naturally get 0 weight because their μ=0 ≤ trade-cost penalty.
    """
    bull = _bullish_active(panel, date)
    if bull.sum() <= 3:
        return np.zeros(len(panel.asset_order))
    Sigma = _cov_window(history, cov_hl_days)
    mu = np.where(bull > 0, 0.001, 0.0)  # 10 bps daily — paper Section 4.2
    return solve_mvo(mu, Sigma, w_pre, gamma_risk, gamma_trade,
                     tc_oneway=tc, w_ub=w_ub, L=L)


def weights_mv_baseline(panel, date, w_pre, history, gamma_risk=5.0,
                        tc=0.0005, w_ub=0.40, L=1.0,
                        cov_hl_days=252, mu_hl_years=5.0, **_):
    """MV baseline: μ from EWM hl=5y of *excess* returns; γ_trade = 0."""
    Sigma = _cov_window(history, cov_hl_days)
    # Paper Section 4.1: μ is excess-return forecast.
    rf_history = panel.rf_daily.reindex(history.index).fillna(0)
    excess = history.sub(rf_history, axis=0)
    mu = excess.ewm(halflife=int(mu_hl_years * 252), min_periods=30).mean().iloc[-1].values
    return solve_mvo(mu, Sigma, w_pre, gamma_risk, gamma_trade=0.0,
                     tc_oneway=tc, w_ub=w_ub, L=L)


def weights_mv_jmxgb(panel, date, w_pre, history, gamma_risk=10.0,
                     gamma_trade=1.0, tc=0.0005, w_ub=0.40, L=1.0,
                     cov_hl_days=252, mu_lookback_years=11.0, bear_cap=-0.001, **_):
    """MV(JM-XGB): regime-conditional historical mean over the 11-year training
    window, bearish forecast capped at -10 bps.

    If `panel.insample_mu` is set, uses paper-spec in-sample JM regime means
    (one bull/bear pair per asset per biannual anchor). Otherwise falls back to
    OOS-forecast-conditioned means (noisier, smaller).
    """
    bull = _bullish_active(panel, date)
    if bull.sum() <= 3:
        return np.zeros(len(panel.asset_order))
    Sigma = _cov_window(history, cov_hl_days)
    eligible = panel.forecast.index[panel.forecast.index <= date]
    fc_today = panel.forecast.loc[eligible[-1]].reindex(panel.asset_order) \
        if len(eligible) else panel.forecast.iloc[-1].reindex(panel.asset_order)

    if panel.insample_mu is not None and len(panel.insample_mu):
        mu_vals = np.zeros(len(panel.asset_order))
        for i, asset in enumerate(panel.asset_order):
            if asset not in panel.insample_mu:
                continue
            mu_df = panel.insample_mu[asset]
            anchors_before = mu_df.index[mu_df.index <= date]
            if len(anchors_before) == 0:
                continue
            row = mu_df.loc[anchors_before[-1]]
            f = int(fc_today[asset])
            m = float(row['mu_bull'] if f == 0 else row['mu_bear'])
            if f == 1:
                m = max(m, bear_cap)
            mu_vals[i] = m
        mu = pd.Series(mu_vals, index=panel.asset_order)
    else:
        mu_window_start = date - pd.DateOffset(days=int(mu_lookback_years * 365))
        mu_history = history.loc[mu_window_start:]
        fc_window = panel.forecast.reindex(mu_history.index).fillna(0).astype(int)
        forecast_start = panel.forecast.index.min()
        mu = regime_conditional_mu(mu_history[panel.asset_order],
                                   fc_window[panel.asset_order],
                                   fc_today, lookback_days=len(mu_history),
                                   bear_cap=bear_cap,
                                   forecast_valid_from=forecast_start)
    # Paper formulation (Section 4.1) doesn't restrict the optimizer by an
    # active-mask: bear-forecast assets have negative μ (capped at −10 bps) and
    # the convex optimizer will naturally drive their weights to 0.
    return solve_mvo(mu.values, Sigma, w_pre, gamma_risk, gamma_trade,
                     tc_oneway=tc, w_ub=w_ub, L=L)


def weights_ew_baseline(panel, date, w_pre, history, **_):
    """EW baseline: 1/N to each of 12 risky assets always."""
    n = len(panel.asset_order)
    return np.ones(n) / n


def weights_ew_jmxgb(panel, date, w_pre, history, **_):
    """EW(JM-XGB): equal weight among bullish assets; ≤3 bull → cash."""
    bull = _bullish_active(panel, date)
    if bull.sum() <= 3:
        return np.zeros(len(bull))
    return bull / bull.sum()


# ──────────────────────────────────────────────────────────────────────────────
# Portfolio simulator
# ──────────────────────────────────────────────────────────────────────────────

REBAL_FREQ_OFFSET = {
    'daily':       'D',
    'monthly':     'MS',
    'quarterly':   'QS',
    'biannually':  '2QS',
    'yearly':      'YS',
}


def _rebalance_dates(index: pd.DatetimeIndex, freq: str) -> set:
    """Return the subset of `index` that are rebalance dates for the given freq.

    For daily → every date. Otherwise, the first trading day on or after each
    period boundary.
    """
    if freq == 'daily':
        return set(index)
    period = {'monthly': 'M', 'quarterly': 'Q',
              'biannually': '2Q', 'yearly': 'Y'}.get(freq, 'M')
    if period == '2Q':
        # Use 2-quarter (semi-annual) periods anchored to Jan/Jul
        boundaries = pd.date_range(index.min(), index.max() + pd.Timedelta(days=1),
                                   freq='2QS-JAN')
    else:
        boundaries = pd.date_range(index.min(), index.max() + pd.Timedelta(days=1),
                                   freq={'M': 'MS', 'Q': 'QS', 'Y': 'YS'}[period])
    rebals = set()
    rebals.add(index[0])  # always rebalance on day 1
    for b in boundaries:
        # First trading day >= b
        eligible = index[index >= b]
        if len(eligible):
            rebals.add(eligible[0])
    return rebals


def run_portfolio_backtest(panel: AssetPanel, weight_fn, *,
                           rebal_freq: str = 'daily',
                           tc_oneway: float = 0.0005,
                           min_history_days: int = 252,
                           resume_from: Optional[dict] = None,
                           **weight_kwargs) -> dict:
    """Simulate the portfolio. weight_fn(panel, date, w_pre, history, **kwargs)
    returns the target risky-asset weights. Cash = 1 - sum(w).

    `history` is the full panel of returns *before* `date` (no look-ahead).
    Each weight function uses what it needs (Σ uses 252-day EWM weighting; μ
    uses 5y or 11y depending on strategy).

    Returns a dict containing:
        'returns': daily portfolio returns (pd.Series)
        'weights': daily weights (pd.DataFrame)
        'turnover_daily': daily turnover series
        'rebal_dates': set of dates where MVO was solved
        'mu_forecast': DataFrame of forecasted μ at each rebalance (for Table 7)
        'final_state': dict {'w_curr', 'cash', 'last_date'} for incremental
                       resumption on extension.

    `resume_from`: optional dict carrying prior results with `final_state`. When
    provided, the simulation skips dates ≤ final_state['last_date'] and
    initializes (w_curr, cash) from the saved state. The result merges the
    prior series with the newly-simulated tail.
    """
    returns = panel.returns.copy()
    rf_daily = panel.rf_daily.copy()
    n_assets = returns.shape[1]

    rebal_dates = _rebalance_dates(returns.index, rebal_freq)

    # Initialize (optionally from a prior cached run)
    if resume_from is not None and 'final_state' in resume_from:
        state = resume_from['final_state']
        last_date = pd.Timestamp(state['last_date'])
        resume_assets = list(state.get('asset_order', []))
        w_arr = np.asarray(state['w_curr'], dtype=float)
        # Re-align to current asset order — assume identical (panel order is stable)
        if resume_assets and resume_assets != panel.asset_order:
            ser = pd.Series(w_arr, index=resume_assets)
            w_curr = ser.reindex(panel.asset_order).fillna(0.0).values
        else:
            w_curr = w_arr
        cash = float(state['cash'])
        # Pre-load logs from the prior result; only sim dates strictly after last_date.
        prior_returns = resume_from['returns']
        prior_weights = resume_from['weights']
        prior_turnover = resume_from['turnover_daily']
        prior_mu = resume_from.get('mu_forecast', pd.DataFrame())

        sim_mask = returns.index > last_date
        # Index slicing helpers for the loop
        start_idx = int(np.searchsorted(returns.index.values, last_date.to_datetime64(), side='right'))
        port_rets: List[float] = []
        weight_log: List[np.ndarray] = []
        turnover_log: List[float] = []
        mu_forecast_log: Dict[pd.Timestamp, pd.Series] = {}
    else:
        last_date = None
        prior_returns = prior_weights = prior_turnover = None
        prior_mu = None
        sim_mask = np.ones(len(returns), dtype=bool)
        start_idx = 0
        w_curr = np.zeros(n_assets)
        cash = 1.0
        port_rets = []
        weight_log = []
        turnover_log = []
        mu_forecast_log = {}

    for t, date in enumerate(returns.index):
        if t < start_idx:
            continue
        is_rebal = date in rebal_dates and t >= min_history_days

        if is_rebal:
            history = returns.iloc[:t]  # all rows strictly before today (no look-ahead)
            if len(history) < 30:
                w_target = w_curr
            else:
                w_target = weight_fn(panel, date, w_curr, history, **weight_kwargs)

                # Capture μ forecast for Table 7 (MV and MV(JM-XGB))
                if weight_fn is weights_mv_jmxgb:
                    fc_eligible = panel.forecast.index[panel.forecast.index <= date]
                    fc_today = panel.forecast.loc[fc_eligible[-1]].reindex(panel.asset_order)
                    bear_cap_v = weight_kwargs.get('bear_cap', -0.001)
                    if panel.insample_mu is not None and len(panel.insample_mu):
                        mu_today = pd.Series(0.0, index=panel.asset_order)
                        for asset in panel.asset_order:
                            if asset not in panel.insample_mu:
                                continue
                            mu_df = panel.insample_mu[asset]
                            anchors_before = mu_df.index[mu_df.index <= date]
                            if len(anchors_before) == 0:
                                continue
                            row = mu_df.loc[anchors_before[-1]]
                            f = int(fc_today[asset])
                            m = float(row['mu_bull'] if f == 0 else row['mu_bear'])
                            if f == 1:
                                m = max(m, bear_cap_v)
                            mu_today[asset] = m
                    else:
                        mu_lb_yrs = weight_kwargs.get('mu_lookback_years', 11.0)
                        mu_window_start = date - pd.DateOffset(days=int(mu_lb_yrs * 365))
                        mu_history = history.loc[mu_window_start:]
                        fc_window = panel.forecast.reindex(mu_history.index).fillna(0).astype(int)
                        mu_today = regime_conditional_mu(mu_history[panel.asset_order],
                                                         fc_window[panel.asset_order],
                                                         fc_today, lookback_days=len(mu_history),
                                                         bear_cap=bear_cap_v,
                                                         forecast_valid_from=panel.forecast.index.min())
                    mu_forecast_log[date] = mu_today
                elif weight_fn is weights_mv_baseline:
                    mu_hl_yrs = weight_kwargs.get('mu_hl_years', 5.0)
                    mu_today = history[panel.asset_order].ewm(halflife=int(mu_hl_yrs * 252),
                                                              min_periods=30).mean().iloc[-1]
                    mu_forecast_log[date] = mu_today

            w_target = np.array(w_target, dtype=float)
            w_target = np.clip(w_target, 0.0, 1.0)
            if w_target.sum() > 1.0 + 1e-9:
                w_target = w_target / w_target.sum()

            trade = np.abs(w_target - w_curr).sum()
            turnover_log.append(trade)
            # Apply transaction cost on the rebalance day
            tc_drag = trade * tc_oneway
            w_curr = w_target
            cash = 1.0 - w_curr.sum()
        else:
            turnover_log.append(0.0)
            tc_drag = 0.0

        r_assets = returns.iloc[t].values
        rf_t = float(rf_daily.iloc[t])
        port_ret = float(w_curr @ r_assets) + cash * rf_t - tc_drag
        port_rets.append(port_ret)
        weight_log.append(w_curr.copy())

        # Drift weights with realized returns (no leverage; cash earns rf)
        if (1 + port_ret) > 0:
            new_w = w_curr * (1 + r_assets) / (1 + port_ret)
            new_cash = cash * (1 + rf_t) / (1 + port_ret)
            # Numerical safety: keep nonneg
            new_w = np.maximum(new_w, 0.0)
            new_cash = max(0.0, float(new_cash))
            w_curr = new_w
            cash = new_cash

    sim_index = returns.index[start_idx:]
    new_returns = pd.Series(port_rets, index=sim_index, name='Portfolio_Return')
    new_weights = pd.DataFrame(weight_log, index=sim_index, columns=panel.asset_order)
    new_turnover = pd.Series(turnover_log, index=sim_index)
    new_mu = pd.DataFrame(mu_forecast_log).T if mu_forecast_log else pd.DataFrame()

    if resume_from is not None:
        # Trim prior series to ≤ last_date to be safe, then concat.
        keep = prior_returns.index <= last_date
        merged_returns = pd.concat([prior_returns[keep], new_returns])
        merged_returns.name = 'Portfolio_Return'
        merged_weights = pd.concat([prior_weights[keep], new_weights])
        merged_turnover = pd.concat([prior_turnover[keep], new_turnover])
        if not prior_mu.empty or not new_mu.empty:
            merged_mu = pd.concat([prior_mu[prior_mu.index <= last_date] if not prior_mu.empty
                                   else prior_mu, new_mu])
        else:
            merged_mu = pd.DataFrame()
    else:
        merged_returns = new_returns
        merged_weights = new_weights
        merged_turnover = new_turnover
        merged_mu = new_mu

    final_last_date = merged_returns.index.max() if len(merged_returns) else last_date
    final_state = {
        'w_curr': np.asarray(w_curr, dtype=float).copy(),
        'cash': float(cash),
        'last_date': final_last_date,
        'asset_order': list(panel.asset_order),
    }

    return {
        'returns': merged_returns,
        'weights': merged_weights,
        'turnover_daily': merged_turnover,
        'rebal_dates': rebal_dates,
        'mu_forecast': merged_mu,
        'final_state': final_state,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Metrics — Table 6
# ──────────────────────────────────────────────────────────────────────────────

def portfolio_metrics(port_returns: pd.Series, rf_daily: pd.Series,
                      weights: pd.DataFrame, turnover: pd.Series) -> dict:
    """Computes Table 6 metrics for one portfolio.

    Return / Volatility / Sharpe are reported as **annualized excess** quantities
    (paper: rf=1.1%/yr).
    """
    excess = port_returns - rf_daily
    ann_ret = float(excess.mean() * 252)
    ann_vol = float(port_returns.std() * np.sqrt(252))
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0

    cum_wealth = (1 + port_returns).cumprod()
    peak = cum_wealth.cummax()
    dd = (cum_wealth - peak) / peak
    mdd = float(dd.min())

    calmar = ann_ret / abs(mdd) if mdd < 0 else 0.0

    # Annualized turnover = sum |Δw| over a year (one-way). Paper turnover = sum
    # of one-way turnover per year.
    n_years = max(1.0, len(port_returns) / 252.0)
    annual_turnover = float(turnover.sum() / n_years)

    # Average leverage (sum of risky weights, since cash = 1 - sum(w))
    leverage = float(weights.sum(axis=1).mean())

    return {
        'Return':     ann_ret,
        'Volatility': ann_vol,
        'Sharpe':     sharpe,
        'MDD':        mdd,
        'Calmar':     calmar,
        'Turnover':   annual_turnover,
        'Leverage':   leverage,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Forecast correlation (Table 7)
# ──────────────────────────────────────────────────────────────────────────────

def forecast_correlation(panel: AssetPanel, mu_forecast: pd.DataFrame,
                         horizon_days: int = 21) -> pd.Series:
    """Pearson correlation between forecasted return at each rebalance and the
    realized average return over the next `horizon_days`. Computed per asset
    plus an "Overall" pooled value.

    The paper reports per-asset and overall correlations using the forecasted
    return vector at each rebalance vs the next-period realized return.
    """
    if mu_forecast.empty:
        return pd.Series({a: np.nan for a in panel.asset_order} | {'Overall': np.nan})

    realized = pd.DataFrame(index=mu_forecast.index, columns=panel.asset_order, dtype=float)
    for d in mu_forecast.index:
        future = panel.returns.loc[d:].head(horizon_days)
        if len(future) == 0:
            continue
        realized.loc[d] = future.mean().reindex(panel.asset_order).values
    realized = realized.dropna(how='all')

    out = {}
    for a in panel.asset_order:
        f = mu_forecast[a].reindex(realized.index)
        r = realized[a]
        if f.std() == 0 or r.std() == 0:
            out[a] = 0.0
        else:
            out[a] = float(f.corr(r))
    # Pooled "Overall": stack all asset-date pairs
    f_all = mu_forecast.stack()
    r_all = realized.stack()
    f_all, r_all = f_all.align(r_all, join='inner')
    out['Overall'] = float(f_all.corr(r_all)) if len(f_all) > 5 else np.nan
    return pd.Series(out)


# ──────────────────────────────────────────────────────────────────────────────
# Top-level orchestrator
# ──────────────────────────────────────────────────────────────────────────────

STRATEGY_LABELS = ['60/40', 'MinVar', 'MinVar(JM-XGB)', 'MV', 'MV(JM-XGB)',
                   'EW', 'EW(JM-XGB)']


def run_all_portfolios(panel: AssetPanel, *,
                       rebal_freq: str = 'daily',
                       gamma_risk_minvar: float = 10.0,
                       gamma_risk_mv_baseline: float = 5.0,
                       gamma_risk_mv_jmxgb: float = 10.0,
                       gamma_trade: float = 1.0,
                       tc_oneway: float = 0.0005,
                       w_ub: float = 0.40,
                       cov_hl_days: int = 252,
                       mu_baseline_hl_years: float = 5.0,
                       mu_jmxgb_lookback_years: float = 11.0,
                       resume_from_results: Optional[Dict[str, dict]] = None,
                       progress_callback=None) -> Dict[str, dict]:
    """Run all 7 portfolios. Returns dict keyed by strategy label.

    `resume_from_results`: optional dict from a prior call (same MVO params,
    earlier oos_end). When provided, each strategy resumes from its saved
    `final_state` so the inner loop only sims dates strictly after that point.
    """
    results: Dict[str, dict] = {}

    common = {'tc': tc_oneway, 'w_ub': w_ub, 'cov_hl_days': cov_hl_days}
    sched = [
        ('60/40',           weights_60_40,           {}),
        ('MinVar',          weights_minvar_baseline, {**common, 'gamma_risk': gamma_risk_minvar}),
        ('MinVar(JM-XGB)',  weights_minvar_jmxgb,    {**common, 'gamma_risk': gamma_risk_minvar,
                                                      'gamma_trade': gamma_trade}),
        ('MV',              weights_mv_baseline,     {**common, 'gamma_risk': gamma_risk_mv_baseline,
                                                      'mu_hl_years': mu_baseline_hl_years}),
        ('MV(JM-XGB)',      weights_mv_jmxgb,        {**common, 'gamma_risk': gamma_risk_mv_jmxgb,
                                                      'gamma_trade': gamma_trade,
                                                      'mu_lookback_years': mu_jmxgb_lookback_years}),
        ('EW',              weights_ew_baseline,     {}),
        ('EW(JM-XGB)',      weights_ew_jmxgb,        {}),
    ]

    for i, (label, fn, kwargs) in enumerate(sched):
        if progress_callback is not None:
            progress_callback(i, len(sched), label)
        resume = None
        if resume_from_results is not None and label in resume_from_results:
            prev = resume_from_results[label]
            if 'final_state' in prev and prev['final_state'].get('last_date') is not None:
                # Only resume if the cached series ends before the current panel
                if prev['final_state']['last_date'] < panel.returns.index.max():
                    resume = prev
        results[label] = run_portfolio_backtest(
            panel, fn, rebal_freq=rebal_freq, tc_oneway=tc_oneway,
            resume_from=resume, **kwargs,
        )
    if progress_callback is not None:
        progress_callback(len(sched), len(sched), 'done')
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Disk-cached orchestrator (instant page load + incremental extension)
# ──────────────────────────────────────────────────────────────────────────────

def _portfolio_results_cache_path(universe: str, oos_start: str, oos_end: str,
                                  tc_mode: str, params_hash: str) -> str:
    suffix = '' if tc_mode == 'net' else f'_{tc_mode}'
    fname = (f'portfolio_results_{universe}_{oos_start[:7]}_'
             f'{oos_end[:7]}{suffix}_{params_hash}.pkl')
    return os.path.join(CACHE_DIR, fname)


def _portfolio_params_hash(rebal_freq: str, gamma_risk_minvar: float,
                           gamma_risk_mv_baseline: float, gamma_risk_mv_jmxgb: float,
                           gamma_trade: float, tc_oneway: float, w_ub: float,
                           cov_hl_days: int, mu_baseline_hl_years: float,
                           mu_jmxgb_lookback_years: float,
                           use_insample_mu: bool,
                           signal_mtime: float, insample_mu_mtime: float) -> str:
    key = repr((
        rebal_freq,
        round(float(gamma_risk_minvar), 6),
        round(float(gamma_risk_mv_baseline), 6),
        round(float(gamma_risk_mv_jmxgb), 6),
        round(float(gamma_trade), 6),
        round(float(tc_oneway), 8),
        round(float(w_ub), 6),
        int(cov_hl_days),
        round(float(mu_baseline_hl_years), 4),
        round(float(mu_jmxgb_lookback_years), 4),
        bool(use_insample_mu),
        round(float(signal_mtime), 3),
        round(float(insample_mu_mtime), 3),
    ))
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _enumerate_portfolio_results_caches(universe: str, oos_start: str,
                                        tc_mode: str, params_hash: str
                                        ) -> List[Tuple[pd.Timestamp, str, Dict[str, dict]]]:
    """Return (file_end, path, cached_dict) for every results cache that matches
    the same universe/oos_start/tc_mode/params_hash. file_end is the latest
    `last_date` across strategies in the file."""
    import glob
    suffix = '' if tc_mode == 'net' else f'_{tc_mode}'
    pattern = (f'portfolio_results_{universe}_{oos_start[:7]}_*'
               f'{suffix}_{params_hash}.pkl')
    out: List[Tuple[pd.Timestamp, str, Dict[str, dict]]] = []
    for path in glob.glob(os.path.join(CACHE_DIR, pattern)):
        try:
            with open(path, 'rb') as fh:
                cached = pickle.load(fh)
        except Exception:
            continue
        if not isinstance(cached, dict) or not cached:
            continue
        ends = [s.get('final_state', {}).get('last_date')
                for s in cached.values() if isinstance(s, dict)]
        ends = [pd.Timestamp(e) for e in ends if e is not None]
        if not ends:
            continue
        out.append((max(ends), path, cached))
    return out


def _find_extensible_portfolio_results(universe: str, oos_start: str, oos_end: str,
                                       tc_mode: str, params_hash: str
                                       ) -> Optional[Dict[str, dict]]:
    """Locate a cached results file with the same params_hash whose
    `last_date` is < target oos_end (= a candidate base for incremental
    extension). Returns the cached results dict or None."""
    target_end = pd.Timestamp(oos_end)
    candidates = [(end, path, cached) for end, path, cached
                  in _enumerate_portfolio_results_caches(universe, oos_start,
                                                        tc_mode, params_hash)
                  if end < target_end]
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][2]


def _find_covering_portfolio_results(universe: str, oos_start: str, oos_end: str,
                                     tc_mode: str, params_hash: str
                                     ) -> Optional[Dict[str, dict]]:
    """Locate a cached results file with the same params_hash whose
    `last_date` is ≥ target oos_end. Returns the cached (untrimmed) dict so
    the caller can slice it; prefers the smallest covering cache."""
    target_end = pd.Timestamp(oos_end)
    candidates = [(end, path, cached) for end, path, cached
                  in _enumerate_portfolio_results_caches(universe, oos_start,
                                                        tc_mode, params_hash)
                  if end >= target_end - pd.Timedelta(days=1)]
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][2]


def _trim_portfolio_results(results: Dict[str, dict],
                            target_end: pd.Timestamp) -> Dict[str, dict]:
    """Slice each strategy's per-day series down to dates ≤ target_end.

    `final_state` is intentionally invalidated (cash=None, last_date=None) so
    the trimmed dict is NOT used as a resume base — `cash` drifts with rf
    accrual and TC, and we can't reconstruct it from the trimmed tail."""
    target_end = pd.Timestamp(target_end)
    out: Dict[str, dict] = {}
    for label, res in results.items():
        if not isinstance(res, dict):
            out[label] = res
            continue

        def _slice(obj):
            if isinstance(obj, (pd.Series, pd.DataFrame)) and len(obj):
                return obj[obj.index <= target_end]
            return obj

        trimmed_returns  = _slice(res.get('returns'))
        trimmed_weights  = _slice(res.get('weights'))
        trimmed_turnover = _slice(res.get('turnover_daily'))
        trimmed_mu       = _slice(res.get('mu_forecast'))
        rebal_dates = res.get('rebal_dates', [])
        if hasattr(rebal_dates, '__iter__'):
            trimmed_rebal = type(rebal_dates)(
                d for d in rebal_dates if pd.Timestamp(d) <= target_end)
        else:
            trimmed_rebal = rebal_dates
        out[label] = {
            'returns':        trimmed_returns,
            'weights':        trimmed_weights,
            'turnover_daily': trimmed_turnover,
            'rebal_dates':    trimmed_rebal,
            'mu_forecast':    trimmed_mu,
            'final_state': {'w_curr': None, 'cash': None, 'last_date': None,
                            'asset_order': res.get('final_state', {}).get('asset_order')},
        }
    return out


def mvo_results_cache_will_hit(universe: str, oos_start: str, oos_end: str,
                               tc_mode: str, use_insample_mu: bool,
                               signal_mtime: float, insample_mu_mtime: float,
                               **mvo_params) -> bool:
    """Return True iff `run_all_portfolios_cached` would serve this request
    from the on-disk cache (exact hit or covering trim, no compute). Use this
    to decide whether to show a "Solving MVO portfolios…" spinner."""
    try:
        params_hash = _portfolio_params_hash(
            use_insample_mu=use_insample_mu,
            signal_mtime=signal_mtime, insample_mu_mtime=insample_mu_mtime,
            **mvo_params,
        )
    except (KeyError, TypeError):
        return False
    exact_path = _portfolio_results_cache_path(universe, oos_start, oos_end,
                                                tc_mode, params_hash)
    if os.path.exists(exact_path):
        return True
    return _find_covering_portfolio_results(
        universe, oos_start, oos_end, tc_mode, params_hash) is not None


def run_all_portfolios_cached(panel: AssetPanel, *,
                              universe: str, oos_start: str, oos_end: str,
                              tc_mode: str, use_insample_mu: bool,
                              signal_mtime: float, insample_mu_mtime: float,
                              progress_callback=None,
                              **mvo_params) -> Dict[str, dict]:
    """Disk-cached, optionally incremental wrapper around `run_all_portfolios`.

    Loading paths (fastest first):
      1. Exact hash hit at the target oos_end → return cached dict (instant).
      2. Same hash, earlier oos_end → load and resume each strategy from its
         saved `final_state`; only simulate dates strictly after that point.
      3. Otherwise → full `run_all_portfolios` from scratch.

    The result is persisted under the target oos_end's cache file.
    """
    params_hash = _portfolio_params_hash(
        rebal_freq=mvo_params['rebal_freq'],
        gamma_risk_minvar=mvo_params['gamma_risk_minvar'],
        gamma_risk_mv_baseline=mvo_params['gamma_risk_mv_baseline'],
        gamma_risk_mv_jmxgb=mvo_params['gamma_risk_mv_jmxgb'],
        gamma_trade=mvo_params['gamma_trade'],
        tc_oneway=mvo_params['tc_oneway'],
        w_ub=mvo_params['w_ub'],
        cov_hl_days=mvo_params['cov_hl_days'],
        mu_baseline_hl_years=mvo_params['mu_baseline_hl_years'],
        mu_jmxgb_lookback_years=mvo_params['mu_jmxgb_lookback_years'],
        use_insample_mu=use_insample_mu,
        signal_mtime=signal_mtime,
        insample_mu_mtime=insample_mu_mtime,
    )
    cache_path = _portfolio_results_cache_path(universe, oos_start, oos_end,
                                               tc_mode, params_hash)

    # 1) Exact hit
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as fh:
                cached = pickle.load(fh)
            if isinstance(cached, dict) and cached:
                return cached
        except Exception:
            pass

    # 1b) Covering cache (same hash, end ≥ target): trim and return.
    # NOT persisted to disk — `final_state` cannot be reconstructed from a
    # trimmed tail, so the original longer cache stays the resume base.
    covering = _find_covering_portfolio_results(
        universe, oos_start, oos_end, tc_mode, params_hash)
    if covering is not None:
        return _trim_portfolio_results(covering, pd.Timestamp(oos_end))

    # 2) Extensible base (same hash, earlier end)
    resume_from = _find_extensible_portfolio_results(
        universe, oos_start, oos_end, tc_mode, params_hash)

    # 3) Run (full or incremental)
    results = run_all_portfolios(
        panel, resume_from_results=resume_from,
        progress_callback=progress_callback, **mvo_params)

    # Persist
    try:
        with open(cache_path, 'wb') as fh:
            pickle.dump(results, fh)
    except Exception as e:
        print(f'[portfolio] failed to cache results: {e}')

    return results


# Paper Table 6 reference
PAPER_TABLE_6 = pd.DataFrame({
    '60/40':           [0.050, 0.089, 0.57, -0.315, 0.16, 0.74, 1.00],
    'MinVar':          [0.028, 0.040, 0.70, -0.193, 0.15, 0.49, 1.00],
    'MinVar(JM-XGB)':  [0.039, 0.035, 1.12, -0.071, 0.55, 2.06, 0.91],
    'MV':              [0.026, 0.071, 0.37, -0.256, 0.10, 3.40, 0.95],
    'MV(JM-XGB)':      [0.089, 0.087, 1.02, -0.135, 0.66, 9.12, 0.86],
    'EW':              [0.055, 0.108, 0.51, -0.375, 0.15, 0.81, 1.00],
    'EW(JM-XGB)':      [0.082, 0.090, 0.91, -0.176, 0.47, 11.70, 0.92],
}, index=['Return', 'Volatility', 'Sharpe', 'MDD', 'Calmar', 'Turnover', 'Leverage'])

# Paper Table 7 reference
PAPER_TABLE_7 = pd.DataFrame({
    'EWMA':   [-0.0104, -0.0158, -0.0386, -0.0372, -0.0373, -0.0203, -0.0509,
                0.0125, -0.0117, -0.0016, -0.0006, -0.0105, -0.0159],
    'JM-XGB': [ 0.0243,  0.0166,  0.0090,  0.0103,  0.0453,  0.0602,  0.0210,
                0.0322,  0.0164,  0.1054,  0.0262,  0.0339,  0.0032],
}, index=['Overall', 'LargeCap', 'MidCap', 'SmallCap', 'EAFE', 'EM', 'REIT',
          'AggBond', 'Treasury', 'HighYield', 'Corporate', 'Commodity', 'Gold'])
