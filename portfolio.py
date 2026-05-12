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


# ──────────────────────────────────────────────────────────────────────────────
# Per-asset signal computation (heavy step — cached to disk)
# ──────────────────────────────────────────────────────────────────────────────

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


def compute_asset_signals(universe: str = 'bloomberg',
                          oos_start: str = '2007-01-01',
                          oos_end: str = '2023-12-31',
                          force_refresh: bool = False,
                          tc_mode: str = 'net',
                          progress_callback=None) -> Dict[str, pd.DataFrame]:
    """Run walk-forward backtest for each of the 12 assets and return a dict
    keyed by asset name. Each value is a DataFrame containing at least:
        Target_Return, RF_Rate, Forecast_State, Raw_Prob (when JM-XGB),
        and lambda_history attribute under .attrs.

    tc_mode='net'  → walk-forward tunes λ on net-of-TC Sharpe (5 bps default).
    tc_mode='gross' → tunes on gross Sharpe (TC=0) — apples-to-apples with paper
                      Tables 4/6/7 (gross-of-TC).  Forecasts differ from 'net'
                      because λ choices differ, so this caches separately.

    Cached on disk under cache/portfolio_signals_<universe>_<oos>[_gross].pkl.
    """
    if tc_mode not in ('net', 'gross'):
        raise ValueError(f"tc_mode must be 'net' or 'gross', got {tc_mode!r}")
    cache_path = _signal_cache_path(universe, oos_start, oos_end, tc_mode=tc_mode)

    if not force_refresh and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as fh:
                signals = pickle.load(fh)
            if all(isinstance(v, pd.DataFrame) for v in signals.values()):
                return signals
        except Exception:
            pass  # fall through to recompute

    import main as _main
    from config import StrategyConfig

    # Configure pipeline-level constants
    _main.OOS_START_DATE = oos_start
    _main.END_DATE = oos_end
    _main.LAMBDA_GRID = list(GRID_8PT)
    # Wide enough to give the 11-year JM lookback room
    _main.START_DATE_DATA = '1991-01-01' if universe == 'bloomberg' else '1996-01-01'

    # tc_mode='gross': zero out TC during walk-forward so λ-selection uses gross
    # Sharpe (matches paper Table 4). Restored in the finally block.
    original_tc = _main.TRANSACTION_COST
    if tc_mode == 'gross':
        _main.TRANSACTION_COST = 0.0

    if universe == 'bloomberg':
        asset_specs = BBG_ASSETS
    elif universe == 'yahoo':
        asset_specs = YAHOO_ASSETS
    else:
        raise ValueError(f'Unknown universe: {universe!r}')

    signals: Dict[str, pd.DataFrame] = {}

    try:
        for i, spec in enumerate(asset_specs):
            if universe == 'bloomberg':
                asset_name, bbg_col, hl_proxy, include_dd = spec
                try:
                    df_feat = _build_bbg_features(bbg_col, include_dd=include_dd)
                except Exception as e:
                    print(f'[portfolio] {asset_name}: feature build failed — {e}')
                    continue
            else:
                asset_name, ticker, hl_proxy, include_dd = spec
                try:
                    df_feat = _build_yahoo_features(ticker, _main.START_DATE_DATA, oos_end,
                                                    include_dd=include_dd)
                except Exception as e:
                    print(f'[portfolio] {asset_name}: feature build failed — {e}')
                    continue

            # Drive PAPER_EWMA_HL lookup via TARGET_TICKER
            _main.TARGET_TICKER = hl_proxy
            _main._forecast_cache.clear()

            cfg = StrategyConfig(name=f'Portfolio_{universe}_{asset_name}',
                                 ewma_mode='paper')
            if progress_callback is not None:
                progress_callback(i, len(asset_specs), asset_name)

            try:
                r = _main.walk_forward_backtest(df_feat, cfg)
            except Exception as e:
                print(f'[portfolio] {asset_name}: walk_forward failed — {e}')
                continue

            if r is None or r.empty:
                print(f'[portfolio] {asset_name}: empty result')
                continue

            # Attach JM-only Forecast_State for later regime-conditional μ computation
            # (We re-run JM at each biannual anchor using the selected λ trace.)
            r.attrs['hl_proxy'] = hl_proxy
            r.attrs['include_dd'] = include_dd
            r.attrs['tc_mode'] = tc_mode
            if universe == 'bloomberg':
                r.attrs['source_col'] = bbg_col
            else:
                r.attrs['source_ticker'] = ticker

            signals[asset_name] = r
    finally:
        _main.TRANSACTION_COST = original_tc

    # Cache
    try:
        with open(cache_path, 'wb') as fh:
            pickle.dump(signals, fh)
    except Exception as e:
        print(f'[portfolio] failed to cache signals: {e}')

    if progress_callback is not None:
        progress_callback(len(asset_specs), len(asset_specs), 'done')
    return signals


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

def compute_insample_regime_means(universe: str,
                                  oos_start: str = '2007-01-01',
                                  oos_end: str = '2023-12-31',
                                  force_refresh: bool = False,
                                  tc_mode: str = 'net',
                                  progress_callback=None) -> Dict[str, pd.DataFrame]:
    """For each asset, at each biannual rebalance anchor, fit JM on the 11-year
    lookback window with the asset's selected λ and compute in-sample
    regime-conditional means.

    Returns:
        dict {asset_name: DataFrame(index=anchor_date,
                                   columns=['mu_bull', 'mu_bear'])}

    The means represent the mean **daily total return** for bullish (state 0)
    and bearish (state 1) days as labeled by the in-sample JM fit. This
    matches paper Section 4.5's spec for MV(JM-XGB).

    tc_mode is propagated to the signal lookup so λ traces match the
    requested TC regime; cache is keyed by mode.
    """
    if tc_mode not in ('net', 'gross'):
        raise ValueError(f"tc_mode must be 'net' or 'gross', got {tc_mode!r}")
    cache_path = _insample_mu_cache_path(universe, oos_start, oos_end, tc_mode=tc_mode)

    if not force_refresh and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as fh:
                cached = pickle.load(fh)
            if all(isinstance(v, pd.DataFrame) for v in cached.values()):
                return cached
        except Exception:
            pass

    import main as _main
    from config import StrategyConfig

    sig_path = _signal_cache_path(universe, oos_start, oos_end, tc_mode=tc_mode)
    if not os.path.exists(sig_path):
        raise FileNotFoundError(f'Signals cache not found: {sig_path}. '
                                f'Run compute_asset_signals(tc_mode={tc_mode!r}) first.')
    with open(sig_path, 'rb') as fh:
        signals = pickle.load(fh)

    _main.OOS_START_DATE = oos_start
    _main.END_DATE = oos_end
    _main.LAMBDA_GRID = list(GRID_8PT)
    _main.START_DATE_DATA = '1991-01-01' if universe == 'bloomberg' else '1996-01-01'

    asset_specs = BBG_ASSETS if universe == 'bloomberg' else YAHOO_ASSETS

    out: Dict[str, pd.DataFrame] = {}
    for i, spec in enumerate(asset_specs):
        if universe == 'bloomberg':
            asset_name, bbg_col, hl_proxy, include_dd = spec
            df_feat = _build_bbg_features(bbg_col, include_dd=include_dd)
        else:
            asset_name, ticker, hl_proxy, include_dd = spec
            df_feat = _build_yahoo_features(ticker, _main.START_DATE_DATA, oos_end,
                                            include_dd=include_dd)

        if asset_name not in signals:
            continue
        if progress_callback is not None:
            progress_callback(i, len(asset_specs), asset_name)

        lambda_history = signals[asset_name].attrs.get('lambda_history', [])
        lambda_dates   = pd.to_datetime(signals[asset_name].attrs.get('lambda_dates', []))
        if len(lambda_history) == 0:
            continue

        return_features = [c for c in df_feat.columns
                           if c.startswith(('DD_', 'Avg_Ret_', 'Sortino_'))]

        rows = []
        for anchor, lmbda in zip(lambda_dates, lambda_history):
            train_start = anchor - pd.DateOffset(years=11)
            tr = df_feat[(df_feat.index >= train_start) & (df_feat.index < anchor)].copy()
            if len(tr) < 252 * 5:
                continue
            X = tr[return_features]
            X = (X - X.mean()) / X.std()
            jm = _main.StatisticalJumpModel(n_states=2, lambda_penalty=float(lmbda))
            states = jm.fit_predict(X.values)
            # Align so state 0 = bullish (higher cumulative excess return)
            cum0 = tr['Excess_Return'][states == 0].sum()
            cum1 = tr['Excess_Return'][states == 1].sum()
            if cum1 > cum0:
                states = 1 - states
            # Paper Section 4.1: μ is forecast for *excess* returns.
            mu_bull = float(tr['Excess_Return'][states == 0].mean()) if (states == 0).any() else 0.0
            mu_bear = float(tr['Excess_Return'][states == 1].mean()) if (states == 1).any() else 0.0
            rows.append({'anchor': anchor, 'mu_bull': mu_bull, 'mu_bear': mu_bear})

        if rows:
            df_out = pd.DataFrame(rows).set_index('anchor')
            out[asset_name] = df_out

    if progress_callback is not None:
        progress_callback(len(asset_specs), len(asset_specs), 'done')

    try:
        with open(cache_path, 'wb') as fh:
            pickle.dump(out, fh)
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
    """
    returns = panel.returns.copy()
    rf_daily = panel.rf_daily.copy()
    n_assets = returns.shape[1]

    rebal_dates = _rebalance_dates(returns.index, rebal_freq)

    w_curr = np.zeros(n_assets)
    cash = 1.0
    port_rets = []
    weight_log = []
    turnover_log = []

    mu_forecast_log = {}

    for t, date in enumerate(returns.index):
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

    return {
        'returns': pd.Series(port_rets, index=returns.index, name='Portfolio_Return'),
        'weights': pd.DataFrame(weight_log, index=returns.index, columns=panel.asset_order),
        'turnover_daily': pd.Series(turnover_log, index=returns.index),
        'rebal_dates': rebal_dates,
        'mu_forecast': pd.DataFrame(mu_forecast_log).T if mu_forecast_log else pd.DataFrame(),
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
                       progress_callback=None) -> Dict[str, dict]:
    """Run all 7 portfolios. Returns dict keyed by strategy label."""
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
        results[label] = run_portfolio_backtest(
            panel, fn, rebal_freq=rebal_freq, tc_oneway=tc_oneway, **kwargs,
        )
    if progress_callback is not None:
        progress_callback(len(sched), len(sched), 'done')
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
