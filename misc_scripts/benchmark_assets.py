"""
Multi-Asset Benchmark: JM-XGB Strategy vs Buy & Hold
Tests across 12 ETFs and multiple time periods.

Optimizations for speed (methodology preserved):
- Skip SHAP computation (not needed for performance evaluation)
- Reduced lambda grid (5 candidates instead of 11)
- Full EWMA halflife grid (4 candidates)
- No PDF/chart generation
- Multiprocessing across assets
"""

import sys
import os
import types

# Compatibility shim for distutils.version (Python 3.12+)
try:
    import distutils
    import distutils.version
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
import yfinance as yf
from xgboost import XGBClassifier
import pandas_datareader.data as web
import warnings
from multiprocessing import Pool, cpu_count
import time
from datetime import datetime
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import StrategyConfig

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, 'cache')
BENCHMARKS_DIR = os.path.join(PROJECT_ROOT, 'benchmarks')
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(BENCHMARKS_DIR, exist_ok=True)

# ── Asset List Loading ────────────────────────────────────────────────────────

ASSET_LISTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'asset_lists.md')
DEFAULT_LIST_NAME = 'Default ETFs'


def load_asset_lists(md_path=ASSET_LISTS_PATH):
    """Parse asset_lists.md -> dict[name -> {tickers, asset_classes, data_start}]."""
    lists = {}
    current_name = None
    current_tickers = []
    current_classes = {}
    current_data_start = None

    with open(md_path, 'r') as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith('## '):
                if current_name and current_tickers:
                    lists[current_name] = {
                        'tickers': current_tickers,
                        'asset_classes': current_classes,
                        'data_start': current_data_start,
                    }
                current_name = stripped[3:].strip()
                current_tickers = []
                current_classes = {}
                current_data_start = None
                continue

            if stripped.lower().startswith('data_start:'):
                current_data_start = stripped.split(':', 1)[1].strip()
                continue

            if stripped.startswith('|'):
                cells = [c.strip() for c in stripped.split('|')[1:-1]]
                if not cells or len(cells) < 2:
                    continue
                ticker = cells[0]
                if ticker.lower() == 'ticker' or ticker.startswith('---') or ticker.startswith(':---'):
                    continue
                if set(ticker.replace('-', '')) == set():
                    continue
                asset_class = cells[1]
                current_tickers.append(ticker)
                if asset_class not in current_classes:
                    current_classes[asset_class] = []
                current_classes[asset_class].append(ticker)

    if current_name and current_tickers:
        lists[current_name] = {
            'tickers': current_tickers,
            'asset_classes': current_classes,
            'data_start': current_data_start,
        }

    return lists


def parse_asset_list_selection(args, all_lists):
    """Parse CLI args to select an asset list. Returns list name or None (exit)."""
    if not args:
        return DEFAULT_LIST_NAME

    arg = ' '.join(args).strip()

    if arg.lower() in ('list', 'ls'):
        print("Available asset lists:")
        for name, info in all_lists.items():
            print(f"  {name!r}  ({len(info['tickers'])} tickers, data_start={info['data_start']})")
        return None

    if arg in ('-h', '--help', 'help'):
        print("Usage: python misc_scripts/benchmark_assets.py [LIST_NAME | list | --help]")
        print()
        print("  (no argument)       Use 'Default ETFs' list")
        print("  \"Long History\"      Use 'Long History' list")
        print("  list                Show all available asset lists")
        return None

    if arg in all_lists:
        return arg

    print(f"Error: asset list {arg!r} not found.")
    print("Available lists:", ', '.join(f"'{n}'" for n in all_lists))
    return None


# ── Configuration ─────────────────────────────────────────────────────────────

# Time periods to test (label, oos_start, oos_end)
TIME_PERIODS = [
    ('2007-2009 (GFC)',       '2007-01-01', '2010-01-01'),
    ('2010-2015 (Recovery)',  '2010-01-01', '2016-01-01'),
    ('2016-2019 (Late Cycle)','2016-01-01', '2020-01-01'),
    ('2020-2021 (COVID)',     '2020-01-01', '2022-01-01'),
    ('2022-2025 (Post-COVID)','2022-01-01', '2026-01-01'),
    ('Full (2007-2025)',      '2007-01-01', '2026-01-01'),
]

RISK_FREE_TICKER = '^IRX'
BOND_TICKER = 'VBMFX'
VIX_TICKER = '^VIX'
LARGECAP_TICKER = '^SP500TR'  # Paper Table 3: Stock-Bond Corr uses LargeCap vs AggBond for ALL assets
DATA_START = '1993-01-01'  # ETFs need less lookback than ^SP500TR

TRANSACTION_COST = 0.0005
LAMBDA_GRID = [0.0, 3.0, 10.0, 30.0, 100.0]  # 5 candidates (speed; finer grids overfit validation window)
EWMA_HL_GRID = [0, 2, 4, 8]  # 4 candidates (asset-specific smoothing)
VALIDATION_WINDOW_YRS = 5

# Paper Table 2: bonds and gold use only 6 return features (Avg_Return×3, Sortino×3),
# excluding Downside Deviation (DD_log_5, DD_log_21) which are equity-specific.
DD_EXCLUDE_TICKERS = {'AGG', 'VBMFX',        # AggBond proxies
                      'SPTL', 'VUSTX', 'IEF', 'TLT', 'VGLT',  # Treasury proxies
                      'GLD', 'GC=F', 'IAU'}   # Gold proxies

# ── Statistical Jump Model (same as main.py) ─────────────────────────────────

class StatisticalJumpModel:
    def __init__(self, n_states=2, lambda_penalty=10.0, max_iter=20):
        self.n_states = n_states
        self.lambda_penalty = lambda_penalty
        self.max_iter = max_iter
        self.means = None

    def fit_predict(self, X):
        X_arr = np.array(X)
        n_samples, n_features = X_arr.shape
        np.random.seed(42)
        idx = np.random.choice(n_samples, self.n_states, replace=False)
        self.means = X_arr[idx].copy()
        states = np.zeros(n_samples, dtype=int)

        for _ in range(self.max_iter):
            distances = 0.5 * np.sum((X_arr[:, None, :] - self.means[None, :, :])**2, axis=2)
            cost_matrix = np.zeros((n_samples, self.n_states))
            back_pointers = np.zeros((n_samples, self.n_states), dtype=int)
            cost_matrix[0] = distances[0]

            for t in range(1, n_samples):
                c00 = cost_matrix[t-1, 0]
                c10 = cost_matrix[t-1, 1] + self.lambda_penalty
                if c00 <= c10:
                    cost_matrix[t, 0] = c00 + distances[t, 0]
                    back_pointers[t, 0] = 0
                else:
                    cost_matrix[t, 0] = c10 + distances[t, 0]
                    back_pointers[t, 0] = 1
                c01 = cost_matrix[t-1, 0] + self.lambda_penalty
                c11 = cost_matrix[t-1, 1]
                if c01 <= c11:
                    cost_matrix[t, 1] = c01 + distances[t, 1]
                    back_pointers[t, 1] = 0
                else:
                    cost_matrix[t, 1] = c11 + distances[t, 1]
                    back_pointers[t, 1] = 1

            new_states = np.zeros(n_samples, dtype=int)
            new_states[-1] = np.argmin(cost_matrix[-1])
            for t in range(n_samples - 2, -1, -1):
                new_states[t] = back_pointers[t+1, new_states[t+1]]

            for k in range(self.n_states):
                mask = (new_states == k)
                if np.sum(mask) > 0:
                    self.means[k] = np.mean(X_arr[mask], axis=0)
            if np.array_equal(states, new_states):
                break
            states = new_states
        return states

    def predict_online(self, X, last_known_state):
        X_arr = np.array(X)
        n_samples = X_arr.shape[0]
        states = np.zeros(n_samples, dtype=int)
        prev_state = last_known_state
        for t in range(n_samples):
            dist = 0.5 * np.sum((X_arr[t, None, :] - self.means)**2, axis=1)
            penalty = np.full(self.n_states, self.lambda_penalty)
            penalty[prev_state] = 0.0
            costs = dist + penalty
            current_state = np.argmin(costs)
            states[t] = current_state
            prev_state = current_state
        return states

# ── Data Fetching ─────────────────────────────────────────────────────────────

def fetch_etf_data(ticker, data_start=None):
    """Fetch and prepare features for a single ETF. Returns (ticker, df) or (ticker, None)."""
    if data_start is None:
        data_start = DATA_START
    exclude_dd = ticker in DD_EXCLUDE_TICKERS
    cache_suffix = '_noDD' if exclude_dd else ''
    cache_file = os.path.join(CACHE_DIR, f'data_cache_{ticker}_{data_start.replace("-", "")}{cache_suffix}_v2.pkl')
    if os.path.exists(cache_file):
        return ticker, pd.read_pickle(cache_file)

    try:
        # Download ETF + auxiliary tickers.
        # Always include LARGECAP_TICKER so Stock_Bond_Corr = corr(LargeCap, AggBond) for all assets.
        tickers_to_fetch = dict.fromkeys([ticker, BOND_TICKER, RISK_FREE_TICKER, VIX_TICKER, LARGECAP_TICKER])
        raw = {}
        for t in tickers_to_fetch:
            df_t = yf.download(t, start=data_start, end='2026-03-01', auto_adjust=False, progress=False)
            if df_t.empty:
                continue
            if isinstance(df_t.columns, pd.MultiIndex):
                if 'Adj Close' in df_t.columns.get_level_values(0):
                    s = df_t['Adj Close'].iloc[:, 0].rename(t)
                else:
                    s = df_t.iloc[:, 0].rename(t)
            else:
                s = df_t.get('Adj Close', df_t.get('Close', df_t.iloc[:, 0])).rename(t)
            raw[t] = s

        if ticker not in raw or RISK_FREE_TICKER not in raw or VIX_TICKER not in raw:
            return ticker, None

        data = pd.concat(raw.values(), axis=1).ffill().dropna()

        # FRED macro data
        fred = web.DataReader(['DGS2', 'DGS10'], 'fred', data_start, '2026-03-01').ffill().dropna()
        df = data.join(fred, how='inner')

        if len(df) < 252 * 5:
            return ticker, None

        features = pd.DataFrame(index=df.index)
        target_returns = df[ticker].pct_change().fillna(0)
        features['Target_Return'] = target_returns
        features['RF_Rate'] = (df[RISK_FREE_TICKER] / 100) / 252
        features['Excess_Return'] = target_returns - features['RF_Rate']

        downside_returns = np.minimum(features['Excess_Return'], 0)
        if not exclude_dd:
            for hl in [5, 21]:
                ewm_var = (downside_returns ** 2).ewm(halflife=hl).mean()
                features[f'DD_log_{hl}'] = np.log(np.sqrt(ewm_var).fillna(0) + 1e-8)
        for hl in [5, 10, 21]:
            features[f'Avg_Ret_{hl}'] = features['Excess_Return'].ewm(halflife=hl).mean()
        for hl in [5, 10, 21]:
            ewm_var = (downside_returns ** 2).ewm(halflife=hl).mean()
            ewm_dd_raw = np.maximum(np.sqrt(ewm_var).fillna(1e-8), 1e-8)
            features[f'Sortino_{hl}'] = (features[f'Avg_Ret_{hl}'] / ewm_dd_raw).clip(-10, 10)

        features['Yield_2Y_EWMA_diff'] = df['DGS2'].diff().fillna(0).ewm(halflife=21).mean()
        slope = df['DGS10'] - df['DGS2']
        features['Yield_Slope_EWMA_10'] = slope.ewm(halflife=10).mean()
        features['Yield_Slope_EWMA_diff_21'] = slope.diff().fillna(0).ewm(halflife=21).mean()
        features['VIX_EWMA_log_diff'] = np.log(df[VIX_TICKER] / df[VIX_TICKER].shift(1)).fillna(0).ewm(halflife=63).mean()

        # Paper Table 3: Stock-Bond Corr = rolling corr(LargeCap_ret, AggBond_ret), 252-day lookback.
        # This is a macro feature identical for all assets — NOT corr(target, bond).
        if BOND_TICKER in df.columns and LARGECAP_TICKER in df.columns:
            largecap_rets = df[LARGECAP_TICKER].pct_change().fillna(0)
            bond_rets = df[BOND_TICKER].pct_change().fillna(0)
            features['Stock_Bond_Corr'] = largecap_rets.rolling(window=252).corr(bond_rets).fillna(0)
        else:
            features['Stock_Bond_Corr'] = 0.0

        final = features.dropna()
        final.to_pickle(cache_file)
        return ticker, final

    except Exception as e:
        print(f"  [WARN] Failed to fetch {ticker}: {e}")
        return ticker, None

# ── Core Backtest Logic (fast version, no SHAP) ──────────────────────────────

def run_period_forecast_fast(df, current_date, lambda_penalty, cache, config: StrategyConfig = None, include_xgboost=True):
    if config is None:
        config = StrategyConfig()
        
    cache_key = (current_date, lambda_penalty, include_xgboost, config.name)
    if cache_key in cache:
        return cache[cache_key]

    train_start = current_date - pd.DateOffset(years=11)
    train_df = df[(df.index >= train_start) & (df.index < current_date)].copy()
    if len(train_df) < 252 * 3:  # relaxed for shorter ETF histories
        cache[cache_key] = None
        return None

    return_features = [c for c in train_df.columns if c.startswith(('DD_', 'Avg_Ret_', 'Sortino_'))]
    X_train_jm = train_df[return_features]
    std = X_train_jm.std()
    std[std == 0] = 1.0
    X_train_jm = (X_train_jm - X_train_jm.mean()) / std

    jm = StatisticalJumpModel(n_states=2, lambda_penalty=lambda_penalty)
    identified_states = jm.fit_predict(X_train_jm.values)

    cum_ret_0 = train_df['Excess_Return'][identified_states == 0].sum()
    cum_ret_1 = train_df['Excess_Return'][identified_states == 1].sum()
    if cum_ret_1 > cum_ret_0:
        identified_states = 1 - identified_states
        jm.means = jm.means[::-1].copy()

    train_df['Target_State'] = np.roll(identified_states, -1)
    train_df = train_df.iloc[:-1]

    oos_end = current_date + pd.DateOffset(months=6)
    oos_df = df[(df.index >= current_date) & (df.index < oos_end)].copy()
    if len(oos_df) == 0:
        cache[cache_key] = None
        return None

    if not include_xgboost:
        X_oos_jm = oos_df[return_features]
        train_std = train_df[return_features].std()
        train_std[train_std == 0] = 1.0
        X_oos_jm = (X_oos_jm - train_df[return_features].mean()) / train_std
        oos_states = jm.predict_online(X_oos_jm.values, last_known_state=identified_states[-1])
        oos_df['Forecast_State'] = oos_states
        result = oos_df[['Target_Return', 'RF_Rate', 'Forecast_State']]
        cache[cache_key] = result
        return result

    macro_features = ['Yield_2Y_EWMA_diff', 'Yield_Slope_EWMA_10',
                      'Yield_Slope_EWMA_diff_21', 'VIX_EWMA_log_diff', 'Stock_Bond_Corr']
    all_features = return_features + macro_features

    X_train_xgb = train_df[all_features]
    y_train_xgb = train_df['Target_State']
    X_oos_xgb = oos_df[all_features]

    # Edge case: if JM assigned all data to one state, XGB can't train a classifier
    unique_labels = y_train_xgb.unique()
    if len(unique_labels) < 2:
        # All one class — predict that class uniformly
        only_class = unique_labels[0]
        oos_probs = np.full(len(X_oos_xgb), float(only_class))
        oos_df['Raw_Prob'] = oos_probs
        result = oos_df[['Target_Return', 'RF_Rate', 'Raw_Prob']]
        cache[cache_key] = result
        return result

    xgb = XGBClassifier(
        eval_metric='logloss', random_state=42, verbosity=0, **config.xgb_params
    )
    xgb.fit(X_train_xgb, y_train_xgb)
    oos_probs = xgb.predict_proba(X_oos_xgb)[:, 1]
    oos_df['Raw_Prob'] = oos_probs
    result = oos_df[['Target_Return', 'RF_Rate', 'Raw_Prob']]
    cache[cache_key] = result
    return result


def simulate_strategy_fast(df, start_date, end_date, lambda_penalty, cache, config: StrategyConfig = None, include_xgboost=True, ewma_halflife=8):
    if config is None:
        config = StrategyConfig()
        
    results = []
    current_date = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    while current_date < end_dt:
        res = run_period_forecast_fast(df, current_date, lambda_penalty, cache, config, include_xgboost)
        if res is not None:
            results.append(res)
        current_date += pd.DateOffset(months=6)
    if not results:
        return pd.DataFrame()
    full_res = pd.concat(results)

    if include_xgboost:
        if ewma_halflife == 0:
            full_res['State_Prob'] = full_res['Raw_Prob']
        else:
            full_res['State_Prob'] = full_res['Raw_Prob'].ewm(halflife=ewma_halflife).mean()
            
        if config.allocation_style == "binary":
            full_res['Forecast_State'] = (full_res['State_Prob'] > config.prob_threshold).astype(int)
            trading_signals = full_res['Forecast_State'].shift(1).fillna(0)
            alloc_target = 1.0 - trading_signals
        elif config.allocation_style == "continuous":
            alloc_target = (1.0 - full_res['State_Prob']).shift(1).fillna(1.0)
    else:
        trading_signals = full_res['Forecast_State'].shift(1).fillna(0)
        alloc_target = 1.0 - trading_signals

    strat_returns = (alloc_target * full_res['Target_Return']) + ((1.0 - alloc_target) * full_res['RF_Rate'])
    trades = alloc_target.diff().abs().fillna(0)
    
    full_res['Strat_Return'] = strat_returns - (trades * TRANSACTION_COST)
    return full_res


def calculate_metrics(returns_series, rf_series):
    if len(returns_series) < 10:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    excess_returns = returns_series - rf_series
    ann_excess_ret = excess_returns.mean() * 252
    ann_vol = returns_series.std() * np.sqrt(252)
    sharpe = ann_excess_ret / ann_vol if ann_vol != 0 else 0.0

    downside_returns = np.minimum(excess_returns, 0)
    ann_downside_vol = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(252)
    sortino = ann_excess_ret / ann_downside_vol if ann_downside_vol != 0 else 0.0

    cum_wealth = (1 + returns_series).cumprod()
    peak = cum_wealth.cummax()
    drawdown = (cum_wealth - peak) / peak
    mdd = drawdown.min()

    n_days = len(returns_series)
    ann_geom_ret = cum_wealth.iloc[-1] ** (252 / n_days) - 1 if n_days > 0 else 0.0
    return ann_geom_ret, ann_vol, sharpe, sortino, mdd


# ── Single-Asset Full Backtest ────────────────────────────────────────────────

def backtest_single_asset(args):
    """Run full walk-forward backtest for one ETF. Returns list of result dicts."""
    ticker, df, config, data_start = args
    results = []
    cache = {}  # per-asset forecast cache

    for period_name, oos_start, oos_end in TIME_PERIODS:
        oos_start_dt = pd.to_datetime(oos_start)
        oos_end_dt = pd.to_datetime(oos_end)

        if df.index[0] > oos_start_dt - pd.DateOffset(years=3):
            results.append({'Ticker': ticker, 'Period': period_name, 'Strategy': 'JM-XGB', 'Ann_Ret': np.nan, 'Ann_Vol': np.nan, 'Sharpe': np.nan, 'Sortino': np.nan, 'Max_DD': np.nan})
            results.append({'Ticker': ticker, 'Period': period_name, 'Strategy': 'B&H', 'Ann_Ret': np.nan, 'Ann_Vol': np.nan, 'Sharpe': np.nan, 'Sortino': np.nan, 'Max_DD': np.nan})
            continue

        # Phase 1: Tune EWMA halflife on initial validation window
        init_val_start = oos_start_dt - pd.DateOffset(years=VALIDATION_WINDOW_YRS)
        if config.validation_window_type == 'expanding':
            init_val_start = pd.to_datetime(data_start)
            
        best_ewma_hl = EWMA_HL_GRID[-1]
        best_init_metric = -np.inf
        for hl in EWMA_HL_GRID:
            for lmbda in LAMBDA_GRID:
                val_res = simulate_strategy_fast(df, init_val_start, oos_start_dt, lmbda, cache, config, include_xgboost=True, ewma_halflife=hl)
                if not val_res.empty:
                    _, _, sharpe, sortino, _ = calculate_metrics(val_res['Strat_Return'], val_res['RF_Rate'])
                    metric_val = sortino if config.tuning_metric == 'sortino' else sharpe
                    if not np.isnan(metric_val) and metric_val > best_init_metric:
                        best_init_metric = metric_val
                        best_ewma_hl = hl

        # Phase 2: Walk-forward lambda tuning + OOS execution
        current_date = oos_start_dt
        jm_xgb_chunks = []
        lambda_history = []

        while current_date < oos_end_dt:
            chunk_end = min(current_date + pd.DateOffset(months=6), oos_end_dt)
            
            if config.validation_window_type == 'expanding':
                val_start = pd.to_datetime(data_start)
            else:
                val_start = current_date - pd.DateOffset(years=VALIDATION_WINDOW_YRS)

            lambda_scores = []
            for lmbda in LAMBDA_GRID:
                val_res = simulate_strategy_fast(df, val_start, current_date, lmbda, cache, config, include_xgboost=True, ewma_halflife=best_ewma_hl)
                if not val_res.empty:
                    _, _, sharpe, sortino, _ = calculate_metrics(val_res['Strat_Return'], val_res['RF_Rate'])
                    metric_val = sortino if config.tuning_metric == 'sortino' else sharpe
                    if not np.isnan(metric_val):
                        lambda_scores.append((metric_val, lmbda))
                        
            lambda_scores.sort(key=lambda x: x[0], reverse=True)
            if not lambda_scores:
                lambda_scores = [(0.0, LAMBDA_GRID[len(LAMBDA_GRID)//2])]
                
            best_lambdas = [l for _, l in lambda_scores[:max(1, config.lambda_ensemble_k)]]
            best_lambda = best_lambdas[0]

            if config.lambda_smoothing and lambda_history:
                best_lambda = (0.7 * best_lambda) + (0.3 * lambda_history[-1])

            lambda_history.append(best_lambda)

            if config.lambda_ensemble_k > 1:
                oos_chunks = []
                for l_val in best_lambdas:
                    chunk = run_period_forecast_fast(df, current_date, l_val, cache, config, include_xgboost=True)
                    if chunk is not None: oos_chunks.append(chunk)
                if oos_chunks:
                    avg_prob = sum(c['Raw_Prob'] for c in oos_chunks) / len(oos_chunks)
                    final_chunk = oos_chunks[0].copy()
                    final_chunk['Raw_Prob'] = avg_prob
                    jm_xgb_chunks.append(final_chunk)
            else:
                oos_chunk = run_period_forecast_fast(df, current_date, best_lambda, cache, config, include_xgboost=True)
                if oos_chunk is not None:
                    jm_xgb_chunks.append(oos_chunk)
                    
            current_date = chunk_end

        if not jm_xgb_chunks:
            results.append({
                'Ticker': ticker, 'Period': period_name,
                'Strategy': 'JM-XGB', 'Ann_Ret': np.nan, 'Ann_Vol': np.nan,
                'Sharpe': np.nan, 'Sortino': np.nan, 'Max_DD': np.nan,
            })
            results.append({
                'Ticker': ticker, 'Period': period_name,
                'Strategy': 'B&H', 'Ann_Ret': np.nan, 'Ann_Vol': np.nan,
                'Sharpe': np.nan, 'Sortino': np.nan, 'Max_DD': np.nan,
            })
            continue

        jm_xgb_df = pd.concat(jm_xgb_chunks)

        # Apply EWMA + threshold + signal shift
        if best_ewma_hl == 0:
            jm_xgb_df['State_Prob'] = jm_xgb_df['Raw_Prob']
        else:
            jm_xgb_df['State_Prob'] = jm_xgb_df['Raw_Prob'].ewm(halflife=best_ewma_hl).mean()
            
        if config.allocation_style == "binary":
            jm_xgb_df['Forecast_State'] = (jm_xgb_df['State_Prob'] > config.prob_threshold).astype(int)
            trading_signals = jm_xgb_df['Forecast_State'].shift(1).fillna(0)
            alloc_target = 1.0 - trading_signals
        elif config.allocation_style == "continuous":
            alloc_target = (1.0 - jm_xgb_df['State_Prob']).shift(1).fillna(1.0)
            
        strat_rets = (alloc_target * jm_xgb_df['Target_Return']) + ((1.0 - alloc_target) * jm_xgb_df['RF_Rate'])
        trades = alloc_target.diff().abs().fillna(0)
        jm_xgb_df['Strat_Return'] = strat_rets - (trades * TRANSACTION_COST)

        # Trim to exact OOS window
        mask = (jm_xgb_df.index >= oos_start_dt) & (jm_xgb_df.index < oos_end_dt)
        jm_xgb_df = jm_xgb_df[mask]
        if jm_xgb_df.empty:
            continue

        # JM-XGB metrics
        ret, vol, sharpe, sortino, mdd = calculate_metrics(jm_xgb_df['Strat_Return'], jm_xgb_df['RF_Rate'])
        results.append({
            'Ticker': ticker, 'Period': period_name,
            'Strategy': 'JM-XGB', 'Ann_Ret': ret, 'Ann_Vol': vol,
            'Sharpe': sharpe, 'Sortino': sortino, 'Max_DD': mdd,
        })

        # B&H metrics
        bh_ret, bh_vol, bh_sharpe, bh_sortino, bh_mdd = calculate_metrics(
            jm_xgb_df['Target_Return'], jm_xgb_df['RF_Rate'])
        results.append({
            'Ticker': ticker, 'Period': period_name,
            'Strategy': 'B&H', 'Ann_Ret': bh_ret, 'Ann_Vol': bh_vol,
            'Sharpe': bh_sharpe, 'Sortino': bh_sortino, 'Max_DD': bh_mdd,
        })

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_markdown_report(results_df, elapsed, timestamp, asset_data, tickers, asset_classes, list_name):
    config = StrategyConfig()
    """Generate a markdown report with parameters and full results."""
    lines = []
    lines.append(f"# JM-XGB Multi-Asset Benchmark Report — {list_name}")
    lines.append(f"")
    lines.append(f"**Generated:** {timestamp}")
    lines.append(f"**Runtime:** {elapsed:.1f}s")
    lines.append(f"")

    # Data coverage
    lines.append(f"## Data Coverage")
    lines.append(f"")
    lines.append(f"| Ticker | Start | End | Rows |")
    lines.append(f"|---|---|---|---|")
    for ticker in tickers:
        if ticker in asset_data:
            df = asset_data[ticker]
            lines.append(f"| {ticker} | {df.index[0].date()} | {df.index[-1].date()} | {len(df)} |")
        else:
            lines.append(f"| {ticker} | N/A | N/A | N/A |")
    lines.append(f"")

    # DD exclusion note
    dd_excluded = [t for t in tickers if t in DD_EXCLUDE_TICKERS]
    dd_standard = [t for t in tickers if t not in DD_EXCLUDE_TICKERS]
    lines.append(f"## Feature Set")
    lines.append(f"")
    lines.append(f"**Paper Table 2 compliance:** Bonds and gold use 6 return features (Avg_Return×3, Sortino×3); "
                 f"equities use 8 features (DD_log×2, Avg_Return×3, Sortino×3).")
    lines.append(f"")
    if dd_excluded:
        lines.append(f"- **DD excluded (6 features):** {', '.join(dd_excluded)}")
    if dd_standard:
        lines.append(f"- **DD included (8 features):** {', '.join(dd_standard)}")
    lines.append(f"")

    # Aggregate summary
    full_data = results_df[results_df['Period'] == 'Full (2007-2025)']
    if not full_data.empty:
        jm_full = full_data[full_data['Strategy'] == 'JM-XGB'].dropna(subset=['Sharpe'])
        bh_full = full_data[full_data['Strategy'] == 'B&H'].dropna(subset=['Sharpe'])

        wins, total = 0, 0
        for ticker in tickers:
            j = jm_full[jm_full['Ticker'] == ticker]
            b = bh_full[bh_full['Ticker'] == ticker]
            if not j.empty and not b.empty:
                total += 1
                if j.iloc[0]['Sharpe'] > b.iloc[0]['Sharpe']:
                    wins += 1

        lines.append(f"## Aggregate Summary (Full Period)")
        lines.append(f"")
        if total > 0:
            lines.append(f"**JM-XGB beats B&H on Sharpe: {wins}/{total} assets ({wins/total*100:.0f}%)**")
            lines.append(f"")

        lines.append(f"| Ticker | JM-XGB | B&H | Sharpe Delta | JM-XGB MDD | B&H MDD | Verdict |")
        lines.append(f"|---|---:|---:|---:|---:|---:|---|")
        for ticker in tickers:
            j = jm_full[jm_full['Ticker'] == ticker]
            b = bh_full[bh_full['Ticker'] == ticker]
            if j.empty or b.empty:
                lines.append(f"| {ticker} | N/A | N/A | N/A | N/A | N/A | N/A |")
                continue
            js, bs = j.iloc[0]['Sharpe'], b.iloc[0]['Sharpe']
            delta = js - bs
            jmdd, bmdd = j.iloc[0]['Max_DD'], b.iloc[0]['Max_DD']
            verdict = 'WIN' if js > bs else 'LOSE'
            lines.append(f"| {ticker} | {js:.2f} | {bs:.2f} | {delta:+.2f} | {jmdd*100:.1f}% | {bmdd*100:.1f}% | **{verdict}** |")

        if not jm_full.empty and not bh_full.empty:
            avg_j = jm_full['Sharpe'].mean()
            avg_b = bh_full['Sharpe'].mean()
            avg_jmdd = jm_full['Max_DD'].mean()
            avg_bmdd = bh_full['Max_DD'].mean()
            lines.append(f"| **AVG** | **{avg_j:.2f}** | **{avg_b:.2f}** | **{avg_j-avg_b:+.2f}** | **{avg_jmdd*100:.1f}%** | **{avg_bmdd*100:.1f}%** | |")
        lines.append(f"")

        # Asset class summary
        lines.append(f"### Asset Class Averages (Full Period Sharpe)")
        lines.append(f"")
        lines.append(f"| Asset Class | JM-XGB Avg | B&H Avg | Delta |")
        lines.append(f"|---|---:|---:|---:|")
        for cls_name, cls_tickers in asset_classes.items():
            j_sharpes, b_sharpes = [], []
            for t in cls_tickers:
                j = full_data[(full_data['Ticker'] == t) & (full_data['Strategy'] == 'JM-XGB')]
                b = full_data[(full_data['Ticker'] == t) & (full_data['Strategy'] == 'B&H')]
                if not j.empty and not b.empty and not np.isnan(j.iloc[0]['Sharpe']):
                    j_sharpes.append(j.iloc[0]['Sharpe'])
                    b_sharpes.append(b.iloc[0]['Sharpe'])
            if j_sharpes:
                lines.append(f"| {cls_name} | {np.mean(j_sharpes):.2f} | {np.mean(b_sharpes):.2f} | {np.mean(j_sharpes)-np.mean(b_sharpes):+.2f} |")
            else:
                lines.append(f"| {cls_name} | N/A | N/A | N/A |")
        lines.append(f"")

    # Per-period tables
    for period_name, _, _ in TIME_PERIODS:
        period_data = results_df[results_df['Period'] == period_name]
        if period_data.empty:
            continue

        lines.append(f"## {period_name}")
        lines.append(f"")
        lines.append(f"| Ticker | JM-XGB Sharpe | B&H Sharpe | Delta | JM-XGB Ret | B&H Ret | JM-XGB MDD | B&H MDD |")
        lines.append(f"|---|---:|---:|---:|---:|---:|---:|---:|")

        for ticker in tickers:
            jmxgb = period_data[(period_data['Ticker'] == ticker) & (period_data['Strategy'] == 'JM-XGB')]
            bh = period_data[(period_data['Ticker'] == ticker) & (period_data['Strategy'] == 'B&H')]
            if jmxgb.empty or bh.empty:
                continue
            j = jmxgb.iloc[0]
            b = bh.iloc[0]
            if np.isnan(j['Sharpe']):
                lines.append(f"| {ticker} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
                continue
            delta = j['Sharpe'] - b['Sharpe']
            lines.append(f"| {ticker} | {j['Sharpe']:.2f} | {b['Sharpe']:.2f} | {delta:+.2f} | {j['Ann_Ret']*100:.1f}% | {b['Ann_Ret']*100:.1f}% | {j['Max_DD']*100:.1f}% | {b['Max_DD']*100:.1f}% |")
        lines.append(f"")

    # Parameters
    lines.append(f"## Strategy Configuration (`{config.name}`)")
    lines.append(f"")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Allocation style | {config.allocation_style} |")
    lines.append(f"| Tuning metric | {config.tuning_metric} |")
    lines.append(f"| Binary Prob threshold | {config.prob_threshold} |")
    lines.append(f"| Validation window | {config.validation_window_type} ({VALIDATION_WINDOW_YRS}yrs) |")
    lines.append(f"| Lambda smoothing | {config.lambda_smoothing} |")
    lines.append(f"| Lambda ensemble K | {config.lambda_ensemble_k} |")
    lines.append(f"| Dyn. Feature Select | {config.dynamic_feature_selection} |")
    lines.append(f"| Online learning | {config.xgb_online_learning} |")
    lines.append(f"| Lambda grid | {LAMBDA_GRID} |")
    lines.append(f"| EWMA HL grid | {EWMA_HL_GRID} |")
    lines.append(f"")

    return '\n'.join(lines)


def main():
    # Load asset lists and parse CLI
    all_lists = load_asset_lists()
    selected_name = parse_asset_list_selection(sys.argv[1:], all_lists)
    if selected_name is None:
        sys.exit(0)

    selected = all_lists[selected_name]
    tickers = selected['tickers']
    asset_classes = selected['asset_classes']
    data_start = selected['data_start']

    t0 = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Fetch all data (sequential — yfinance doesn't parallelize well)
    print("=" * 80)
    print(f"  MULTI-ASSET JM-XGB BENCHMARK  [{selected_name}]")
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

    # 2. Run backtests (parallel across assets)
    print(f"\nRunning walk-forward backtests across {len(TIME_PERIODS)} time periods...")
    print(f"Using {min(cpu_count(), len(asset_data))} parallel workers\n")

    config = StrategyConfig()
    args_list = [(ticker, df, config, data_start) for ticker, df in asset_data.items()]

    all_results = []
    n_workers = min(cpu_count(), len(args_list))
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
    print(f"\nBacktest completed in {elapsed:.1f}s")

    # 3. Build results DataFrame and save
    results_df = pd.DataFrame(all_results)

    csv_path = os.path.join(BENCHMARKS_DIR, f'benchmark_results_{timestamp}.csv')
    results_df.to_csv(csv_path, index=False)

    # 4. Generate and save markdown report
    md_content = generate_markdown_report(results_df, elapsed, timestamp, asset_data, tickers, asset_classes, selected_name)
    md_path = os.path.join(BENCHMARKS_DIR, f'benchmark_report_{timestamp}.md')
    with open(md_path, 'w') as f:
        f.write(md_content)

    # 5. Print summary to console
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)

    for period_name, _, _ in TIME_PERIODS:
        period_data = results_df[results_df['Period'] == period_name]
        if period_data.empty:
            continue

        print(f"\n{'─' * 80}")
        print(f"  {period_name}")
        print(f"{'─' * 80}")
        print(f"  {'Ticker':<8} │ {'JM-XGB Sharpe':>13} │ {'B&H Sharpe':>10} │ {'Delta':>7} │ {'JM-XGB Ret':>10} │ {'B&H Ret':>9} │ {'JM-XGB MDD':>10} │ {'B&H MDD':>9}")
        print(f"  {'─'*8}─┼─{'─'*13}─┼─{'─'*10}─┼─{'─'*7}─┼─{'─'*10}─┼─{'─'*9}─┼─{'─'*10}─┼─{'─'*9}")

        for ticker in tickers:
            jmxgb = period_data[(period_data['Ticker'] == ticker) & (period_data['Strategy'] == 'JM-XGB')]
            bh = period_data[(period_data['Ticker'] == ticker) & (period_data['Strategy'] == 'B&H')]
            if jmxgb.empty or bh.empty:
                continue
            j = jmxgb.iloc[0]
            b = bh.iloc[0]
            if np.isnan(j['Sharpe']):
                print(f"  {ticker:<8} │ {'N/A':>13} │ {'N/A':>10} │ {'N/A':>7} │ {'N/A':>10} │ {'N/A':>9} │ {'N/A':>10} │ {'N/A':>9}")
                continue
            delta = j['Sharpe'] - b['Sharpe']
            marker = '+' if delta > 0 else ''
            print(f"  {ticker:<8} │ {j['Sharpe']:>13.2f} │ {b['Sharpe']:>10.2f} │ {marker}{delta:>6.2f} │ {j['Ann_Ret']*100:>9.1f}% │ {b['Ann_Ret']*100:>8.1f}% │ {j['Max_DD']*100:>9.1f}% │ {b['Max_DD']*100:>8.1f}%")

    # Aggregate
    full_data = results_df[results_df['Period'] == 'Full (2007-2025)']
    if not full_data.empty:
        jm_full = full_data[full_data['Strategy'] == 'JM-XGB'].dropna(subset=['Sharpe'])
        bh_full = full_data[full_data['Strategy'] == 'B&H'].dropna(subset=['Sharpe'])

        wins, total = 0, 0
        for ticker in tickers:
            j = jm_full[jm_full['Ticker'] == ticker]
            b = bh_full[bh_full['Ticker'] == ticker]
            if not j.empty and not b.empty:
                total += 1
                if j.iloc[0]['Sharpe'] > b.iloc[0]['Sharpe']:
                    wins += 1

        print(f"\n{'=' * 80}")
        print("  AGGREGATE SUMMARY (Full Period: 2007-2025)")
        print(f"{'=' * 80}")
        if total > 0:
            print(f"\n  JM-XGB beats B&H on Sharpe: {wins}/{total} assets ({wins/total*100:.0f}%)")

        print(f"\n  {'Ticker':<8} │ {'JM-XGB':>8} │ {'B&H':>8} │ {'Sharpe Δ':>9} │ {'JM-XGB MDD':>10} │ {'B&H MDD':>9} │ {'Verdict':>8}")
        print(f"  {'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*9}─┼─{'─'*10}─┼─{'─'*9}─┼─{'─'*8}")
        for ticker in tickers:
            j = jm_full[jm_full['Ticker'] == ticker]
            b = bh_full[bh_full['Ticker'] == ticker]
            if j.empty or b.empty:
                print(f"  {ticker:<8} │ {'N/A':>8} │ {'N/A':>8} │ {'N/A':>9} │ {'N/A':>10} │ {'N/A':>9} │ {'N/A':>8}")
                continue
            js, bs = j.iloc[0]['Sharpe'], b.iloc[0]['Sharpe']
            delta = js - bs
            jmdd, bmdd = j.iloc[0]['Max_DD'], b.iloc[0]['Max_DD']
            verdict = 'WIN' if js > bs else 'LOSE'
            print(f"  {ticker:<8} │ {js:>8.2f} │ {bs:>8.2f} │ {delta:>+9.2f} │ {jmdd*100:>9.1f}% │ {bmdd*100:>8.1f}% │ {verdict:>8}")

        if not jm_full.empty and not bh_full.empty:
            print(f"  {'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*9}─┼─{'─'*10}─┼─{'─'*9}─┼─{'─'*8}")
            avg_j = jm_full['Sharpe'].mean()
            avg_b = bh_full['Sharpe'].mean()
            avg_jmdd = jm_full['Max_DD'].mean()
            avg_bmdd = bh_full['Max_DD'].mean()
            print(f"  {'AVG':<8} │ {avg_j:>8.2f} │ {avg_b:>8.2f} │ {avg_j-avg_b:>+9.2f} │ {avg_jmdd*100:>9.1f}% │ {avg_bmdd*100:>8.1f}% │ {'':>8}")

    print(f"\nOutputs saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  Report: {md_path}")
    print(f"Total runtime: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
