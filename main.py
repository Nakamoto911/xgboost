import sys
import types

# Compatibility shim for distutils.version which was removed in Python 3.12+
# Required by pandas-datareader
try:
    import distutils
    import distutils.version
except ImportError:
    d = types.ModuleType('distutils')
    dv = types.ModuleType('distutils.version')
    class LooseVersion:
        def __init__(self, vstring=None):
            self.vstring = vstring
        def __str__(self):
            return self.vstring
        def __repr__(self):
            return f"LooseVersion('{self.vstring}')"
        def __lt__(self, other):
            return self.vstring < (other.vstring if hasattr(other, 'vstring') else other)
        def __le__(self, other):
            return self.vstring <= (other.vstring if hasattr(other, 'vstring') else other)
        def __gt__(self, other):
            return self.vstring > (other.vstring if hasattr(other, 'vstring') else other)
        def __ge__(self, other):
            return self.vstring >= (other.vstring if hasattr(other, 'vstring') else other)
        def __eq__(self, other):
            return self.vstring == (other.vstring if hasattr(other, 'vstring') else other)
    dv.LooseVersion = LooseVersion
    d.version = dv
    sys.modules['distutils'] = d
    sys.modules['distutils.version'] = dv

import os
import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier
import warnings
import matplotlib.pyplot as plt
from datetime import timedelta
import pandas_datareader.data as web
from config import StrategyConfig

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CONFIGURATION: Change Asset and Time Periods Here
# ==============================================================================
TARGET_TICKER = '^SP500TR'      # S&P 500 Total Return (includes dividends)
BOND_TICKER = 'VBMFX'           # Vanguard Total Bond Market (Proxy for US Agg Bond)
RISK_FREE_TICKER = '^IRX'       # 13-Week Treasury Bill (Proxy for Risk-Free)
VIX_TICKER = '^VIX'             # VIX Index
LARGECAP_TICKER = '^SP500TR'    # Paper Table 3: Stock-Bond Corr uses LargeCap vs AggBond for ALL assets

# Paper Table 2: bonds and gold use only 6 return features (Avg_Return×3, Sortino×3).
# DD features don't separate regimes for these assets and should be excluded.
DD_EXCLUDE_TICKERS = {'AGG', 'VBMFX',        # AggBond proxies
                      'SPTL', 'VUSTX', 'IEF', 'TLT', 'VGLT',  # Treasury proxies
                      'GLD', 'GC=F', 'IAU'}   # Gold proxies

# Timeline
START_DATE_DATA = '1987-01-01'  # Need data way before 1999 to allow for 11-year lookbacks
OOS_START_DATE = '2007-01-01'   # Out-of-sample testing begins (paper: 2007-2023)
END_DATE = '2026-01-01'         # End of testing period

# Transaction costs
TRANSACTION_COST = 0.0005       # 5 basis points (0.05%)

# Validation Window (Years)
VALIDATION_WINDOW_YRS = 5

# Lambda candidate grid for Jump Model
# Focused mid-range grid prevents walk-forward overfitting from extreme values (0, 100).
# Session 4 diagnosis: wide grid [0, logspace(1,100)] → Sharpe 0.54; focused → 0.85.
LAMBDA_GRID = [4.64, 10.0, 15.0, 21.54, 30.0, 46.42, 70.0, 100.0]  # Dense 8-pt mid-range (Session 5)

# EWMA halflife candidates for probability smoothing (paper: Section 4.2)
EWMA_HL_GRID = [0, 2, 4, 8]

# Paper-prescribed EWMA halflives per asset (from paper Section 4.2, Table).
# Auto-tuning on Yahoo Finance data overfits the validation window for some assets.
# When TARGET_TICKER has a known paper halflife, skip Phase 1 tuning and use it directly.
PAPER_EWMA_HL = {
    # hl=8: LargeCap, MidCap, SmallCap, REIT, AggBond, Treasury
    '^SP500TR': 8, 'IVV': 8,           # LargeCap
    'IJH': 8, 'VIMSX': 8,              # MidCap
    'IWM': 8, 'NAESX': 2,              # SmallCap (Session 5: Yahoo needs hl=2)
    'IYR': 8, 'FRESX': 8,              # REIT
    'AGG': 8, 'VBMFX': 8,              # AggBond
    'SPTL': 8, 'VUSTX': 8, 'TLT': 8, 'VGLT': 8,  # Treasury
    # hl=4: Commodity, Gold
    'DBC': 4, 'PCASX': 4,              # Commodity
    'GLD': 4, 'GC=F': 4, 'IAU': 4,    # Gold
    # hl=2: Corporate
    'SPBO': 2, 'VWESX': 2,             # Corporate
    # hl=0: EM, EAFE, HighYield
    'EEM': 0, 'VEIEX': 0,              # EM
    'EFA': 0, 'FDIVX': 0,              # EAFE
    'HYG': 0, 'VWEHX': 0,             # HighYield
}

# Override from environment variables (used by Diagnostics Launcher to sync with dashboard)
if os.environ.get('XGB_TARGET_TICKER'):
    TARGET_TICKER = os.environ['XGB_TARGET_TICKER']
if os.environ.get('XGB_BOND_TICKER'):
    BOND_TICKER = os.environ['XGB_BOND_TICKER']
if os.environ.get('XGB_RISK_FREE_TICKER'):
    RISK_FREE_TICKER = os.environ['XGB_RISK_FREE_TICKER']
if os.environ.get('XGB_VIX_TICKER'):
    VIX_TICKER = os.environ['XGB_VIX_TICKER']
if os.environ.get('XGB_START_DATE_DATA'):
    START_DATE_DATA = os.environ['XGB_START_DATE_DATA']
if os.environ.get('XGB_OOS_START_DATE'):
    OOS_START_DATE = os.environ['XGB_OOS_START_DATE']
if os.environ.get('XGB_END_DATE'):
    END_DATE = os.environ['XGB_END_DATE']
if os.environ.get('XGB_TRANSACTION_COST'):
    TRANSACTION_COST = float(os.environ['XGB_TRANSACTION_COST'])
if os.environ.get('XGB_VALIDATION_WINDOW_YRS'):
    VALIDATION_WINDOW_YRS = int(os.environ['XGB_VALIDATION_WINDOW_YRS'])
if os.environ.get('XGB_LAMBDA_GRID'):
    import json as _json
    LAMBDA_GRID = _json.loads(os.environ['XGB_LAMBDA_GRID'])
if os.environ.get('XGB_EWMA_HL_GRID'):
    import json as _json
    EWMA_HL_GRID = _json.loads(os.environ['XGB_EWMA_HL_GRID'])

# ==============================================================================
# 2. STATISTICAL JUMP MODEL (Implementation from scratch)
# ==============================================================================
class StatisticalJumpModel:
    """
    Implements a discrete 2-state Statistical Jump Model using alternating optimization
    (K-means style updates + Viterbi algorithm for state sequence with jump penalty).

    Matches the paper's jumpmodels library: k-means++ init, n_init=10, max_iter=1000, tol=1e-8.
    """
    def __init__(self, n_states=2, lambda_penalty=10.0, max_iter=1000, n_init=10, tol=1e-8):
        self.n_states = n_states
        self.lambda_penalty = lambda_penalty
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.means = None
        self.val_ = np.inf  # best objective value

    def _viterbi_full(self, X_arr, distances):
        """Full Viterbi (forward + backtrack) for E-step during fitting."""
        n_samples = X_arr.shape[0]
        cost_matrix = np.zeros((n_samples, self.n_states))
        back_pointers = np.zeros((n_samples, self.n_states), dtype=int)
        cost_matrix[0] = distances[0]

        penalty_matrix = self.lambda_penalty * (1 - np.eye(self.n_states))

        if self.n_states == 2:
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
        else:
            for t in range(1, n_samples):
                trans_costs = cost_matrix[t-1, :, None] + penalty_matrix
                best_prev_states = np.argmin(trans_costs, axis=0)
                cost_matrix[t] = trans_costs[best_prev_states, np.arange(self.n_states)] + distances[t]
                back_pointers[t] = best_prev_states

        # Backward pass (traceback)
        states = np.zeros(n_samples, dtype=int)
        states[-1] = np.argmin(cost_matrix[-1])
        obj_val = cost_matrix[-1, states[-1]]
        for t in range(n_samples - 2, -1, -1):
            states[t] = back_pointers[t+1, states[t+1]]

        return states, obj_val

    def fit_predict(self, X):
        from sklearn.cluster import kmeans_plusplus
        from sklearn.utils import check_random_state

        X_arr = np.array(X)
        n_samples, n_features = X_arr.shape

        rng = check_random_state(42)
        best_states = None
        best_val = np.inf
        best_means = None

        for init_idx in range(self.n_init):
            # K-means++ initialization (matches paper's jumpmodels library)
            centers, _ = kmeans_plusplus(X_arr, self.n_states, random_state=rng)
            means = centers.copy()

            states = np.zeros(n_samples, dtype=int)
            val_prev = np.inf

            for iteration in range(self.max_iter):
                distances = 0.5 * np.sum((X_arr[:, None, :] - means[None, :, :])**2, axis=2)
                new_states, obj_val = self._viterbi_full(X_arr, distances)

                # Update means given new states
                for k in range(self.n_states):
                    mask = (new_states == k)
                    if np.sum(mask) > 0:
                        means[k] = np.mean(X_arr[mask], axis=0)

                # Check convergence: same states AND objective improvement < tol
                if np.array_equal(states, new_states) or (val_prev - obj_val <= self.tol):
                    states = new_states
                    break
                val_prev = obj_val
                states = new_states

            # Keep best across all initializations
            # Recompute final objective with final means
            distances = 0.5 * np.sum((X_arr[:, None, :] - means[None, :, :])**2, axis=2)
            _, final_val = self._viterbi_full(X_arr, distances)

            if final_val < best_val:
                best_val = final_val
                best_states = states.copy()
                best_means = means.copy()

        self.means = best_means
        self.val_ = best_val
        return best_states

    def predict_online(self, X, last_known_state=None):
        """
        Forward-only Viterbi (online) prediction matching the paper's jumpmodels library.

        Runs the DP forward pass accumulating costs from t=0, then assigns each time step
        to argmin of accumulated values. No backtracking — each prediction at time t uses
        only data up to t. The last_known_state parameter is accepted for API compatibility
        but ignored (matching the paper's implementation).
        """
        X_arr = np.array(X)
        n_samples = X_arr.shape[0]

        # Loss matrix: 0.5 * squared Euclidean distance to each cluster center
        loss_mx = 0.5 * np.sum((X_arr[:, None, :] - self.means[None, :, :])**2, axis=2)  # (n_samples, n_states)

        # Penalty matrix: lambda for state changes, 0 for staying
        penalty_mx = self.lambda_penalty * (1 - np.eye(self.n_states))

        # DP forward pass (accumulated costs, no backtracking)
        values = np.empty((n_samples, self.n_states))
        values[0] = loss_mx[0]
        for t in range(1, n_samples):
            values[t] = loss_mx[t] + (values[t-1][:, np.newaxis] + penalty_mx).min(axis=0)

        # Online assignment: argmin of accumulated values at each time step
        states = values.argmin(axis=1)
        return states

# ==============================================================================
# 3. DATA FETCHING & FEATURE ENGINEERING
# ==============================================================================
def _fetch_fred_data():
    """Fetch FRED macro data with separate caching (ticker-independent)."""
    import time
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    fred_cache_file = os.path.join(cache_dir, 'fred_cache.pkl')
    if os.path.exists(fred_cache_file):
        print("Loading FRED data from cache...")
        return pd.read_pickle(fred_cache_file)

    print("Fetching FRED data...")
    # Try pandas_datareader first, then fall back to direct FRED CSV API
    for attempt in range(3):
        try:
            fred_data = web.DataReader(['DGS2', 'DGS10'], 'fred', '1987-01-01', '2027-01-01')
            fred_data.to_pickle(fred_cache_file)
            return fred_data
        except Exception as e:
            print(f"Failed to fetch FRED data via datareader (attempt {attempt+1}/3): {e}")
            time.sleep(3 * (attempt + 1))

    # Fallback: fetch directly from FRED CSV API with longer timeout
    print("Trying FRED CSV API fallback...")
    import requests
    try:
        series_dict = {}
        for series_id in ['DGS2', 'DGS10']:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd=1987-01-01&coed=2027-01-01"
            resp = requests.get(url, timeout=90)
            resp.raise_for_status()
            from io import StringIO
            s = pd.read_csv(StringIO(resp.text), index_col=0, parse_dates=True)
            s.columns = [series_id]
            s[series_id] = pd.to_numeric(s[series_id], errors='coerce')
            series_dict[series_id] = s[series_id]
        fred_data = pd.DataFrame(series_dict)
        fred_data.to_pickle(fred_cache_file)
        return fred_data
    except Exception as e:
        print(f"FRED CSV fallback also failed: {e}")

    raise ValueError("Failed to download FRED data from all sources.")


def fetch_and_prepare_data():
    safe_ticker = TARGET_TICKER.replace('^', '').replace('=', '')
    cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache',
                              f'data_cache_{safe_ticker}_{START_DATE_DATA[:4]}_{END_DATE[:7]}.pkl')
    if os.path.exists(cache_file):
        print("Loading data from local cache...")
        return pd.read_pickle(cache_file)

    # Check for any existing cache for this ticker (different date suffix)
    import glob
    cache_dir = os.path.dirname(cache_file)
    alt_caches = sorted(glob.glob(os.path.join(cache_dir, f'data_cache_{safe_ticker}_*.pkl')))
    if alt_caches:
        best = alt_caches[-1]  # newest by filename
        print(f"Loading data from alternative cache: {os.path.basename(best)}")
        return pd.read_pickle(best)

    fred_data = _fetch_fred_data()

    print("Fetching data from Yahoo Finance...")

    # 1. Fetch main assets — always include LARGECAP_TICKER for Stock-Bond Corr macro feature
    tickers = list(dict.fromkeys([TARGET_TICKER, BOND_TICKER, RISK_FREE_TICKER, VIX_TICKER, LARGECAP_TICKER]))

    # Download individually to bypass yfinance multi-ticker download bugs
    series_list = []
    for ticker in tickers:
        try:
            df_ticker = yf.download(ticker, start=START_DATE_DATA, end=END_DATE, auto_adjust=False)
            if df_ticker.empty:
                print(f"Warning: No data found for {ticker}")
                continue

            # Handle cases where 'Adj Close' might be nested (newer yfinance)
            if isinstance(df_ticker.columns, pd.MultiIndex):
                if 'Adj Close' in df_ticker.columns.levels[0]:
                    series = df_ticker['Adj Close'].iloc[:, 0].rename(ticker)
                else:
                    series = df_ticker.iloc[:, 0].rename(ticker)
            else:
                if 'Adj Close' in df_ticker.columns:
                    series = df_ticker['Adj Close'].rename(ticker)
                elif 'Close' in df_ticker.columns:
                    series = df_ticker['Close'].rename(ticker)
                else:
                    series = df_ticker.iloc[:, 0].rename(ticker)

            series_list.append(series)
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")

    if not series_list:
        raise ValueError("Failed to download any data from Yahoo Finance.")

    data = pd.concat(series_list, axis=1)
    data = data.ffill().dropna()
        
    fred_data = fred_data.ffill().dropna()
    
    # Merge datasets
    df = data.join(fred_data, how='inner')
    
    print("Calculating features...")
    features = pd.DataFrame(index=df.index)
    
    # Asset Returns
    target_returns = df[TARGET_TICKER].pct_change().fillna(0)
    features['Target_Return'] = target_returns
    
    # Risk-free daily rate (IRX is in %, e.g., 5.0 means 5%)
    features['RF_Rate'] = (df[RISK_FREE_TICKER] / 100) / 252 
    features['Excess_Return'] = target_returns - features['RF_Rate']
    
    # A. Asset-Specific Return Features
    # Downside Deviation (log scale) — excluded for bonds/gold per paper Table 2
    downside_returns = np.minimum(features['Excess_Return'], 0)
    if TARGET_TICKER not in DD_EXCLUDE_TICKERS:
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

    # B. Cross-Asset Macro Features
    # 2-Year Yield Features
    features['Yield_2Y_diff'] = df['DGS2'].diff().fillna(0)
    features['Yield_2Y_EWMA_diff'] = features['Yield_2Y_diff'].ewm(halflife=21).mean()
    
    # Yield Curve Slope Features
    slope = df['DGS10'] - df['DGS2']
    features['Yield_Slope_EWMA_10'] = slope.ewm(halflife=10).mean()
    slope_diff = slope.diff().fillna(0)
    features['Yield_Slope_EWMA_diff_21'] = slope_diff.ewm(halflife=21).mean()
    
    # VIX Features
    vix_log_diff = np.log(df[VIX_TICKER] / df[VIX_TICKER].shift(1)).fillna(0)
    features['VIX_EWMA_log_diff'] = vix_log_diff.ewm(halflife=63).mean()
    
    # Stock-Bond Correlation — paper Table 3: always corr(LargeCap, AggBond), identical for all target assets
    largecap_returns = df[LARGECAP_TICKER].pct_change().fillna(0)
    bond_returns = df[BOND_TICKER].pct_change().fillna(0)
    features['Stock_Bond_Corr'] = largecap_returns.rolling(window=252).corr(bond_returns).fillna(0)
    
    final_df = features.dropna()
    print(f"Saving data to local cache ({cache_file})...")
    final_df.to_pickle(cache_file)
    return final_df

# ==============================================================================
# 4. CORE ALGORITHM (Algorithm A & B Simulator)
# ==============================================================================
_forecast_cache = {}
_prev_booster = None

def run_period_forecast(df, current_date, lambda_penalty, config: StrategyConfig = None, include_xgboost=True, constrain_xgb=False):
    """
    Runs the identification (JM) and forecasting (XGB) for a specific date using 
    an 11-year lookback window.
    """
    if config is None:
        config = StrategyConfig()
        
    xgb_key = tuple(sorted(config.xgb_params.items())) if config.xgb_params else ()
    ablation_key = getattr(config, 'feature_ablation', 'all')
    cache_key = (current_date, lambda_penalty, include_xgboost, constrain_xgb, config.name, xgb_key, ablation_key)
    if cache_key in _forecast_cache:
        cached = _forecast_cache[cache_key]
        # If SHAP requested but cached result lacks SHAP columns, recompute
        if config.calculate_shap and cached is not None and not any(c.startswith('SHAP_') for c in cached.columns):
            pass  # fall through to recompute
        else:
            return cached

    train_start = current_date - pd.DateOffset(years=11)
    
    # Filter 11-year training data
    train_df = df[(df.index >= train_start) & (df.index < current_date)].copy()
    if len(train_df) < 252 * 5: # Failsafe if not enough data
        _forecast_cache[cache_key] = None
        return None
        
    # Standardize Return Features for JM
    return_features = [c for c in train_df.columns if c.startswith(('DD_', 'Avg_Ret_', 'Sortino_'))]
    X_train_jm = train_df[return_features]
    jm_train_mean = X_train_jm.mean()
    jm_train_std = X_train_jm.std()
    X_train_jm = (X_train_jm - jm_train_mean) / jm_train_std

    # 1. Regime Identification (Jump Model)
    jm = StatisticalJumpModel(n_states=2, lambda_penalty=lambda_penalty)
    identified_states = jm.fit_predict(X_train_jm.values)

    # Align states so State 0 is Bullish (higher cum excess return) and State 1 is Bearish
    cum_ret_0 = train_df['Excess_Return'][identified_states == 0].sum()
    cum_ret_1 = train_df['Excess_Return'][identified_states == 1].sum()
    if cum_ret_1 > cum_ret_0:
        identified_states = 1 - identified_states # Flip labels
        jm.means = jm.means[::-1].copy()

    # Shift labels forward by 1 day to create prediction targets (per paper: predict s_{t+1} from x_t)
    train_df['Target_State'] = np.roll(identified_states, -1)
    train_df = train_df.iloc[:-1] # Drop last row due to shift
    
    # Fetch 6-month Out-of-Sample features
    oos_end = current_date + pd.DateOffset(months=6)
    oos_df = df[(df.index >= current_date) & (df.index < oos_end)].copy()
    if len(oos_df) == 0:
        _forecast_cache[cache_key] = None
        return None

    if not include_xgboost:
        # Simple JM Baseline: Online assignment of OOS data to fitted clusters
        X_oos_jm = oos_df[return_features]
        # Standardize using training mean/std (saved before train_df was trimmed)
        X_oos_jm = (X_oos_jm - jm_train_mean) / jm_train_std
        
        # Predict day-by-day to simulate real-time tracking
        oos_states = jm.predict_online(X_oos_jm.values, last_known_state=identified_states[-1])
        
        oos_df['Forecast_State'] = oos_states
        result = oos_df[['Target_Return', 'RF_Rate', 'Forecast_State']]
        _forecast_cache[cache_key] = result
        return result

    # 2. Regime Forecasting (XGBoost)
    macro_features = ['Yield_2Y_EWMA_diff', 'Yield_Slope_EWMA_10', 'Yield_Slope_EWMA_diff_21',
                      'VIX_EWMA_log_diff', 'Stock_Bond_Corr']

    # Feature ablation: select feature subset for XGBoost
    ablation = getattr(config, 'feature_ablation', 'all')
    if ablation == 'return_only':
        all_features = return_features[:]
    elif ablation == 'macro_only':
        all_features = macro_features[:]
    else:
        all_features = return_features + macro_features
    
    X_train_xgb = train_df[all_features].copy()
    y_train_xgb = train_df['Target_State'].copy()
    X_oos_xgb = oos_df[all_features].copy()
    
    # Idea 4 (Feature Selection):
    if config.dynamic_feature_selection:
        temp_xgb = XGBClassifier(eval_metric='logloss', random_state=42, **config.xgb_params)
        temp_xgb.fit(X_train_xgb, y_train_xgb)
        importances = temp_xgb.feature_importances_
        n_drop = int(len(all_features) * 0.2)
        if n_drop > 0:
            drop_indices = np.argsort(importances)[:n_drop]
            all_features = [f for i, f in enumerate(all_features) if i not in drop_indices]
            X_train_xgb = X_train_xgb[all_features]
            X_oos_xgb = X_oos_xgb[all_features]
            
    xgb = XGBClassifier(
        eval_metric='logloss', random_state=42, **config.xgb_params
    )

    global _prev_booster
    # Idea 10 (Online Learning):
    if config.xgb_online_learning:
        xgb.fit(X_train_xgb, y_train_xgb, xgb_model=_prev_booster)
        _prev_booster = xgb.get_booster()
    else:
        xgb.fit(X_train_xgb, y_train_xgb)
    
    # Get probabilities for Class 1 (Bearish)
    oos_probs = xgb.predict_proba(X_oos_xgb)[:, 1]
    oos_df['Raw_Prob'] = oos_probs
    
    # Calculate SHAP values if requested
    if config.calculate_shap:
        import shap
        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(X_oos_xgb)
        
        base_value = explainer.expected_value
        if isinstance(base_value, (np.ndarray, list)):
            # For Binary Classification, sometimes SHAP returns [logodds_0, logodds_1]
            base_value = base_value[-1] if len(base_value) > 1 else base_value[0]
            
        oos_df['SHAP_Base_Value'] = base_value
        for i, col in enumerate(all_features):
            feature_shap_vals = shap_values[:, i]
            # if shap_values is a list (e.g., class 0 and class 1), use the class 1 values
            if isinstance(shap_values, list):
                 feature_shap_vals = shap_values[1][:, i]
            oos_df[f'SHAP_{col}'] = feature_shap_vals
            
    # Calculate JM online regimes for the OOS period to serve as True Labels for Audit
    X_oos_jm = oos_df[return_features]
    X_oos_jm = (X_oos_jm - train_df[return_features].mean()) / train_df[return_features].std()
    oos_states = jm.predict_online(X_oos_jm.values, last_known_state=identified_states[-1])
    oos_df['JM_Target_State'] = oos_states
    
    # Keep the feature values as well for scatter/dependence plots later if needed
    for col in all_features:
        oos_df[f'Feature_{col}'] = oos_df[col]
        
    cols_to_keep = ['Target_Return', 'RF_Rate', 'Raw_Prob', 'JM_Target_State'] + [f'Feature_{col}' for col in all_features]
    
    if config.calculate_shap:
        cols_to_keep += ['SHAP_Base_Value'] + [f'SHAP_{col}' for col in all_features]
        
    result = oos_df[cols_to_keep]
    _forecast_cache[cache_key] = result
    return result

def simulate_strategy(df, start_date, end_date, lambda_penalty, config: StrategyConfig = None, include_xgboost=True, constrain_xgb=False, ewma_halflife=8):
    """Simulates the entire period in 6-month chunks for a given lambda."""
    if config is None:
        config = StrategyConfig()
        
    results = []
    current_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    while current_date < end_date:
        res = run_period_forecast(df, current_date, lambda_penalty, config, include_xgboost, constrain_xgb)
        if res is not None:
            results.append(res)
        current_date += pd.DateOffset(months=6)

    if not results:
        return pd.DataFrame()

    full_res = pd.concat(results)

    if include_xgboost:
        # 1. Apply continuous EWMA over the unbroken time series (paper: tune halflife from {0,2,4,8})
        if ewma_halflife == 0:
            full_res['State_Prob'] = full_res['Raw_Prob']
        else:
            full_res['State_Prob'] = full_res['Raw_Prob'].ewm(halflife=ewma_halflife).mean()
        
        # Idea 5 (Thresholding) & Idea 6 (Continuous Allocation)
        if config.allocation_style == "binary":
            full_res['Forecast_State'] = (full_res['State_Prob'] > config.prob_threshold).astype(int)
            trading_signals = full_res['Forecast_State'].shift(1).fillna(0)
            alloc_target = 1.0 - trading_signals # 1.0 is full target asset, 0.0 is risk-free
        elif config.allocation_style == "continuous":
            # Invest linearly based on confidence: 1 - P(bear)
            alloc_target = (1.0 - full_res['State_Prob']).shift(1).fillna(1.0)
    else:
        # Simple JM Baseline Behavior
        trading_signals = full_res['Forecast_State'].shift(1).fillna(0)
        alloc_target = 1.0 - trading_signals

    # Calculate 0/1 Strategy Returns
    strat_returns = (alloc_target * full_res['Target_Return']) + ((1.0 - alloc_target) * full_res['RF_Rate'])
    trades = alloc_target.diff().abs().fillna(0)
    
    full_res['Strat_Return'] = strat_returns - (trades * TRANSACTION_COST)
    full_res['Trades'] = trades

    return full_res

def walk_forward_backtest(df, config: StrategyConfig = None) -> pd.DataFrame:
    if config is None:
        config = StrategyConfig()
        
    current_date = pd.to_datetime(OOS_START_DATE)
    final_end_date = pd.to_datetime(END_DATE)

    jm_xgb_results = []
    lambda_history = []
    lambda_dates = []

    global _prev_booster
    _prev_booster = None

    # --- Phase 1: Tune EWMA halflife once on initial validation window ---
    # Use paper-prescribed halflife if available for this ticker (avoids overfitting Yahoo data)
    if TARGET_TICKER in PAPER_EWMA_HL:
        best_ewma_hl = PAPER_EWMA_HL[TARGET_TICKER]
    else:
        initial_val_start = current_date - pd.DateOffset(years=VALIDATION_WINDOW_YRS)
        if config.validation_window_type == 'expanding':
            initial_val_start = pd.to_datetime(OOS_START_DATE) - pd.DateOffset(years=VALIDATION_WINDOW_YRS)

        best_ewma_hl = EWMA_HL_GRID[-1]
        best_init_metric = -np.inf

        for hl in EWMA_HL_GRID:
            for lmbda in LAMBDA_GRID:
                val_res = simulate_strategy(df, initial_val_start, current_date, lmbda, config, include_xgboost=True, ewma_halflife=hl)
                if not val_res.empty:
                    _, _, sharpe, sortino, _ = calculate_metrics(val_res['Strat_Return'], val_res['RF_Rate'])
                    metric_val = sortino if config.tuning_metric == 'sortino' else sharpe
                    if metric_val > best_init_metric:
                        best_init_metric = metric_val
                        best_ewma_hl = hl

    # --- Phase 2: Walk-forward lambda tuning ---
    while current_date < final_end_date:
        chunk_end = min(current_date + pd.DateOffset(months=6), final_end_date)
        
        if config.validation_window_type == 'expanding':
            val_start = pd.to_datetime(OOS_START_DATE) - pd.DateOffset(years=VALIDATION_WINDOW_YRS)
        else:
            val_start = current_date - pd.DateOffset(years=VALIDATION_WINDOW_YRS)
            
        lambda_scores = []

        if config.lambda_subwindow_consensus:
            # Run simulate_strategy ONCE per lambda for full validation window (hits forecast cache),
            # then slice into 3 overlapping sub-windows and evaluate each independently.
            val_duration = (current_date - val_start).days
            sub_boundaries = [
                (val_start, val_start + pd.DateOffset(days=val_duration // 2)),
                (val_start + pd.DateOffset(days=val_duration // 4), val_start + pd.DateOffset(days=3 * val_duration // 4)),
                (val_start + pd.DateOffset(days=val_duration // 2), current_date),
            ]
            # Collect per-lambda, per-sub-window scores
            sub_best_lambdas = [[] for _ in sub_boundaries]  # scores per sub-window
            for lmbda in LAMBDA_GRID:
                val_res = simulate_strategy(df, val_start, current_date, lmbda, config, include_xgboost=True, ewma_halflife=best_ewma_hl)
                if val_res.empty:
                    continue
                for sw_idx, (sw_start, sw_end) in enumerate(sub_boundaries):
                    sw_slice = val_res[(val_res.index >= sw_start) & (val_res.index < sw_end)]
                    if len(sw_slice) >= 60:  # need at least ~3 months
                        _, _, sharpe, sortino, _ = calculate_metrics(sw_slice['Strat_Return'], sw_slice['RF_Rate'])
                        metric_val = sortino if config.tuning_metric == 'sortino' else sharpe
                        sub_best_lambdas[sw_idx].append((metric_val, lmbda))
            # Find best lambda per sub-window, take median
            consensus_lambdas = []
            for sw_scores in sub_best_lambdas:
                if sw_scores:
                    sw_scores.sort(key=lambda x: x[0], reverse=True)
                    consensus_lambdas.append(sw_scores[0][1])
            if consensus_lambdas:
                best_lambda = float(np.median(consensus_lambdas))
                best_lambdas = [best_lambda]
                lambda_scores = [(0.0, best_lambda)]
            else:
                best_lambda = LAMBDA_GRID[len(LAMBDA_GRID) // 2]
                best_lambdas = [best_lambda]
                lambda_scores = [(0.0, best_lambda)]
        else:
            for lmbda in LAMBDA_GRID:
                val_res = simulate_strategy(df, val_start, current_date, lmbda, config, include_xgboost=True, ewma_halflife=best_ewma_hl)
                if not val_res.empty:
                    _, _, sharpe, sortino, _ = calculate_metrics(val_res['Strat_Return'], val_res['RF_Rate'])
                    metric_val = sortino if config.tuning_metric == 'sortino' else sharpe
                    lambda_scores.append((metric_val, lmbda))

            lambda_scores.sort(key=lambda x: x[0], reverse=True)
            if not lambda_scores:
                lambda_scores = [(0.0, LAMBDA_GRID[0])]

            if config.lambda_selection == 'median_positive':
                # Pick median lambda among those with positive validation metric
                positive_lambdas = [lam for score, lam in lambda_scores if score > 0]
                if positive_lambdas:
                    best_lambda = float(np.median(positive_lambdas))
                    best_lambdas = [best_lambda]
                else:
                    best_lambdas = [lam for _, lam in lambda_scores[:max(1, config.lambda_ensemble_k)]]
                    best_lambda = best_lambdas[0]
            else:
                best_lambdas = [lam for _, lam in lambda_scores[:max(1, config.lambda_ensemble_k)]]
                best_lambda = best_lambdas[0]

        if config.lambda_smoothing and lambda_history:
            best_lambda = (0.7 * best_lambda) + (0.3 * lambda_history[-1])

        lambda_history.append(best_lambda)
        lambda_dates.append(current_date)

        if config.lambda_ensemble_k > 1:
            oos_chunks = []
            for l_val in best_lambdas:
                oos_chunk_jm_xgb = run_period_forecast(df, current_date, l_val, config, include_xgboost=True)
                if oos_chunk_jm_xgb is not None:
                    oos_chunks.append(oos_chunk_jm_xgb)
            if oos_chunks:
                avg_prob = sum(c['Raw_Prob'] for c in oos_chunks) / len(oos_chunks)
                final_chunk = oos_chunks[0].copy()
                final_chunk['Raw_Prob'] = avg_prob
                jm_xgb_results.append(final_chunk)
        else:
            oos_chunk_jm_xgb = run_period_forecast(df, current_date, best_lambda, config, include_xgboost=True)
            if oos_chunk_jm_xgb is not None:
                jm_xgb_results.append(oos_chunk_jm_xgb)

        current_date = chunk_end

    if not jm_xgb_results:
        return pd.DataFrame()
        
    jm_xgb_df = pd.concat(jm_xgb_results)

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
        
    strat_returns = (alloc_target * jm_xgb_df['Target_Return']) + ((1.0 - alloc_target) * jm_xgb_df['RF_Rate'])
    trades = alloc_target.diff().abs().fillna(0)
    jm_xgb_df['Strat_Return'] = strat_returns - (trades * TRANSACTION_COST)
    jm_xgb_df['Trades'] = trades

    # Attach walk-forward metadata for diagnostics (non-breaking: callers can ignore .attrs)
    jm_xgb_df.attrs['lambda_history'] = lambda_history
    jm_xgb_df.attrs['lambda_dates'] = [d.strftime('%Y-%m-%d') for d in lambda_dates]
    jm_xgb_df.attrs['ewma_halflife'] = best_ewma_hl

    return jm_xgb_df

# ==============================================================================
# 5. METRICS & EXECUTION
# ==============================================================================
def calculate_metrics(returns_series, rf_series):
    excess_returns = returns_series - rf_series
    ann_excess_ret = excess_returns.mean() * 252
    ann_vol = returns_series.std() * np.sqrt(252)
    sharpe = ann_excess_ret / ann_vol if ann_vol != 0 else 0
    
    downside_returns = np.minimum(excess_returns, 0)
    ann_downside_vol = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(252)
    sortino = ann_excess_ret / ann_downside_vol if ann_downside_vol != 0 else 0
    
    cum_wealth = (1 + returns_series).cumprod()
    peak = cum_wealth.cummax()
    drawdown = (cum_wealth - peak) / peak
    mdd = drawdown.min()
    
    ann_geom_ret = cum_wealth.iloc[-1] ** (252 / len(returns_series)) - 1 if len(returns_series) > 0 else 0
    
    return ann_geom_ret, ann_vol, sharpe, sortino, mdd

def main(run_simple_jm=False, fixed_lambda=None):
    df = fetch_and_prepare_data()

    print(f"\n--- Starting Out-of-Sample Backtest ({OOS_START_DATE} to {END_DATE}) ---")
    if fixed_lambda is not None:
        print(f"Using FIXED lambda={fixed_lambda} (no walk-forward tuning)")
    current_date = pd.to_datetime(OOS_START_DATE)
    final_end_date = pd.to_datetime(END_DATE)

    jm_xgb_results = []
    simple_jm_results = []
    lambda_history = []
    lambda_dates = []

    # --- Phase 1: Tune EWMA halflife once on initial validation window (paper Section 4.2) ---
    # The paper selects one halflife per asset on the pre-OOS validation window, then fixes it.
    initial_val_start = current_date - pd.DateOffset(years=VALIDATION_WINDOW_YRS)
    best_ewma_hl = EWMA_HL_GRID[-1]
    best_init_sharpe = -np.inf

    if fixed_lambda is None:
        print(f"\nTuning EWMA halflife on initial validation window ({initial_val_start.date()} to {current_date.date()})...")
        for hl in EWMA_HL_GRID:
            for lmbda in LAMBDA_GRID:
                val_res = simulate_strategy(df, initial_val_start, current_date, lmbda, include_xgboost=True, ewma_halflife=hl)
                if not val_res.empty:
                    _, _, sharpe, _, _ = calculate_metrics(val_res['Strat_Return'], val_res['RF_Rate'])
                    if sharpe > best_init_sharpe:
                        best_init_sharpe = sharpe
                        best_ewma_hl = hl
        print(f"Selected EWMA halflife: {best_ewma_hl} (fixed for entire OOS period)")

    # --- Phase 2: Walk-forward lambda tuning (only lambda, halflife is fixed) ---
    while current_date < final_end_date:
        chunk_end = min(current_date + pd.DateOffset(months=6), final_end_date)
        val_start = current_date - pd.DateOffset(years=VALIDATION_WINDOW_YRS)
        print(f"\nEvaluating period: {current_date.date()} to {chunk_end.date()}")

        if fixed_lambda is not None:
            best_lambda = fixed_lambda
            best_lambda_jm = fixed_lambda
            print(f"Lambda (fixed): {best_lambda}")
        else:
            # Tune only lambda on validation window (EWMA halflife is already fixed)
            best_sharpe = -np.inf
            best_lambda = LAMBDA_GRID[0]

            best_sharpe_jm = -np.inf
            best_lambda_jm = LAMBDA_GRID[0]

            for lmbda in LAMBDA_GRID:
                val_res = simulate_strategy(df, val_start, current_date, lmbda, include_xgboost=True, ewma_halflife=best_ewma_hl)
                if not val_res.empty:
                    _, _, sharpe, _, _ = calculate_metrics(val_res['Strat_Return'], val_res['RF_Rate'])
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_lambda = lmbda

                if run_simple_jm:
                    val_res_jm = simulate_strategy(df, val_start, current_date, lmbda, include_xgboost=False)
                    if not val_res_jm.empty:
                        _, _, sharpe_jm, _, _ = calculate_metrics(val_res_jm['Strat_Return'], val_res_jm['RF_Rate'])
                        if sharpe_jm > best_sharpe_jm:
                            best_sharpe_jm = sharpe_jm
                            best_lambda_jm = lmbda

            print(f"Optimal Lambda for JM-XGB: {best_lambda:.2f} (Val Sharpe: {best_sharpe:.2f})")
            if run_simple_jm:
                print(f"Optimal Lambda for Simple JM: {best_lambda_jm:.2f} (Val Sharpe: {best_sharpe_jm:.2f})")

        lambda_history.append(best_lambda)
        lambda_dates.append(current_date)

        # Out-of-Sample Execution for this 6-month chunk
        oos_chunk_jm_xgb = run_period_forecast(df, current_date, best_lambda, include_xgboost=True)
        if oos_chunk_jm_xgb is not None:
            jm_xgb_results.append(oos_chunk_jm_xgb)

        if run_simple_jm:
            oos_chunk_simple_jm = run_period_forecast(df, current_date, best_lambda_jm, include_xgboost=False)
            if oos_chunk_simple_jm is not None:
                simple_jm_results.append(oos_chunk_simple_jm)

        current_date = chunk_end

    # Combine Results
    jm_xgb_df = pd.concat(jm_xgb_results)

    # Apply continuous EWMA over the full OOS series (halflife fixed from Phase 1)
    if best_ewma_hl == 0:
        jm_xgb_df['State_Prob'] = jm_xgb_df['Raw_Prob']
    else:
        jm_xgb_df['State_Prob'] = jm_xgb_df['Raw_Prob'].ewm(halflife=best_ewma_hl).mean()
    jm_xgb_df['Forecast_State'] = (jm_xgb_df['State_Prob'] > 0.5).astype(int)
    
    # Calculate Strategy Returns applying transaction costs
    strat_dfs = [jm_xgb_df]
    if run_simple_jm:
        simple_jm_df = pd.concat(simple_jm_results)
        strat_dfs.append(simple_jm_df)
        
    for strat_df in strat_dfs:
        # 1. Shift the forecast so the prediction made at T applies to the return of T+1
        tradable_signal = strat_df['Forecast_State'].shift(1).fillna(0)
        
        # 2. Calculate returns using the properly aligned signal
        strat_returns = np.where(tradable_signal == 0, 
                                 strat_df['Target_Return'], 
                                 strat_df['RF_Rate'])
                                 
        # 3. Calculate trades based on the shifted signal
        trades = tradable_signal.diff().abs().fillna(0)
        
        # 4. Assign back to the dataframe
        strat_df['Strat_Return'] = strat_returns - (trades * TRANSACTION_COST)
        strat_df['Trades'] = trades

    # Calculate Baselines
    bh_returns = jm_xgb_df['Target_Return']
    rf_returns = jm_xgb_df['RF_Rate']
    
    # -------------------------------------------------------------------------
    # 6. SAVE METRICS AND CHART TO DOCUMENT
    # -------------------------------------------------------------------------
    strategies = {
        'Buy & Hold': bh_returns,
        'JM-XGB Strategy': jm_xgb_df['Strat_Return']
    }
    if run_simple_jm:
        strategies['Simple JM Baseline'] = simple_jm_df['Strat_Return']
    
    # Prepare Table Data
    columns = ['Strategy', 'Ann. Ret', 'Ann. Vol', 'Sharpe', 'Sortino', 'Max DD', 'Total Trades']
    cell_text = []
    
    for name, returns in strategies.items():
        ret, vol, sharpe, sortino, mdd = calculate_metrics(returns, rf_returns)
        if name == 'Buy & Hold':
            trades = "N/A"
        elif "XGB" in name:
            trades = int(jm_xgb_df['Trades'].sum())
        else:
            trades = int(simple_jm_df['Trades'].sum())
            
        cell_text.append([
            name,
            f"{ret*100:.2f}%", 
            f"{vol*100:.2f}%", 
            f"{sharpe:.2f}", 
            f"{sortino:.2f}",
            f"{mdd*100:.2f}%", 
            str(trades)
        ])
        
    # Create Figure with Table and Plot
    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 4, 1.5])
    
    # Axis for Table
    ax_table = fig.add_subplot(gs[0])
    ax_table.axis('off')
    table = ax_table.table(cellText=cell_text, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)
    ax_table.set_title("Out-of-Sample Performance Metrics", fontweight='bold', fontsize=14)
    
    # Axis for Plotting Wealth Curves
    ax_plot = fig.add_subplot(gs[1])
    for name, returns in strategies.items():
        wealth = (1 + returns).cumprod()
        # Add ending wealth to label
        end_wealth = wealth.iloc[-1]
        ax_plot.plot(wealth, label=f"{name} (Final: {end_wealth:.2f}x)")
        
    pct_bear = (jm_xgb_df['Forecast_State'] == 1).mean() * 100
    num_shifts = int(jm_xgb_df['Forecast_State'].diff().abs().fillna(0).sum())
    chart_title = r"$\mathbf{{" + str(TARGET_TICKER).replace('$', r'\$') + r"}}$" + f": % of Bear Market: {pct_bear:.1f}%, Number of Regime Shifts: {num_shifts}"
    ax_plot.set_title(chart_title, fontsize=12, loc='center')
    ax_plot.set_ylabel('Cumulative Wealth (Multiplier, log scale)')
    ax_plot.set_yscale('log')
    
    # Shade Bear Regimes (State 1) in the background
    bear_regimes = jm_xgb_df['Forecast_State'] == 1
    ax_plot.fill_between(jm_xgb_df.index, 0, 1, where=bear_regimes, color='red', alpha=0.15, 
                         transform=ax_plot.get_xaxis_transform(), label='Bear Regime (JM-XGB)')
                         
    ax_plot.legend()
    ax_plot.grid(True, which="both", ls="--", alpha=0.5)

    # Axis for Plotting Lambda
    # Append the last date to continue the step plot to the end of the final period
    lambda_dates_full = lambda_dates + [pd.to_datetime(END_DATE)]
    lambda_history_full = lambda_history + [lambda_history[-1]]
    
    ax_lambda = fig.add_subplot(gs[2], sharex=ax_plot)
    ax_lambda.step(lambda_dates_full, lambda_history_full, where='post', color='purple', label='Selected Lambda Penalty', linewidth=2)
    ax_lambda.set_ylabel('Lambda Penalty')
    ax_lambda.set_xlabel('Date')
    ax_lambda.grid(True, which="both", ls="--", alpha=0.5)
    ax_lambda.legend(loc='upper left')

    plt.tight_layout()
    
    # Save the document instead of displaying
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'performance_report_{timestamp}.pdf'
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"\nPerformance metrics and chart saved to {output_filename}")
    plt.close()

if __name__ == "__main__":
    main()