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

# Timeline
START_DATE_DATA = '1987-01-01'  # Need data way before 1999 to allow for 11-year lookbacks
OOS_START_DATE = '2007-01-01'   # Out-of-sample testing begins (paper: 2007-2023)
END_DATE = '2026-01-01'         # End of testing period

# Transaction costs
TRANSACTION_COST = 0.0005       # 5 basis points (0.05%)

# Validation Window (Years)
VALIDATION_WINDOW_YRS = 5

# Lambda candidate grid for Jump Model (paper: 0.0 to 100.0, log-spaced)
# Reduced from 20 to 10 candidates to limit walk-forward overfitting on single-asset backtest
LAMBDA_GRID = [0.0] + list(np.logspace(0, 2, 10))

# EWMA halflife candidates for probability smoothing (paper: Section 4.2)
EWMA_HL_GRID = [0, 2, 4, 8]

# ==============================================================================
# 2. STATISTICAL JUMP MODEL (Implementation from scratch)
# ==============================================================================
class StatisticalJumpModel:
    """
    Implements a discrete 2-state Statistical Jump Model using alternating optimization
    (K-means style updates + Viterbi algorithm for state sequence with jump penalty).
    """
    def __init__(self, n_states=2, lambda_penalty=10.0, max_iter=20):
        self.n_states = n_states
        self.lambda_penalty = lambda_penalty
        self.max_iter = max_iter
        self.means = None

    def fit_predict(self, X):
        X_arr = np.array(X)
        n_samples, n_features = X_arr.shape
        
        # Initialize means randomly from the data
        np.random.seed(42)
        idx = np.random.choice(n_samples, self.n_states, replace=False)
        self.means = X_arr[idx].copy()
        
        states = np.zeros(n_samples, dtype=int)
        
        for iteration in range(self.max_iter):
            # Step 1: Viterbi decoding to find optimal state sequence given means
            # Scaled squared L2 distance as loss (vectorized)
            distances = 0.5 * np.sum((X_arr[:, None, :] - self.means[None, :, :])**2, axis=2)
                
            cost_matrix = np.zeros((n_samples, self.n_states))
            back_pointers = np.zeros((n_samples, self.n_states), dtype=int)
            
            cost_matrix[0] = distances[0]
            
            # Forward pass
            if self.n_states == 2:
                # Optimized fast path for 2 states
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
                # Generalized path for arbitrary states
                penalty_matrix = self.lambda_penalty * (1 - np.eye(self.n_states))
                for t in range(1, n_samples):
                    trans_costs = cost_matrix[t-1, :, None] + penalty_matrix
                    best_prev_states = np.argmin(trans_costs, axis=0)
                    cost_matrix[t] = trans_costs[best_prev_states, np.arange(self.n_states)] + distances[t]
                    back_pointers[t] = best_prev_states
            
            # Backward pass (traceback)
            new_states = np.zeros(n_samples, dtype=int)
            new_states[-1] = np.argmin(cost_matrix[-1])
            for t in range(n_samples - 2, -1, -1):
                new_states[t] = back_pointers[t+1, new_states[t+1]]
                
            # Step 2: Update means given new states
            for k in range(self.n_states):
                mask = (new_states == k)
                if np.sum(mask) > 0:
                    self.means[k] = np.mean(X_arr[mask], axis=0)
            
            # Check convergence
            if np.array_equal(states, new_states):
                break
            states = new_states
            
        return states

    def predict_online(self, X, last_known_state):
        """
        Simulates online/real-time assignment of new data points to clusters,
        incorporating the jump penalty to prevent excessive switching.
        """
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

# ==============================================================================
# 3. DATA FETCHING & FEATURE ENGINEERING
# ==============================================================================
import os

def fetch_and_prepare_data():
    cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache', 'data_cache.pkl')
    if os.path.exists(cache_file):
        print("Loading data from local cache...")
        return pd.read_pickle(cache_file)
        
    print("Fetching data from Yahoo Finance and FRED...")
    
    # 1. Fetch main assets
    tickers = [TARGET_TICKER, BOND_TICKER, RISK_FREE_TICKER, VIX_TICKER]
    
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
    
    # 2. Fetch FRED Macro data (2-Year and 10-Year Treasury Yields)
    fred_data = web.DataReader(['DGS2', 'DGS10'], 'fred', START_DATE_DATA, END_DATE)
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
    # Downside Deviation (log scale)
    downside_returns = np.minimum(features['Excess_Return'], 0)
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
    
    # Stock-Bond Correlation
    bond_returns = df[BOND_TICKER].pct_change().fillna(0)
    features['Stock_Bond_Corr'] = target_returns.rolling(window=252).corr(bond_returns).fillna(0)
    
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
        
    cache_key = (current_date, lambda_penalty, include_xgboost, constrain_xgb, config.name)
    if cache_key in _forecast_cache:
        return _forecast_cache[cache_key]

    train_start = current_date - pd.DateOffset(years=11)
    
    # Filter 11-year training data
    train_df = df[(df.index >= train_start) & (df.index < current_date)].copy()
    if len(train_df) < 252 * 5: # Failsafe if not enough data
        _forecast_cache[cache_key] = None
        return None
        
    # Standardize Return Features for JM
    return_features = [c for c in train_df.columns if c.startswith(('DD_', 'Avg_Ret_', 'Sortino_'))]
    X_train_jm = train_df[return_features]
    X_train_jm = (X_train_jm - X_train_jm.mean()) / X_train_jm.std()
    
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
        # Standardize using training mean/std
        X_oos_jm = (X_oos_jm - train_df[return_features].mean()) / train_df[return_features].std()
        
        # Predict day-by-day to simulate real-time tracking
        oos_states = jm.predict_online(X_oos_jm.values, last_known_state=identified_states[-1])
        
        oos_df['Forecast_State'] = oos_states
        result = oos_df[['Target_Return', 'RF_Rate', 'Forecast_State']]
        _forecast_cache[cache_key] = result
        return result

    # 2. Regime Forecasting (XGBoost)
    macro_features = ['Yield_2Y_EWMA_diff', 'Yield_Slope_EWMA_10', 'Yield_Slope_EWMA_diff_21', 
                      'VIX_EWMA_log_diff', 'Stock_Bond_Corr']
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
    
    # Calculate SHAP values
    import shap
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_oos_xgb)
    
    # Calculate JM online regimes for the OOS period to serve as True Labels for Audit
    X_oos_jm = oos_df[return_features]
    X_oos_jm = (X_oos_jm - train_df[return_features].mean()) / train_df[return_features].std()
    oos_states = jm.predict_online(X_oos_jm.values, last_known_state=identified_states[-1])
    oos_df['JM_Target_State'] = oos_states
    
    # Add SHAP values to dataframe
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
    
    # Keep the feature values as well for scatter/dependence plots later if needed
    for col in all_features:
        oos_df[f'Feature_{col}'] = oos_df[col]
        
    cols_to_keep = ['Target_Return', 'RF_Rate', 'Raw_Prob', 'SHAP_Base_Value', 'JM_Target_State'] + [f'SHAP_{col}' for col in all_features] + [f'Feature_{col}' for col in all_features]
    
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
        for lmbda in LAMBDA_GRID:
            val_res = simulate_strategy(df, val_start, current_date, lmbda, config, include_xgboost=True, ewma_halflife=best_ewma_hl)
            if not val_res.empty:
                _, _, sharpe, sortino, _ = calculate_metrics(val_res['Strat_Return'], val_res['RF_Rate'])
                metric_val = sortino if config.tuning_metric == 'sortino' else sharpe
                lambda_scores.append((metric_val, lmbda))
        
        lambda_scores.sort(key=lambda x: x[0], reverse=True)
        if not lambda_scores:
            lambda_scores = [(0.0, LAMBDA_GRID[0])]
            
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
        
    ax_plot.set_title(f'Wealth Curves: {OOS_START_DATE} to {END_DATE}', fontsize=12)
    ax_plot.set_ylabel('Cumulative Wealth (Multiplier)')
    
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