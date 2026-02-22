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

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CONFIGURATION: Change Asset and Time Periods Here
# ==============================================================================
TARGET_TICKER = '^GSPC'         # S&P 500
BOND_TICKER = 'VBMFX'           # Vanguard Total Bond Market (Proxy for US Agg Bond)
RISK_FREE_TICKER = '^IRX'       # 13-Week Treasury Bill (Proxy for Risk-Free)
VIX_TICKER = '^VIX'             # VIX Index

# Timeline
START_DATE_DATA = '1987-01-01'  # Need data way before 1999 to allow for 11-year lookbacks
OOS_START_DATE = '2004-01-01'   # Out-of-sample testing begins
END_DATE = '2026-01-01'         # End of testing period

# Transaction costs
TRANSACTION_COST = 0.0005       # 5 basis points (0.05%)

# Lambda candidate grid for Jump Model (reduced to 4 for computational speed in testing)
# In production, expand this array (e.g., np.logspace(0, 2, 10))
LAMBDA_GRID = [1.0, 10.0, 50.0, 100.0] 

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
    cache_file = 'data_cache.pkl'
    if os.path.exists(cache_file):
        print("Loading data from local cache...")
        return pd.read_pickle(cache_file)
        
    print("Fetching data from Yahoo Finance and FRED...")
    
    # 1. Fetch main assets
    data = yf.download([TARGET_TICKER, BOND_TICKER, RISK_FREE_TICKER, VIX_TICKER], 
                       start=START_DATE_DATA, end=END_DATE, auto_adjust=False)['Adj Close']
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
        ewm_dd = downside_returns.ewm(halflife=hl).std().fillna(0)
        features[f'DD_log_{hl}'] = np.log(ewm_dd + 1e-8)
        
    # EWM Average Return
    for hl in [5, 10, 21]:
        features[f'Avg_Ret_{hl}'] = features['Excess_Return'].ewm(halflife=hl).mean()
        
    # EWM Sortino Ratio
    for hl in [5, 10, 21]:
        ewm_dd_raw = downside_returns.ewm(halflife=hl).std().fillna(1e-8)
        features[f'Sortino_{hl}'] = features[f'Avg_Ret_{hl}'] / ewm_dd_raw

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

def run_period_forecast(df, current_date, lambda_penalty, include_xgboost=True):
    """
    Runs the identification (JM) and forecasting (XGB) for a specific date using 
    an 11-year lookback window.
    """
    cache_key = (current_date, lambda_penalty, include_xgboost)
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
        
    # Shift labels forward by 1 day to create prediction targets
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
        
        # Shift the labels forward by 1 day to create the forecast for tomorrow
        forecasts = np.roll(oos_states, 1)
        forecasts[0] = identified_states[-1] # First day uses last known state from training
        
        oos_df['Forecast_State'] = forecasts
        result = oos_df[['Target_Return', 'RF_Rate', 'Forecast_State']]
        _forecast_cache[cache_key] = result
        return result

    # 2. Regime Forecasting (XGBoost)
    macro_features = ['Yield_2Y_EWMA_diff', 'Yield_Slope_EWMA_10', 'Yield_Slope_EWMA_diff_21', 
                      'VIX_EWMA_log_diff', 'Stock_Bond_Corr']
    all_features = return_features + macro_features
    
    X_train_xgb = train_df[all_features]
    y_train_xgb = train_df['Target_State']
    X_oos_xgb = oos_df[all_features]
    
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb.fit(X_train_xgb, y_train_xgb)
    
    # Get probabilities for Class 1 (Bearish)
    oos_probs = xgb.predict_proba(X_oos_xgb)[:, 1]
    
    # Apply exponential smoothing (halflife 8 days) to probabilities
    smoothed_probs = pd.Series(oos_probs).ewm(halflife=8).mean().values
    
    # Threshold at 0.5
    oos_df['Forecast_State'] = (smoothed_probs > 0.5).astype(int)
    
    result = oos_df[['Target_Return', 'RF_Rate', 'Forecast_State']]
    _forecast_cache[cache_key] = result
    return result

def simulate_strategy(df, start_date, end_date, lambda_penalty, include_xgboost=True):
    """Simulates the entire period in 6-month chunks for a given lambda."""
    results = []
    current_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    while current_date < end_date:
        res = run_period_forecast(df, current_date, lambda_penalty, include_xgboost)
        if res is not None:
            results.append(res)
        current_date += pd.DateOffset(months=6)
        
    if not results:
        return pd.DataFrame()
        
    full_res = pd.concat(results)
    
    # FIX: Shift the forecast state by 1 day to prevent look-ahead bias in validation! 
    # Forecast made at end of Day T applies to Return of Day T+1
    trading_signals = full_res['Forecast_State'].shift(1).fillna(0)
    
    # Calculate 0/1 Strategy Returns
    # State 0 (Bullish) = Target Asset, State 1 (Bearish) = Risk-Free
    strat_returns = np.where(trading_signals == 0, 
                             full_res['Target_Return'], 
                             full_res['RF_Rate'])
    full_res['Strat_Return'] = strat_returns
    
    # Apply Transaction Costs (when forecast changes)
    trades = trading_signals.diff().abs().fillna(0)
    full_res['Strat_Return'] -= trades * TRANSACTION_COST
    
    return full_res

# ==============================================================================
# 5. METRICS & EXECUTION
# ==============================================================================
def calculate_metrics(returns_series, rf_series):
    excess_returns = returns_series - rf_series
    ann_excess_ret = excess_returns.mean() * 252
    ann_vol = returns_series.std() * np.sqrt(252)
    sharpe = ann_excess_ret / ann_vol if ann_vol != 0 else 0
    
    downside_returns = np.minimum(excess_returns, 0)
    ann_downside_vol = downside_returns.std() * np.sqrt(252)
    sortino = ann_excess_ret / ann_downside_vol if ann_downside_vol != 0 else 0
    
    cum_wealth = (1 + returns_series).cumprod()
    peak = cum_wealth.cummax()
    drawdown = (cum_wealth - peak) / peak
    mdd = drawdown.min()
    
    ann_geom_ret = cum_wealth.iloc[-1] ** (252 / len(returns_series)) - 1 if len(returns_series) > 0 else 0
    
    return ann_geom_ret, ann_vol, sharpe, sortino, mdd

def main(run_simple_jm=False):
    df = fetch_and_prepare_data()
    
    print(f"\n--- Starting Out-of-Sample Backtest ({OOS_START_DATE} to {END_DATE}) ---")
    current_date = pd.to_datetime(OOS_START_DATE)
    final_end_date = pd.to_datetime(END_DATE)
    
    jm_xgb_results = []
    simple_jm_results = []
    lambda_history = []
    lambda_dates = []
    
    # Rolling Walk-Forward Optimization
    while current_date < final_end_date:
        chunk_end = min(current_date + pd.DateOffset(months=6), final_end_date)
        val_start = current_date - pd.DateOffset(years=5)
        print(f"\nEvaluating period: {current_date.date()} to {chunk_end.date()}")
        
        # 1. Hyperparameter Tuning on Validation Window (for JM-XGB)
        best_sharpe = -np.inf
        best_lambda = LAMBDA_GRID[0]
        
        for lmbda in LAMBDA_GRID:
            val_res = simulate_strategy(df, val_start, current_date, lmbda, include_xgboost=True)
            if not val_res.empty:
                _, _, sharpe, _, _ = calculate_metrics(val_res['Strat_Return'], val_res['RF_Rate'])
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_lambda = lmbda
                    
        print(f"Optimal Lambda selected for JM-XGB: {best_lambda} (Val Sharpe: {best_sharpe:.2f})")
        lambda_history.append(best_lambda)
        lambda_dates.append(current_date)
        
        # 2. Out-of-Sample Execution for this 6-month chunk
        # JM-XGB
        oos_chunk_jm_xgb = run_period_forecast(df, current_date, best_lambda, include_xgboost=True)
        if oos_chunk_jm_xgb is not None:
            jm_xgb_results.append(oos_chunk_jm_xgb)
            
        # Simple JM (Using a default lambda for the simple baseline to save compute, 
        # normally you'd tune this separately)
        if run_simple_jm:
            oos_chunk_simple_jm = run_period_forecast(df, current_date, 10.0, include_xgboost=False)
            if oos_chunk_simple_jm is not None:
                simple_jm_results.append(oos_chunk_simple_jm)
            
        current_date = chunk_end

    # Combine Results
    jm_xgb_df = pd.concat(jm_xgb_results)
    
    # Calculate Strategy Returns applying transaction costs
    strat_dfs = [jm_xgb_df]
    if run_simple_jm:
        simple_jm_df = pd.concat(simple_jm_results)
        strat_dfs.append(simple_jm_df)
        
    for strat_df in strat_dfs:
        strat_returns = np.where(strat_df['Forecast_State'] == 0, 
                                 strat_df['Target_Return'], 
                                 strat_df['RF_Rate'])
        trades = strat_df['Forecast_State'].diff().abs().fillna(0)
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