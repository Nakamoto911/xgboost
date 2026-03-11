import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import pickle
import json
import tempfile
import time
import importlib
from datetime import datetime
import main as backend

# Force reload backend because streamlit caches imported modules!
importlib.reload(backend)

from config import StrategyConfig

st.set_page_config(layout="wide")

st.title("XGBoost Strategy Backtest Portal")
export_container = st.container()

# =============================================================================
# Experiment Presets
# =============================================================================
EXPERIMENT_PRESETS = {
    "1. Paper Baseline": StrategyConfig(name="1. Paper Baseline"),
    "2. Sortino Tuned": StrategyConfig(name="2. Sortino Tuned", tuning_metric="sortino"),
    "3. Conservative Threshold (0.6)": StrategyConfig(name="3. Conservative Threshold (0.6)", prob_threshold=0.60),
    "4. Continuous Allocation": StrategyConfig(name="4. Continuous Allocation", allocation_style="continuous"),
    "5. Lambda Smoothing": StrategyConfig(name="5. Lambda Smoothing", lambda_smoothing=True),
    "6. Expanding Window": StrategyConfig(name="6. Expanding Window", validation_window_type="expanding"),
    "7. Lambda Ensemble (Top 3)": StrategyConfig(name="7. Lambda Ensemble (Top 3)", lambda_ensemble_k=3),
    "8. The Ultimate Combo": StrategyConfig(
        name="8. The Ultimate Combo",
        tuning_metric="sortino", prob_threshold=0.55,
        allocation_style="continuous", lambda_smoothing=True,
    ),
    "9. Expanding + Lambda Smoothing": StrategyConfig(
        name="9. Expanding + Lambda Smoothing",
        validation_window_type="expanding", lambda_smoothing=True,
    ),
    "10. Median-Positive Lambda": StrategyConfig(
        name="10. Median-Positive Lambda", lambda_selection="median_positive",
    ),
    "11. Sub-Window Consensus": StrategyConfig(
        name="11. Sub-Window Consensus", lambda_subwindow_consensus=True,
    ),
}
PRESET_NAMES = list(EXPERIMENT_PRESETS.keys()) + ["Custom"]

# =============================================================================
# Session State Initialization
# =============================================================================
_defaults = {
    'start_date_input': backend.START_DATE_DATA,
    'oos_start_input': backend.OOS_START_DATE,
    'end_date_input': backend.END_DATE,
    'target_ticker_input': backend.TARGET_TICKER,
    'bond_ticker_input': backend.BOND_TICKER,
    'rf_ticker_input': backend.RISK_FREE_TICKER,
    'vix_ticker_input': backend.VIX_TICKER,
    'transaction_cost_input': float(backend.TRANSACTION_COST),
    'val_window_input': backend.VALIDATION_WINDOW_YRS,
    'tuning_metric': "sharpe",
    'validation_window_type': "rolling",
    'lambda_smoothing': False,
    'prob_threshold': 0.50,
    'allocation_style': "binary",
    'lambda_ensemble_k': 1,
    'lambda_selection': "best",
    'lambda_subwindow_consensus': False,
    'xgb_max_depth': 6,
    'xgb_n_estimators': 100,
    'xgb_learning_rate': 0.3,
    'xgb_subsample': 1.0,
    'xgb_colsample_bytree': 1.0,
    'xgb_reg_alpha': 0.0,
    'xgb_reg_lambda': 1.0,
    'calculate_shap': False,
}
for key, val in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

@st.cache_data(show_spinner=False)
def get_earliest_date(ticker):
    import yfinance as yf
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="max")
        if not hist.empty:
            return hist.index.min().strftime('%Y-%m-%d')
    except Exception:
        pass
    return "Unknown"

def on_preset_change():
    preset_name = st.session_state.experiment_preset
    if preset_name == "Custom":
        return
    cfg = EXPERIMENT_PRESETS[preset_name]
    st.session_state.tuning_metric = cfg.tuning_metric
    st.session_state.validation_window_type = cfg.validation_window_type
    st.session_state.lambda_smoothing = cfg.lambda_smoothing
    st.session_state.prob_threshold = cfg.prob_threshold
    st.session_state.allocation_style = cfg.allocation_style
    st.session_state.lambda_ensemble_k = cfg.lambda_ensemble_k
    st.session_state.lambda_selection = cfg.lambda_selection
    st.session_state.lambda_subwindow_consensus = cfg.lambda_subwindow_consensus
    st.session_state.xgb_max_depth = 6
    st.session_state.xgb_n_estimators = 100
    st.session_state.xgb_learning_rate = 0.3
    st.session_state.xgb_subsample = 1.0
    st.session_state.xgb_colsample_bytree = 1.0
    st.session_state.xgb_reg_alpha = 0.0
    st.session_state.xgb_reg_lambda = 1.0

def on_strategy_param_change():
    """Auto-switch to 'Custom' if user manually changes a param that no longer matches the preset."""
    preset_name = st.session_state.get('experiment_preset', 'Custom')
    if preset_name == 'Custom':
        return
    cfg = EXPERIMENT_PRESETS.get(preset_name)
    if cfg is None:
        return
    if (st.session_state.tuning_metric == cfg.tuning_metric and
        st.session_state.validation_window_type == cfg.validation_window_type and
        st.session_state.lambda_smoothing == cfg.lambda_smoothing and
        abs(st.session_state.prob_threshold - cfg.prob_threshold) < 0.001 and
        st.session_state.allocation_style == cfg.allocation_style and
        st.session_state.lambda_ensemble_k == cfg.lambda_ensemble_k and
        st.session_state.lambda_selection == cfg.lambda_selection and
        st.session_state.lambda_subwindow_consensus == cfg.lambda_subwindow_consensus and
        st.session_state.xgb_max_depth == 6 and
        st.session_state.xgb_n_estimators == 100 and
        abs(st.session_state.xgb_learning_rate - 0.3) < 0.001 and
        abs(st.session_state.xgb_reg_alpha - 0.0) < 0.001 and
        abs(st.session_state.xgb_reg_lambda - 1.0) < 0.001 and
        abs(st.session_state.xgb_subsample - 1.0) < 0.001 and
        abs(st.session_state.xgb_colsample_bytree - 1.0) < 0.001):
        return
    st.session_state.experiment_preset = 'Custom'

# =============================================================================
# Sidebar
# =============================================================================
st.sidebar.header("Experiment Preset")
st.sidebar.selectbox(
    "Strategy", PRESET_NAMES, index=0,
    key='experiment_preset', on_change=on_preset_change,
    help="Select a predefined experiment configuration. All parameters below will be filled automatically. Choose 'Custom' to set parameters manually."
)

with st.sidebar.form("config_form"):
    with st.expander("1. Data Source", expanded=True):
        val_win = st.session_state.get('val_window_input', backend.VALIDATION_WINDOW_YRS)
        oos_val = st.session_state.get('oos_start_input', backend.OOS_START_DATE)
        
        def get_earliest_info(ticker):
            earliest = get_earliest_date(ticker)
            if earliest == "Unknown":
                return "", None
            try:
                earliest_dt = pd.to_datetime(earliest)
                oos_dt = pd.to_datetime(oos_val)
                req_start = oos_dt - pd.DateOffset(years=int(11 + val_win))
                if earliest_dt > req_start:
                    safe_oos = earliest_dt + pd.DateOffset(years=int(11 + val_win))
                    return " ⚠️", f"Earliest data: {earliest} (Insufficient history! Model needs 11Y for features + {val_win}Y validation before OOS. Safe OOS start: {safe_oos.strftime('%Y-%m-%d')})"
                return "", f"Earliest data: {earliest}"
            except Exception:
                return "", f"Earliest data: {earliest}"
        
        tt_val = st.session_state.get('target_ticker_input', backend.TARGET_TICKER)
        tt_suffix, tt_help = get_earliest_info(tt_val if tt_val else backend.TARGET_TICKER)
        target_ticker = st.text_input(
            f"Target Ticker{tt_suffix}", 
            value=tt_val if tt_val else backend.TARGET_TICKER, 
            key='target_ticker_input',
            help=tt_help
        )

        bond_suffix, bond_help = get_earliest_info(backend.BOND_TICKER)
        bond_ticker = st.text_input(f"Bond Ticker{bond_suffix}", value=backend.BOND_TICKER, key='bond_ticker_input', help=bond_help)

        rf_suffix, rf_help = get_earliest_info(backend.RISK_FREE_TICKER)
        risk_free_ticker = st.text_input(f"Risk-Free Ticker{rf_suffix}", value=backend.RISK_FREE_TICKER, key='rf_ticker_input', help=rf_help)

        vix_suffix, vix_help = get_earliest_info(backend.VIX_TICKER)
        vix_ticker = st.text_input(f"VIX Ticker{vix_suffix}", value=backend.VIX_TICKER, key='vix_ticker_input', help=vix_help)

        # Auto-calculate the latest of all earliest dates
        max_earliest = None
        for t in [target_ticker, bond_ticker, risk_free_ticker, vix_ticker]:
            dt_str = get_earliest_date(t)
            if dt_str != "Unknown":
                dt = pd.to_datetime(dt_str)
                if max_earliest is None or dt > max_earliest:
                    max_earliest = dt
        
        default_start = max_earliest.strftime('%Y-%m-%d') if max_earliest else backend.START_DATE_DATA
        
        # If the currently selected start date is older than the max earliest date, overwrite it
        current_sd = st.session_state.get('start_date_input', default_start)
        try:
            if pd.to_datetime(current_sd) < pd.to_datetime(default_start):
                st.session_state['start_date_input'] = default_start
                current_sd = default_start
        except Exception:
            pass
            
        start_date_data = st.text_input("Data Start Date", value=current_sd, key='start_date_input', help=f"Auto-calculated earliest date where all tickers have data: {default_start}")
        oos_val = st.session_state.get('oos_start_input', backend.OOS_START_DATE)
        oos_start_date = st.text_input("OOS Start Date", value=oos_val if oos_val else backend.OOS_START_DATE, key='oos_start_input')
        end_date = st.text_input("End Date", value=backend.END_DATE, key='end_date_input')

    with st.expander("2. Feature Engineering", expanded=False):
        st.write("Feature parameters configured in code (currently using default).")
        # In the future, feature engineering inputs like window sizes can go here.

    with st.expander("3. Jump Model (JM) Parameters", expanded=False):
        st.selectbox("Validation Window Type", ["rolling", "expanding"], key='validation_window_type')
        validation_window = st.number_input("Validation Window (Years)", min_value=1, max_value=20, key='val_window_input')
        st.checkbox("Lambda Smoothing", key='lambda_smoothing')
        st.number_input("Probability Threshold", min_value=0.30, max_value=0.70, step=0.05, format="%.2f", key='prob_threshold')
        st.selectbox("Allocation Style", ["binary", "continuous"], key='allocation_style')
        st.number_input("Lambda Ensemble K", min_value=1, max_value=5, key='lambda_ensemble_k')
        st.selectbox("Lambda Selection", ["best", "median_positive"], key='lambda_selection',
                     help="'best' = argmax(validation Sharpe). 'median_positive' = median of all lambdas with positive validation Sharpe (more robust).")
        st.checkbox("Sub-Window Consensus", key='lambda_subwindow_consensus',
                    help="Split validation into 3 overlapping sub-windows, find best lambda in each, take median. More robust to validation noise.")
        transaction_cost = st.number_input("Transaction Cost", value=float(backend.TRANSACTION_COST), format="%.4f", key='transaction_cost_input')

        grid_preset = st.selectbox(
            "Lambda Grid Preset",
            ["Dense Mid-Range (8 points)", "Focused Mid-Range (5 points)", "Focused No-100 (4 points)", "Legacy Wide (11 points)", "Expanded (21 points)", "Custom"],
            key='lambda_grid_preset'
        )

        if grid_preset == "Dense Mid-Range (8 points)":
            backend.LAMBDA_GRID = [4.64, 10.0, 15.0, 21.54, 30.0, 46.42, 70.0, 100.0]
            st.caption("Using: 4.64, 10, 15, 21.54, 30, 46.42, 70, 100 (default)")
        elif grid_preset == "Focused Mid-Range (5 points)":
            backend.LAMBDA_GRID = [4.64, 10.0, 21.54, 46.42, 100.0]
            st.caption("Using: 4.64, 10, 21.54, 46.42, 100")
        elif grid_preset == "Focused No-100 (4 points)":
            backend.LAMBDA_GRID = [4.64, 10.0, 21.54, 46.42]
            st.caption("Using: 4.64, 10, 21.54, 46.42 (best single-asset)")
        elif grid_preset == "Legacy Wide (11 points)":
            backend.LAMBDA_GRID = [0.0] + list(np.logspace(0, 2, 10))
            st.caption("Using: 0.0 + 10 log-spaced points up to 100.0")
        elif grid_preset == "Expanded (21 points)":
            backend.LAMBDA_GRID = [0.0] + list(np.logspace(0, 2, 20))
            st.caption("Using: 0.0 + 20 log-spaced points up to 100.0")
        else:
            lambda_grid_str = st.text_input("Custom Grid (comma separated)", "4.64, 10.0, 15.0, 21.54, 30.0, 46.42, 70.0, 100.0")
            try:
                backend.LAMBDA_GRID = [float(x.strip()) for x in lambda_grid_str.split(',')]
            except ValueError:
                st.error("Invalid Lambda Grid format. Using default.")
                backend.LAMBDA_GRID = [4.64, 10.0, 15.0, 21.54, 30.0, 46.42, 70.0, 100.0]
        st.session_state['lambda_grid_value'] = backend.LAMBDA_GRID

        ewma_grid_preset = st.selectbox(
            "EWMA Halflife Grid",
            ["Asset-Specific (Paper: 0,2,4,8)", "Fast (0,8)", "Custom"],
            key='ewma_grid_preset'
        )
        if ewma_grid_preset == "Asset-Specific (Paper: 0,2,4,8)":
            backend.EWMA_HL_GRID = [0, 2, 4, 8]
        elif ewma_grid_preset == "Fast (0,8)":
            backend.EWMA_HL_GRID = [0, 8]
        else:
            ewma_grid_str = st.text_input("Custom EWMA Grid (comma separated)", "0, 2, 4, 8")
            try:
                backend.EWMA_HL_GRID = [int(x.strip()) for x in ewma_grid_str.split(',')]
            except ValueError:
                st.error("Invalid EWMA Grid format. Using default.")
                backend.EWMA_HL_GRID = [0, 2, 4, 8]
        st.session_state['ewma_grid_value'] = backend.EWMA_HL_GRID

    with st.expander("4. XGBoost Parameters", expanded=False):
        st.selectbox("Tuning Metric", ["sharpe", "sortino"], key='tuning_metric')
        
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("max_depth", min_value=1, max_value=20, key='xgb_max_depth')
            st.number_input("n_estimators", min_value=10, max_value=1000, step=10, key='xgb_n_estimators')
            st.number_input("learning_rate", min_value=0.001, max_value=1.0, format="%.3f", key='xgb_learning_rate')
            st.number_input("reg_alpha (L1)", min_value=0.0, max_value=100.0, format="%.2f", key='xgb_reg_alpha')
        with col2:
            st.number_input("subsample", min_value=0.1, max_value=1.0, format="%.2f", key='xgb_subsample')
            st.number_input("colsample_bytree", min_value=0.1, max_value=1.0, format="%.2f", key='xgb_colsample_bytree')
            st.number_input("reg_lambda (L2)", min_value=0.0, max_value=100.0, format="%.2f", key='xgb_reg_lambda')

    run_simple_jm = st.checkbox("Run Simple JM Baseline", value=False)
    
    run_button = st.form_submit_button("Run Backtest", type="primary")

duration_placeholder = st.sidebar.empty()

# =============================================================================
# Data Fetching (cached)
# =============================================================================
# Force clear Streamlit cache so the updated backend.fetch_and_prepare_data runs (once)
if 'cache_cleared_v2' not in st.session_state:
    st.cache_data.clear()
    st.session_state.cache_cleared_v2 = True

@st.cache_data
def get_cached_data(target, bond, rf, vix, start, end):
    backend.TARGET_TICKER = target
    backend.BOND_TICKER = bond
    backend.RISK_FREE_TICKER = rf
    backend.VIX_TICKER = vix
    backend.START_DATE_DATA = start
    backend.END_DATE = end
    return backend.fetch_and_prepare_data()

# =============================================================================
# Backtest Execution
# =============================================================================
if run_button:
    backtest_start_time = time.time()
    backend._forecast_cache.clear()

    # Build StrategyConfig from sidebar values
    xgb_params = {
        "max_depth": st.session_state.xgb_max_depth,
        "n_estimators": st.session_state.xgb_n_estimators,
        "learning_rate": st.session_state.xgb_learning_rate,
        "subsample": st.session_state.xgb_subsample,
        "colsample_bytree": st.session_state.xgb_colsample_bytree,
        "reg_alpha": st.session_state.xgb_reg_alpha,
        "reg_lambda": st.session_state.xgb_reg_lambda,
    }

    # In Performance Tracker, SHAP is forcefully disabled for speed
    config = StrategyConfig(
        name=st.session_state.experiment_preset,
        tuning_metric=st.session_state.tuning_metric,
        validation_window_type=st.session_state.validation_window_type,
        lambda_smoothing=st.session_state.lambda_smoothing,
        prob_threshold=st.session_state.prob_threshold,
        allocation_style=st.session_state.allocation_style,
        lambda_ensemble_k=st.session_state.lambda_ensemble_k,
        lambda_selection=st.session_state.lambda_selection,
        lambda_subwindow_consensus=st.session_state.lambda_subwindow_consensus,
        xgb_params=xgb_params,
        calculate_shap=False,
    )

    # Update backend globals
    backend.TARGET_TICKER = target_ticker
    backend.BOND_TICKER = bond_ticker
    backend.RISK_FREE_TICKER = risk_free_ticker
    backend.VIX_TICKER = vix_ticker
    backend.START_DATE_DATA = start_date_data
    backend.OOS_START_DATE = oos_start_date
    backend.END_DATE = end_date
    backend.TRANSACTION_COST = transaction_cost
    backend.VALIDATION_WINDOW_YRS = validation_window

    st.write("Fetching and preparing data...")
    try:
        df = get_cached_data(target_ticker, bond_ticker, risk_free_ticker, vix_ticker, start_date_data, end_date)
        st.success("Data fetched and features prepared successfully!")
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

    st.write(f"Running walk-forward backtest ({backend.OOS_START_DATE} to {backend.END_DATE})  |  **{config.name}**")
    try:
        with open("cache/debug.txt", "w") as f:
            f.write(f"df max: {df.index.max()}\n")
            f.write(f"YF SP500TR length: {len(df)}\n")
    except Exception as e:
        pass


    with st.spinner("Running JM-XGB walk-forward backtest... This may take several minutes."):
        jm_xgb_df = backend.walk_forward_backtest(df, config)

    lambda_history = jm_xgb_df.attrs.get('lambda_history', [])
    lambda_dates = jm_xgb_df.attrs.get('lambda_dates', [])
    best_ewma_hl = jm_xgb_df.attrs.get('ewma_halflife', 8)

    # Simple JM baseline (separate run if requested)
    simple_jm_df = pd.DataFrame()
    if run_simple_jm:
        with st.spinner("Running Simple JM baseline..."):
            current_date = pd.to_datetime(backend.OOS_START_DATE)
            final_end_date = pd.to_datetime(backend.END_DATE)
            simple_jm_results = []
            jm_lambda_history = []
            while current_date < final_end_date:
                chunk_end = min(current_date + pd.DateOffset(months=6), final_end_date)
                if config.validation_window_type == 'expanding':
                    val_start = pd.to_datetime(backend.OOS_START_DATE) - pd.DateOffset(years=backend.VALIDATION_WINDOW_YRS)
                else:
                    val_start = current_date - pd.DateOffset(years=backend.VALIDATION_WINDOW_YRS)
                best_metric_jm = -np.inf
                best_lambda_jm = backend.LAMBDA_GRID[0]
                for lmbda in backend.LAMBDA_GRID:
                    val_res_jm = backend.simulate_strategy(df, val_start, current_date, lmbda, config=config, include_xgboost=False)
                    if not val_res_jm.empty:
                        _, _, sharpe_jm, sortino_jm, _ = backend.calculate_metrics(val_res_jm['Strat_Return'], val_res_jm['RF_Rate'])
                        metric_jm = sortino_jm if config.tuning_metric == 'sortino' else sharpe_jm
                        if metric_jm > best_metric_jm:
                            best_metric_jm = metric_jm
                            best_lambda_jm = lmbda
                if config.lambda_smoothing and jm_lambda_history:
                    best_lambda_jm = (0.7 * best_lambda_jm) + (0.3 * jm_lambda_history[-1])
                jm_lambda_history.append(best_lambda_jm)
                oos_chunk = backend.run_period_forecast(df, current_date, best_lambda_jm, config=config, include_xgboost=False)
                if oos_chunk is not None:
                    simple_jm_results.append(oos_chunk)
                current_date = chunk_end
            if simple_jm_results:
                simple_jm_df = pd.concat(simple_jm_results)
                simple_jm_df['Forecast_State'] = simple_jm_df.get('Forecast_State', pd.Series(0, index=simple_jm_df.index))
                tradable_signal = simple_jm_df['Forecast_State'].shift(1).fillna(0)
                simple_jm_df['Strat_Return'] = np.where(tradable_signal == 0, simple_jm_df['Target_Return'], simple_jm_df['RF_Rate']) - (tradable_signal.diff().abs().fillna(0) * transaction_cost)
                simple_jm_df['Trades'] = tradable_signal.diff().abs().fillna(0)

    backtest_duration = time.time() - backtest_start_time

    cache_data = {
        'jm_xgb_df': jm_xgb_df,
        'simple_jm_df': simple_jm_df,
        'lambda_history': lambda_history,
        'lambda_dates': lambda_dates,
        'run_simple_jm': run_simple_jm,
        'oos_start_date': backend.OOS_START_DATE,
        'end_date': backend.END_DATE,
        'backtest_duration': backtest_duration,
        'best_ewma_hl': best_ewma_hl,
        'config_name': config.name,
        'cache_version': 2,
    }
    _backtest_cache_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache', 'backtest_cache.pkl')
    with open(_backtest_cache_path, 'wb') as f:
        pickle.dump(cache_data, f)

# =============================================================================
# Load cached results
# =============================================================================
_backtest_cache_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache', 'backtest_cache.pkl')
cache_loaded = False
if os.path.exists(_backtest_cache_path):
    try:
        with open(_backtest_cache_path, 'rb') as f:
            cache_data = pickle.load(f)

        cache_version = cache_data.get('cache_version', 1)

        if cache_version >= 2:
            # New format: complete DataFrames
            jm_xgb_df = cache_data['jm_xgb_df']
            simple_jm_df = cache_data.get('simple_jm_df', pd.DataFrame())
            run_simple_jm_cached = cache_data.get('run_simple_jm', False) and not simple_jm_df.empty
        else:
            # Legacy format: raw result lists
            jm_xgb_results = cache_data.get('jm_xgb_results', [])
            simple_jm_results = cache_data.get('simple_jm_results', [])
            run_simple_jm_cached = cache_data.get('run_simple_jm', False)
            best_ewma_hl_legacy = cache_data.get('best_ewma_hl', 8)
            if jm_xgb_results:
                jm_xgb_df = pd.concat(jm_xgb_results)
                if 'Raw_Prob' in jm_xgb_df.columns:
                    if best_ewma_hl_legacy == 0:
                        jm_xgb_df['State_Prob'] = jm_xgb_df['Raw_Prob']
                    else:
                        jm_xgb_df['State_Prob'] = jm_xgb_df['Raw_Prob'].ewm(halflife=best_ewma_hl_legacy).mean()
                    jm_xgb_df['Forecast_State'] = (jm_xgb_df['State_Prob'] > 0.5).astype(int)
                tradable_signal = jm_xgb_df['Forecast_State'].shift(1).fillna(0)
                jm_xgb_df['Strat_Return'] = np.where(tradable_signal == 0, jm_xgb_df['Target_Return'], jm_xgb_df['RF_Rate']) - (tradable_signal.diff().abs().fillna(0) * backend.TRANSACTION_COST)
                jm_xgb_df['Trades'] = tradable_signal.diff().abs().fillna(0)
            else:
                jm_xgb_df = pd.DataFrame()
            if run_simple_jm_cached and simple_jm_results:
                simple_jm_df = pd.concat(simple_jm_results)
                tradable_signal = simple_jm_df['Forecast_State'].shift(1).fillna(0)
                simple_jm_df['Strat_Return'] = np.where(tradable_signal == 0, simple_jm_df['Target_Return'], simple_jm_df['RF_Rate']) - (tradable_signal.diff().abs().fillna(0) * backend.TRANSACTION_COST)
                simple_jm_df['Trades'] = tradable_signal.diff().abs().fillna(0)
            else:
                simple_jm_df = pd.DataFrame()

        lambda_history = cache_data.get('lambda_history', [])
        lambda_dates = cache_data.get('lambda_dates', [])
        backtest_duration = cache_data.get('backtest_duration', None)
        best_ewma_hl = cache_data.get('best_ewma_hl', 8)
        cache_loaded = True

        if backtest_duration is not None:
            config_label = cache_data.get('config_name', '')
            if backtest_duration >= 3600:
                h = int(backtest_duration // 3600)
                m = int((backtest_duration % 3600) // 60)
                s = int(backtest_duration % 60)
                duration_str = f"{h}h {m}m {s}s"
            elif backtest_duration >= 60:
                m = int(backtest_duration // 60)
                s = int(backtest_duration % 60)
                duration_str = f"{m}m {s}s"
            else:
                duration_str = f"{backtest_duration:.2f} seconds"
            label = f"Last run: {duration_str}"
            if config_label:
                label += f" ({config_label})"
            duration_placeholder.info(label)
    except Exception as e:
        st.error(f"Could not load cache: {e}")

if not cache_loaded:
    st.info("No cached results found. Please configure parameters and click 'Run Backtest'.")
    st.stop()

if jm_xgb_df.empty:
    st.error("No results generated. Check your date ranges and data availability.")
    st.stop()

st.subheader("Analysis Period")

min_date = jm_xgb_df.index.min().date()
max_date = jm_xgb_df.index.max().date()

if 'period_slider' not in st.session_state:
    st.session_state.period_slider = (min_date, max_date)

def set_period(months):
    if months is None:
        st.session_state.period_slider = (min_date, max_date)
    else:
        new_start = (pd.to_datetime(max_date) - pd.DateOffset(months=months)).date()
        new_start = max(min_date, new_start)
        st.session_state.period_slider = (new_start, max_date)

# Quick selection buttons
if hasattr(st, 'segmented_control'):
    def update_from_segment():
        sel = st.session_state.period_selector
        if sel == "1 Year": set_period(12)
        elif sel == "3 Years": set_period(36)
        elif sel == "5 Years": set_period(60)
        elif sel == "10 Years": set_period(120)
        elif sel == "Max": set_period(None)
        
    st.segmented_control(
        "Period Selection", 
        ["1 Year", "3 Years", "5 Years", "10 Years", "Max"],
        key="period_selector", 
        on_change=update_from_segment,
        selection_mode="single",
        label_visibility="collapsed"
    )
elif hasattr(st, 'pills'):
    def update_from_pills():
        sel = st.session_state.period_selector
        if sel == "1 Year": set_period(12)
        elif sel == "3 Years": set_period(36)
        elif sel == "5 Years": set_period(60)
        elif sel == "10 Years": set_period(120)
        elif sel == "Max": set_period(None)
        
    st.pills(
        "Period Selection", 
        ["1 Year", "3 Years", "5 Years", "10 Years", "Max"],
        key="period_selector", 
        on_change=update_from_pills,
        selection_mode="single",
        label_visibility="collapsed"
    )
else:
    col1, col2, col3, col4, col5, _ = st.columns([1, 1, 1, 1, 1, 5])
    with col1:
        st.button("1 Year", on_click=set_period, args=(12,), width='stretch')
    with col2:
        st.button("3 Years", on_click=set_period, args=(36,), width='stretch')
    with col3:
        st.button("5 Years", on_click=set_period, args=(60,), width='stretch')
    with col4:
        st.button("10 Years", on_click=set_period, args=(120,), width='stretch')
    with col5:
        st.button("Max", on_click=set_period, args=(None,), width='stretch')

selected_dates = st.slider(
    "Select Date Range",
    min_value=min_date,
    max_value=max_date,
    key='period_slider',
    format="YYYY-MM-DD"
)

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    filter_start, filter_end = selected_dates
else:
    st.warning("Please select both a start and end date from the slider.")
    st.stop()

if filter_start >= filter_end:
    st.warning("Start date must be before end date.")
    st.stop()

jm_xgb_df = jm_xgb_df.loc[str(filter_start):str(filter_end)]
if jm_xgb_df.empty:
    st.warning("No data in the selected period. Please expand your date range.")
    st.stop()
    
if run_simple_jm_cached:
    simple_jm_df = simple_jm_df.loc[str(filter_start):str(filter_end)]

bh_returns = jm_xgb_df['Target_Return']
rf_returns = jm_xgb_df['RF_Rate']

strategies = {
    'Buy & Hold': bh_returns,
    'JM-XGB Strategy': jm_xgb_df['Strat_Return']
}
if run_simple_jm_cached:
    strategies['Simple JM Baseline'] = simple_jm_df['Strat_Return']

strategy_colors = {
    'Buy & Hold': 'dodgerblue',
    'JM-XGB Strategy': 'darkorange',
    'Simple JM Baseline': 'forestgreen'
}

columns = ['Strategy', 'Ann. Ret', 'Ann. Vol', 'Sharpe', 'Sortino', 'Max DD', 'Total Trades']
cell_text = []

metrics_data = [] # For Streamlit dataframe

for name, returns in strategies.items():
    ret, vol, sharpe, sortino, mdd = backend.calculate_metrics(returns, rf_returns)
    if name == 'Buy & Hold':
        trades = "N/A"
    elif "XGB" in name:
        trades = int(jm_xgb_df['Trades'].sum())
    else:
        trades = int(simple_jm_df['Trades'].sum())
        
    # Calculate Drawdown for the current period
    cum_wealth = (1 + returns).cumprod()
    peak = cum_wealth.cummax()
    drawdowns = (cum_wealth - peak) / peak
    
    cell_text.append([
        name,
        f"{ret*100:.2f}%", 
        f"{vol*100:.2f}%", 
        f"{sharpe:.2f}", 
        f"{sortino:.2f}",
        f"{mdd*100:.2f}%", 
        str(trades)
    ])
    
    metrics_data.append({
        'Strategy': name,
        'Ann. Ret': f"{ret*100:.2f}%",
        'Ann. Vol': f"{vol*100:.2f}%",
        'Sharpe': f"{sharpe:.2f}",
        'Sortino': f"{sortino:.2f}",
        'Max DD': f"{mdd*100:.2f}%",
        'Total Trades': str(trades)
    })

st.subheader("Performance Metrics")
st.dataframe(pd.DataFrame(metrics_data))

# Export UI logic has been moved to the end of the file to capture generated charts.

# We no longer use Matplotlib for the main display, we will just rely entirely on Plotly 
# for a much better interactive experience with these dense charts.

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# We need the benchmark wealth for the Probability chart
bh_wealth = (1 + bh_returns).cumprod()
jm_xgb_wealth = (1 + jm_xgb_df['Strat_Return']).cumprod()

# Overlay Actual Bear Regimes as background color (from thresholded state)
bear_series = jm_xgb_df['Forecast_State'] == 1
starts = bear_series.index[(bear_series) & (~bear_series.shift(1).fillna(False))]
ends = bear_series.index[(bear_series) & (~bear_series.shift(-1).fillna(False))]

def create_base_fig(title, y_title, height=500):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        yaxis_title=y_title,
        height=height,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="top", y=-0.15, x=0.5, xanchor="center", bgcolor="rgba(0,0,0,0)")
    )
    return fig

def apply_bear_shading(fig):
    for s, e in zip(starts, ends):
        fig.add_vrect(x0=s, x1=e, fillcolor="red", opacity=0.1, line_width=0, layer="below")

# --------------------------------------------------------------------------
# Performance Dashboard Layout
# --------------------------------------------------------------------------
st.subheader(
    "1. Cumulative Wealth",
    help="**Calculated:** Compounding growth of daily returns `(1 + returns).cumprod()`.\n\n**Data:** Strategy returns (net of transaction costs) vs. benchmarks.\n\n**Interpret:** Shows the total value of investment over time. A curve that stays above the benchmark indicates outperformance."
)
fig_wealth = create_base_fig("Cumulative Wealth Multiplier", "Wealth Multiplier")

for name, returns in strategies.items():
    if name == 'Simple JM Baseline' and not run_simple_jm_cached:
        continue
    wealth = (1 + returns).cumprod()
    final_val = wealth.iloc[-1]
    
    fig_wealth.add_trace(go.Scatter(
        x=wealth.index, y=wealth, mode='lines', 
        name=f"{name} Wealth",
        line=dict(color=strategy_colors.get(name), width=2 if 'JM' in name else 1.5)
    ))
    
    # Add ending wealth annotation
    fig_wealth.add_annotation(
        x=wealth.index[-1], y=final_val,
        text=f" {final_val:.2f}x",
        showarrow=False,
        xanchor="left",
        xshift=5,
        font=dict(color=strategy_colors.get(name), size=12),
        bgcolor="rgba(0,0,0,0.3)"
    )

apply_bear_shading(fig_wealth)
st.plotly_chart(fig_wealth, width='stretch')

# --------------------------------------------------------------------------
# Chart 2: Drawdown Profile
# --------------------------------------------------------------------------
st.subheader(
    "2. Drawdown Profile",
    help="**Calculated:** Percentage drop from the highest historical peak `(wealth - peak) / peak`.\n\n**Data:** Cumulative wealth trajectories of the strategy vs. benchmarks.\n\n**Interpret:** Visualizes downside risk. The area closer to 0% is better. Shallower drawdowns during red regions (bear markets) show the strategy successfully preserved capital."
)
fig_dd = create_base_fig("Portfolio Drawdown Percentage", "Drawdown")
fig_dd.update_yaxes(tickformat=".1%")

# Sort strategies to ensure JM-XGB is plotted last (rendered on top)
for name, returns in sorted(strategies.items(), key=lambda x: 1 if x[0] == 'JM-XGB Strategy' else 0):
    if name == 'Simple JM Baseline' and not run_simple_jm_cached:
        continue
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak
    
    is_xgb = name == 'JM-XGB Strategy'
    fig_dd.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown, mode='lines', 
        name=f"{name} DD",
        line=dict(color=strategy_colors.get(name), width=2.5 if is_xgb else 1.5),
        fill='tozeroy' if not is_xgb else None,
        fillcolor=strategy_colors.get(name) if not is_xgb else None,
        opacity=1.0 if is_xgb else 0.3
    ))

apply_bear_shading(fig_dd)
st.plotly_chart(fig_dd, width='stretch')

# --------------------------------------------------------------------------
# Chart 3: Rolling 1-Year Sortino Ratio
# --------------------------------------------------------------------------
st.subheader(
    "3. Rolling 1-Year Sortino Ratio",
    help="**Calculated:** Rolling 252-day annualized excess return over the risk-free rate, divided by downside deviation (standard deviation of only negative returns).\n\n**Data:** Daily strategy returns, risk-free rate, and benchmark returns.\n\n**Interpret:** Evaluates risk-adjusted performance while only penalizing downside volatility. Essential for regime-switching models designed to cut losses but let winners run. Higher values are better."
)
fig_sortino = create_base_fig("Rolling 12-Month Sortino Ratio", "Sortino Ratio")
ROLLING_WINDOW = 252 # 1 Trading Year

for name, returns in strategies.items():
    if name == 'Simple JM Baseline' and not run_simple_jm_cached:
        continue
    excess_ret = returns - rf_returns
    rolling_ann_ret = excess_ret.rolling(window=ROLLING_WINDOW).sum()
    
    # Calculate trailing downside deviation
    # Create a boolean mask for negative excess returns, calculate rolling std on that
    downside_returns = excess_ret.where(excess_ret < 0, 0)
    rolling_downside_vol = downside_returns.rolling(window=ROLLING_WINDOW).std() * np.sqrt(252)
    
    rolling_sortino = rolling_ann_ret / rolling_downside_vol.replace(0, np.nan)
    
    fig_sortino.add_trace(go.Scatter(
        x=rolling_sortino.index, y=rolling_sortino, mode='lines', 
        name=f"{name} (1Y Sortino)",
        line=dict(color=strategy_colors.get(name), width=1.5)
    ))

fig_sortino.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
apply_bear_shading(fig_sortino)
st.plotly_chart(fig_sortino, width='stretch')

# --------------------------------------------------------------------------
# Chart 4: Monthly Returns Heatmap
# --------------------------------------------------------------------------
st.subheader(
    "4. Monthly Returns Heatmap (JM-XGB Strategy)",
    help="**Calculated:** Compounded daily returns for each calendar month.\n\n**Data:** Daily strategy returns.\n\n**Interpret:** Shows seasonality and consistency. Green months are profitable. Look for a balance of consistency (lots of green) and magnitude (deep green vs shallow red)."
)

# Calculate monthly returns for the main strategy
strat_returns_daily = strategies['JM-XGB Strategy']
# Resample to monthly and calculate geometric return
monthly_returns = strat_returns_daily.resample('ME').apply(lambda x: (1 + x).prod() - 1)

# Create a DataFrame for the heatmap
heatmap_df = pd.DataFrame({'Return': monthly_returns})
heatmap_df['Year'] = heatmap_df.index.year
heatmap_df['Month'] = heatmap_df.index.strftime('%b')

# Pivot the data
heatmap_pivot = heatmap_df.pivot(index='Year', columns='Month', values='Return')

# Reorder columns to standard calendar order
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# Only include months that exist in the data
valid_months = [m for m in month_order if m in heatmap_pivot.columns]
heatmap_pivot = heatmap_pivot[valid_months]

# Calculate Annual Returns for row totals
heatmap_pivot['YTD'] = strat_returns_daily.resample('YE').apply(lambda x: (1 + x).prod() - 1).values

# Format as percentages
heatmap_text = heatmap_pivot.map(lambda x: f"{x:.1%}" if pd.notnull(x) else "")

# Create the heatmap figure
fig_heatmap = go.Figure(data=go.Heatmap(
    z=heatmap_pivot.values,
    x=heatmap_pivot.columns,
    y=heatmap_pivot.index,
    text=heatmap_text.values,
    texttemplate="%{text}",
    textfont={"size": 10},
    colorscale=["red", "black", "green"],
    zmid=0,  # Center the color scale at 0%
    showscale=False,
    xgap=2,
    ygap=2
))

fig_heatmap.update_layout(
    height=min(400, 50 + 30 * len(heatmap_pivot)), # Dynamic height based on years
    template="plotly_dark",
    margin=dict(l=20, r=20, t=30, b=20),
    yaxis=dict(autorange="reversed", type='category') # Reverse year order so most recent is top
)
st.plotly_chart(fig_heatmap, width='stretch')

col_dist, col_scatter = st.columns(2)

with col_dist:
    # --------------------------------------------------------------------------
    # Chart 5: Return Distribution Histogram
    # --------------------------------------------------------------------------
    st.subheader(
        "5. Return Distribution",
        help="**Calculated:** Histogram of daily return frequencies.\n\n**Data:** Daily returns of the strategy vs. Buy & Hold benchmark.\n\n**Interpret:** A good strategy chops off the 'left tail' (fewer large negative days) while preserving the 'right tail' (large positive days)."
    )
    
    fig_dist = go.Figure()
    
    all_rets = pd.concat([strategies['Buy & Hold'], strategies['JM-XGB Strategy']]).dropna()
    if not all_rets.empty:
        p_low = all_rets.quantile(0.005)
        p_high = all_rets.quantile(0.995)
        max_abs = max(abs(p_low), abs(p_high), 0.02)
        x_range = [-max_abs, max_abs]
        b_size = max_abs / 40.0
        x_bins = dict(start=-max_abs, end=max_abs, size=b_size)
    else:
        x_range = [-0.05, 0.05]
        x_bins = None
    
    # Plot Benchmark first (in background)
    if x_bins is not None:
        import numpy as np
        bins_array = np.arange(x_bins['start'], x_bins['end'] + x_bins['size'], x_bins['size'])
    else:
        bins_array = 100

    def get_hist_lines(data, bins):
        counts, bin_edges = np.histogram(data.dropna(), bins=bins)
        probs = counts / counts.sum()
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return centers, probs

    centers_bh, probs_bh = get_hist_lines(strategies['Buy & Hold'], bins_array)
    fig_dist.add_trace(go.Scatter(
        x=centers_bh, y=probs_bh,
        mode='lines',
        name='Buy & Hold',
        line=dict(color='gray', width=2, shape='spline'),
        fill='tozeroy',
        fillcolor='rgba(128, 128, 128, 0.4)'
    ))

    centers_st, probs_st = get_hist_lines(strategies['JM-XGB Strategy'], bins_array)
    fig_dist.add_trace(go.Scatter(
        x=centers_st, y=probs_st,
        mode='lines',
        name='JM-XGB Strategy',
        line=dict(color='darkorange', width=2, shape='spline'),
        fill='tozeroy',
        fillcolor='rgba(255, 140, 0, 0.5)'
    ))

    fig_dist.update_layout(
        barmode='overlay',
        template="plotly_dark",
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Daily Return",
        yaxis_title="Probability",
        xaxis=dict(tickformat=".1%", range=x_range),
        legend=dict(orientation="h", yanchor="top", y=-0.15, x=0.5, xanchor="center", bgcolor="rgba(0,0,0,0)")
    )
    # Add a vertical line at 0
    fig_dist.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
    st.plotly_chart(fig_dist, width='stretch')
    
with col_scatter:
    # --------------------------------------------------------------------------
    # Chart 6: Risk / Return Scatter
    # --------------------------------------------------------------------------
    st.subheader(
        "6. Risk / Return Profile",
        help="**Calculated:** Annualized Return vs. Annualized Volatility over the entire backtest period.\n\n**Data:** Strategy metrics table.\n\n**Interpret:** The 'Holy Grail' is the top-left quadrant (High Return, Low Risk). Strategies further up and to the left are mathematically superior."
    )
    
    fig_scatter = go.Figure()
    
    for m in metrics_data:
        name = m['Strategy']
        # Parse percentage strings back to floats
        ret = float(m['Ann. Ret'].strip('%')) / 100
        vol = float(m['Ann. Vol'].strip('%')) / 100
        
        fig_scatter.add_trace(go.Scatter(
            x=[vol], y=[ret],
            mode='markers+text',
            name=name,
            marker=dict(
                size=15, 
                color=strategy_colors.get(name, 'gray'),
                line=dict(width=2, color='white')
            ),
            text=[name.replace(' Strategy', '').replace(' Baseline', '')],
            textposition="top center"
        ))
        
    fig_scatter.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Annualized Volatility (Risk)",
        yaxis_title="Annualized Return",
        xaxis=dict(tickformat=".1%", rangemode="tozero"),
        yaxis=dict(tickformat=".1%"),
        showlegend=False
    )
    # Add risk-free rate assumption line (horizontal)
    rf_mean = rf_returns.mean() * 252
    fig_scatter.add_hline(y=rf_mean, line_dash="dot", line_color="gray", annotation_text=f"Risk-Free ({rf_mean:.1%})", annotation_position="bottom right")
    
    st.plotly_chart(fig_scatter, width='stretch')

