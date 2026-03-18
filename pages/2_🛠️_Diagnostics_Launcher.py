import streamlit as st
import subprocess
import glob
import os
import json
import datetime
import numpy as np
import pandas as pd
import main as backend
from config import StrategyConfig

st.set_page_config(page_title="Diagnostics Launcher", page_icon="🛠️", layout="wide")

st.title("🛠️ Diagnostics Launcher")
st.markdown("Run background diagnostic scripts and generate comprehensive LLM-ready markdown reports directly from the UI.")

# =============================================================================
# Experiment Presets (keep in sync with Model Analysis page)
# =============================================================================
_yesterday = (pd.Timestamp.today() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

EXPERIMENT_PRESETS = {
    "1. Paper Baseline": {
        "config": StrategyConfig(name="1. Paper Baseline", ewma_mode="paper"),
        "oos_start": "2007-01-01",
        "end_date": "2023-12-31",
        "start_date": "1991-01-01",
        "val_window": 5,
        "lambda_grid_preset": "Dense Mid-Range (8 points)",
    },
    "2. Optimized": {
        "config": StrategyConfig(name="2. Optimized", lambda_subwindow_consensus=True),
        "oos_start": "2007-01-01",
        "end_date": _yesterday,
        "start_date": "1987-01-01",
        "val_window": 5,
        "lambda_grid_preset": "Focused No-100 (4 points)",
    },
    "3. Tradable": {
        "config": StrategyConfig(name="3. Tradable", lambda_selection="median_positive"),
        "oos_start": "2007-01-01",
        "end_date": _yesterday,
        "start_date": "1991-01-01",
        "val_window": 5,
        "lambda_grid_preset": "Expanded (21 points)",
    },
}
PRESET_NAMES = list(EXPERIMENT_PRESETS.keys()) + ["Custom"]

def on_preset_change():
    preset_name = st.session_state.get('diag_experiment_preset', 'Custom')
    if preset_name == "Custom":
        return
    preset = EXPERIMENT_PRESETS[preset_name]
    cfg = preset["config"]
    st.session_state.start_date_input = preset["start_date"]
    st.session_state.oos_start_input = preset["oos_start"]
    st.session_state.end_date_input = preset["end_date"]
    st.session_state.val_window_input = preset["val_window"]
    st.session_state.lambda_grid_preset = preset["lambda_grid_preset"]
    st.session_state.tuning_metric = cfg.tuning_metric
    st.session_state.validation_window_type = cfg.validation_window_type
    st.session_state.lambda_smoothing = cfg.lambda_smoothing
    st.session_state.prob_threshold = cfg.prob_threshold
    st.session_state.allocation_style = cfg.allocation_style
    st.session_state.lambda_ensemble_k = cfg.lambda_ensemble_k
    st.session_state.lambda_selection = cfg.lambda_selection
    st.session_state.lambda_subwindow_consensus = cfg.lambda_subwindow_consensus
    st.session_state.ewma_mode = cfg.ewma_mode
    st.session_state.execution_mode = cfg.execution_mode == 'next_open'
    st.session_state.xgb_max_depth = 6
    st.session_state.xgb_n_estimators = 100
    st.session_state.xgb_learning_rate = 0.3
    st.session_state.xgb_subsample = 1.0
    st.session_state.xgb_colsample_bytree = 1.0
    st.session_state.xgb_reg_alpha = 0.0
    st.session_state.xgb_reg_lambda = 1.0

# =============================================================================
# Session state defaults
# =============================================================================
_default_preset = EXPERIMENT_PRESETS["2. Optimized"]
_default_cfg = _default_preset["config"]
_defaults = {
    'start_date_input': _default_preset["start_date"],
    'oos_start_input': _default_preset["oos_start"],
    'end_date_input': _default_preset["end_date"],
    'target_ticker_input': backend.TARGET_TICKER,
    'bond_ticker_input': backend.BOND_TICKER,
    'rf_ticker_input': backend.RISK_FREE_TICKER,
    'vix_ticker_input': backend.VIX_TICKER,
    'transaction_cost_input': float(backend.TRANSACTION_COST),
    'val_window_input': _default_preset["val_window"],
    'lambda_grid_preset': _default_preset["lambda_grid_preset"],
    'tuning_metric': _default_cfg.tuning_metric,
    'validation_window_type': _default_cfg.validation_window_type,
    'lambda_smoothing': _default_cfg.lambda_smoothing,
    'prob_threshold': _default_cfg.prob_threshold,
    'allocation_style': _default_cfg.allocation_style,
    'lambda_ensemble_k': _default_cfg.lambda_ensemble_k,
    'lambda_selection': _default_cfg.lambda_selection,
    'lambda_subwindow_consensus': _default_cfg.lambda_subwindow_consensus,
    'ewma_mode': _default_cfg.ewma_mode,
    'execution_mode': _default_cfg.execution_mode == 'next_open',
    'xgb_max_depth': 6,
    'xgb_n_estimators': 100,
    'xgb_learning_rate': 0.3,
    'xgb_subsample': 1.0,
    'xgb_colsample_bytree': 1.0,
    'xgb_reg_alpha': 0.0,
    'xgb_reg_lambda': 1.0,
}
for key, val in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# =============================================================================
# Sidebar: editable parameters (shared keys with Performance Tracker)
# =============================================================================
st.sidebar.header("Experiment Preset")
st.sidebar.selectbox(
    "Strategy", PRESET_NAMES, index=1,
    key='diag_experiment_preset', on_change=on_preset_change,
    help="Select a predefined experiment configuration. All parameters below will be filled automatically."
)

st.sidebar.header("Parameters")
st.sidebar.caption("Shared with Model Analysis page. Changes here are reflected there and vice versa.")

with st.sidebar.expander("Data Source", expanded=True):
    st.text_input("Target Ticker", key='target_ticker_input')
    st.text_input("Bond Ticker", key='bond_ticker_input')
    st.text_input("Risk-Free Ticker", key='rf_ticker_input')
    st.text_input("VIX Ticker", key='vix_ticker_input')
    st.text_input("Data Start Date", key='start_date_input')
    st.text_input("OOS Start Date", key='oos_start_input')
    st.text_input("End Date", key='end_date_input')

with st.sidebar.expander("Jump Model", expanded=False):
    st.selectbox("Validation Window Type", ["rolling", "expanding"], key='validation_window_type')
    st.number_input("Validation Window (Years)", min_value=1, max_value=20, key='val_window_input')
    st.selectbox("Tuning Metric", ["sharpe", "sortino"], key='tuning_metric')
    st.checkbox("Lambda Smoothing", key='lambda_smoothing')
    st.number_input("Probability Threshold", min_value=0.30, max_value=0.70, step=0.05, format="%.2f", key='prob_threshold')
    st.selectbox("Allocation Style", ["binary", "continuous"], key='allocation_style')
    st.number_input("Lambda Ensemble K", min_value=1, max_value=5, key='lambda_ensemble_k')
    st.selectbox("Lambda Selection", ["best", "median_positive"], key='lambda_selection',
                 help="'best' = argmax(validation Sharpe). 'median_positive' = median of all lambdas with positive validation Sharpe.")
    st.checkbox("Sub-Window Consensus", key='lambda_subwindow_consensus',
                help="Split validation into 3 overlapping sub-windows, find best lambda in each, take median.")
    st.checkbox("Realistic Next-Open Execution", key='execution_mode',
                help="If checked, trades execute at the next day's open instead of assumed exact close.")
    st.number_input("Transaction Cost", min_value=0.0, max_value=0.01, format="%.4f", key='transaction_cost_input')

    grid_preset = st.selectbox(
        "Lambda Grid Preset",
        ["Dense Mid-Range (8 points)", "Focused Mid-Range (5 points)", "Focused No-100 (4 points)", "Legacy Wide (11 points)", "Expanded (21 points)", "Custom"],
        key='lambda_grid_preset'
    )
    if grid_preset == "Dense Mid-Range (8 points)":
        lambda_grid = [4.64, 10.0, 15.0, 21.54, 30.0, 46.42, 70.0, 100.0]
        st.caption("4.64, 10, 15, 21.54, 30, 46.42, 70, 100 (default)")
    elif grid_preset == "Focused Mid-Range (5 points)":
        lambda_grid = [4.64, 10.0, 21.54, 46.42, 100.0]
        st.caption("4.64, 10, 21.54, 46.42, 100")
    elif grid_preset == "Focused No-100 (4 points)":
        lambda_grid = [4.64, 10.0, 21.54, 46.42]
        st.caption("4.64, 10, 21.54, 46.42 (best single-asset)")
    elif grid_preset == "Legacy Wide (11 points)":
        lambda_grid = [0.0] + list(np.logspace(0, 2, 10))
        st.caption("0.0 + 10 log-spaced points up to 100.0")
    elif grid_preset == "Expanded (21 points)":
        lambda_grid = [0.0] + list(np.logspace(0, 2, 20))
        st.caption("0.0 + 20 log-spaced points up to 100.0")
    else:
        lambda_grid_str = st.text_input("Custom Grid (comma separated)", "4.64, 10.0, 15.0, 21.54, 30.0, 46.42, 70.0, 100.0")
        try:
            lambda_grid = [float(x.strip()) for x in lambda_grid_str.split(',')]
        except ValueError:
            st.error("Invalid Lambda Grid format. Using default.")
            lambda_grid = [4.64, 10.0, 15.0, 21.54, 30.0, 46.42, 70.0, 100.0]
    st.session_state['lambda_grid_value'] = lambda_grid

    st.selectbox("EWMA Halflife Mode", ["auto", "paper"], key='ewma_mode',
                help="'auto' = tune on pre-OOS validation window. 'paper' = use paper-prescribed values per asset.")

with st.sidebar.expander("XGBoost", expanded=False):
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


# =============================================================================
# Build environment variables for subprocess
# =============================================================================
def get_script_env():
    """Build env dict that syncs subprocess constants with sidebar parameters."""
    env = os.environ.copy()

    # Data source & pipeline constants (overrides in main.py)
    mapping = {
        'XGB_TARGET_TICKER': 'target_ticker_input',
        'XGB_BOND_TICKER': 'bond_ticker_input',
        'XGB_RISK_FREE_TICKER': 'rf_ticker_input',
        'XGB_VIX_TICKER': 'vix_ticker_input',
        'XGB_START_DATE_DATA': 'start_date_input',
        'XGB_OOS_START_DATE': 'oos_start_input',
        'XGB_END_DATE': 'end_date_input',
        'XGB_TRANSACTION_COST': 'transaction_cost_input',
        'XGB_VALIDATION_WINDOW_YRS': 'val_window_input',
    }
    for env_key, ss_key in mapping.items():
        env[env_key] = str(st.session_state[ss_key])

    # Lambda grid
    env['XGB_LAMBDA_GRID'] = json.dumps([float(x) for x in st.session_state['lambda_grid_value']])

    # Strategy parameters
    strategy_mapping = {
        'XGB_TUNING_METRIC': 'tuning_metric',
        'XGB_VALIDATION_WINDOW_TYPE': 'validation_window_type',
        'XGB_LAMBDA_SMOOTHING': 'lambda_smoothing',
        'XGB_PROB_THRESHOLD': 'prob_threshold',
        'XGB_ALLOCATION_STYLE': 'allocation_style',
        'XGB_LAMBDA_ENSEMBLE_K': 'lambda_ensemble_k',
        'XGB_LAMBDA_SELECTION': 'lambda_selection',
        'XGB_LAMBDA_SUBWINDOW_CONSENSUS': 'lambda_subwindow_consensus',
        'XGB_EWMA_MODE': 'ewma_mode',
    }
    for env_key, ss_key in strategy_mapping.items():
        env[env_key] = str(st.session_state[ss_key])
        
    env['XGB_EXECUTION_MODE'] = "next_open" if st.session_state.get('execution_mode', True) else "close"

    # XGBoost parameters (overrides in config.py)
    xgb_mapping = {
        'XGB_PARAM_MAX_DEPTH': 'xgb_max_depth',
        'XGB_PARAM_N_ESTIMATORS': 'xgb_n_estimators',
        'XGB_PARAM_LEARNING_RATE': 'xgb_learning_rate',
        'XGB_PARAM_REG_ALPHA': 'xgb_reg_alpha',
        'XGB_PARAM_REG_LAMBDA': 'xgb_reg_lambda',
        'XGB_PARAM_SUBSAMPLE': 'xgb_subsample',
        'XGB_PARAM_COLSAMPLE_BYTREE': 'xgb_colsample_bytree',
    }
    for env_key, ss_key in xgb_mapping.items():
        env[env_key] = str(st.session_state[ss_key])

    return env


# =============================================================================
# Script definitions
# =============================================================================
SCRIPTS = {
    "Diagnose Pipeline": {
        "file": "diagnose_pipeline.py",
        "description": "Verify the underlying structural health of the ML models.",
        "icon": "🩺"
    },
    "Run Experiments": {
        "file": "run_experiments.py",
        "description": "Test hypothesis-driven strategy modifications and generate comparison reports.",
        "icon": "🧪"
    },
    "Benchmark Assets": {
        "file": "misc_scripts/benchmark_assets.py",
        "description": "Verify the strategy works across other asset classes (optional, for robustness).",
        "icon": "🌍"
    }
}

# Scripts that benefit from env var sync with sidebar parameters
SYNCED_SCRIPTS = {"diagnose_pipeline.py", "run_experiments.py", "misc_scripts/benchmark_assets.py"}

# =============================================================================
def get_asset_lists():
    """Parse asset_lists.md to get available lists without importing."""
    lists = []
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    md_path = os.path.join(base_dir, 'misc_scripts', 'asset_lists.md')
    try:
        with open(md_path, 'r') as f:
            for line in f:
                if line.startswith('## '):
                    lists.append(line[3:].strip())
    except Exception:
        pass
    return lists

# =============================================================================
# Main layout
# =============================================================================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Select Script to Run")
    selected_script = st.radio("Available Scripts:", list(SCRIPTS.keys()), format_func=lambda x: f"{SCRIPTS[x]['icon']} {x}")

    script_info = SCRIPTS[selected_script]
    st.info(script_info["description"])

    if script_info["file"] not in SYNCED_SCRIPTS:
        st.caption("This script uses its own independent configuration.")

    if selected_script == "Benchmark Assets":
        asset_lists = get_asset_lists()
        selected_asset_list = st.selectbox("Select Asset List:", asset_lists)
        if selected_asset_list:
            st.session_state.selected_asset_list = selected_asset_list

    if st.button(f"Run {script_info['file']}", type="primary"):
        st.session_state.running_script = script_info['file']
        st.session_state.script_output = ""
        st.rerun()

with col2:
    st.subheader("Console Output")

    output_container = st.empty()

    if 'running_script' in st.session_state and st.session_state.running_script is not None:
        script_to_run = st.session_state.running_script

        # Use synced env for scripts that import from main.py
        script_env = get_script_env() if script_to_run in SYNCED_SCRIPTS else None

        with st.spinner(f"Running {script_to_run}..."):
            try:
                cmd = ["python", script_to_run]
                if script_to_run == "misc_scripts/benchmark_assets.py" and "selected_asset_list" in st.session_state:
                    cmd.append(st.session_state.selected_asset_list)

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    env=script_env
                )
                output = f"--- STDOUT ---\n{result.stdout}\n\n--- STDERR ---\n{result.stderr}"
                output_container.code(output, language="text")

                if result.returncode == 0:
                    st.success(f"{script_to_run} completed successfully!")
                else:
                    st.error(f"{script_to_run} failed with return code {result.returncode}.")

            except Exception as e:
                output_container.error(f"Failed to execute script: {e}")

        st.session_state.running_script = None

st.divider()

# =============================================================================
# Markdown Report Viewer
# =============================================================================
st.subheader("📄 Recently Generated Reports (LLM Ready)")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
benchmarks_dir = os.path.join(base_dir, "benchmarks")

if os.path.exists(benchmarks_dir):
    md_files = glob.glob(os.path.join(benchmarks_dir, "*.md"))
    md_files.sort(key=os.path.getmtime, reverse=True)

    if md_files:
        selected_file = st.selectbox(
            "Select a report to view/copy:",
            md_files,
            format_func=lambda x: f"{os.path.basename(x)} (Generated: {datetime.datetime.fromtimestamp(os.path.getmtime(x)).strftime('%Y-%m-%d %H:%M:%S')})"
        )

        with open(selected_file, 'r') as f:
            md_content = f.read()

        st.markdown(md_content)

        with st.expander("Raw Markdown (Click 'Copy' button in top right of box)"):
            st.code(md_content, language="markdown")
    else:
        st.info("No markdown reports found in the 'benchmarks' folder.")
else:
    st.info("The 'benchmarks' folder does not exist yet. Run a diagnostic script above to generate reports.")
