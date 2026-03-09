import streamlit as st
import subprocess
import glob
import os
import json
import datetime
import numpy as np
import main as backend

st.set_page_config(page_title="Diagnostics Launcher", page_icon="🛠️", layout="wide")

st.title("🛠️ Diagnostics Launcher")
st.markdown("Run background diagnostic scripts and generate comprehensive LLM-ready markdown reports directly from the UI.")

# =============================================================================
# Session state defaults (same as Performance Tracker, ensures keys exist)
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
st.sidebar.header("Parameters")
st.sidebar.caption("Shared with Performance Tracker. Changes here are reflected there and vice versa.")

with st.sidebar.expander("Data Source", expanded=True):
    st.text_input("Target Ticker", key='target_ticker_input')
    st.text_input("Bond Ticker", key='bond_ticker_input')
    st.text_input("Risk-Free Ticker", key='rf_ticker_input')
    st.text_input("VIX Ticker", key='vix_ticker_input')
    st.text_input("Data Start Date", key='start_date_input')
    st.text_input("OOS Start Date", key='oos_start_input')
    st.text_input("End Date", key='end_date_input')

with st.sidebar.expander("Jump Model", expanded=False):
    st.number_input("Validation Window (Years)", min_value=1, max_value=20, key='val_window_input')
    st.number_input("Transaction Cost", min_value=0.0, max_value=0.01, format="%.4f", key='transaction_cost_input')

    grid_preset = st.selectbox(
        "Lambda Grid Preset",
        ["JM-XGB Improved (10 points)", "Default (Fast: 4 points)", "Expanded (Paper: 21 points)", "Custom"],
        key='lambda_grid_preset'
    )
    if grid_preset == "JM-XGB Improved (10 points)":
        lambda_grid = [0.0] + list(np.logspace(0, 2, 10))
        st.caption("0.0 + 10 log-spaced points up to 100.0")
    elif grid_preset == "Default (Fast: 4 points)":
        lambda_grid = [1.0, 10.0, 50.0, 100.0]
        st.caption("1.0, 10.0, 50.0, 100.0")
    elif grid_preset == "Expanded (Paper: 21 points)":
        lambda_grid = [0.0] + list(np.logspace(0, 2, 20))
        st.caption("0.0 + 20 log-spaced points up to 100.0")
    else:
        lambda_grid_str = st.text_input("Custom Grid (comma separated)", "1.0, 10.0, 50.0, 100.0")
        try:
            lambda_grid = [float(x.strip()) for x in lambda_grid_str.split(',')]
        except ValueError:
            st.error("Invalid Lambda Grid format. Using default.")
            lambda_grid = [1.0, 10.0, 50.0, 100.0]
    st.session_state['lambda_grid_value'] = lambda_grid

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

# Scripts that import from main.py / config.py and benefit from env var sync
SYNCED_SCRIPTS = {"diagnose_pipeline.py", "run_experiments.py"}

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
                result = subprocess.run(
                    ["python", script_to_run],
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
