# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quantitative trading strategy combining a Statistical Jump Model (JM) for market regime identification with XGBoost for return forecasting. Based on arXiv paper 2406.09578v2 (Shu, Yu, Mulvey, 2024). Walk-forward backtested from 2007-2026 on S&P 500 Total Return data.

**Goal:** A deployable trading strategy with no look-ahead bias and minimal overfitting, staying as close as possible to the research paper.

## Commands

```bash
# Activate the virtual environment first
source .venv/bin/activate

# Run the full backtest (generates timestamped PDF report)
python main.py

# Run experiments (generates timestamped MD report in benchmarks/)
python run_experiments.py              # Run all 9 experiments
python run_experiments.py 1,5,6,9      # Run specific experiments
python run_experiments.py 1            # Run single experiment
python run_experiments.py 2-5          # Run a range
python run_experiments.py list         # List available experiments
python run_experiments.py --help       # Show usage

# Launch interactive Streamlit dashboard
streamlit run app.py

# Multi-asset benchmark (12 ETFs, parallel execution)
python misc_scripts/benchmark_assets.py
```

There is no formal test framework (pytest/unittest). Tests are standalone scripts in `misc_scripts/`.

## Architecture

### Entry Points
- `main.py` (~815 lines) -- Core backtest engine. Fetches data, fits models, runs walk-forward simulation, outputs PDF report.
- `app.py` (~2120 lines) -- Streamlit dashboard. Imports `main.py` as `backend`. Sidebar has an experiment preset selector that auto-fills all StrategyConfig parameters (tuning metric, validation window type, lambda smoothing, threshold, allocation style, ensemble K). Uses `walk_forward_backtest()` for execution. Interactive Plotly charts.
- `run_experiments.py` (~300 lines) -- Experiment runner. Tests strategy variants via `StrategyConfig`, compares vs B&H, generates timestamped MD reports with sub-period analysis and lambda stability tracking.
- `misc_scripts/benchmark_assets.py` (~795 lines) -- Multi-asset benchmark. Tests 12 ETFs across 5 market periods with parallel execution.

### Supporting Files
- `config.py` -- `StrategyConfig` dataclass with all tunable strategy parameters.
- `benchmarks/` -- Timestamped experiment reports (MD) and benchmark results (CSV).
- `cache/` -- Data caches (`data_cache.pkl`, per-ticker caches for multi-asset).
- `paper_text.txt` -- Full text of the reference paper for comparison.

### Algorithm Pipeline (in `main.py`)
1. `fetch_and_prepare_data()` -- Downloads from Yahoo Finance + FRED, engineers features (EWMA-based return/macro features), caches to `cache/data_cache.pkl`
2. `StatisticalJumpModel` class -- 2-state regime model using alternating optimization (K-means + Viterbi). Optimized fast-path for 2 states in the Viterbi forward pass.
3. `run_period_forecast()` -- For a given date: fits JM on 11-year lookback, trains XGBClassifier on regime labels + macro features, predicts 6-month OOS window. Computes SHAP values. Results cached in `_forecast_cache` dict.
4. `simulate_strategy()` -- Chains 6-month forecast periods. Applies EWMA smoothing to raw probabilities, thresholds at 0.5, shifts signals by 1 day (look-ahead bias prevention), applies transaction costs.
5. `walk_forward_backtest(df, config)` -- Walk-forward loop: every 6 months tunes lambda on validation window (maximize Sharpe), then runs OOS chunk with best lambda. Returns DataFrame with `.attrs` metadata (lambda_history, lambda_dates, ewma_halflife).

### Key Design Decisions
- State 0 = Bullish (invest in target asset), State 1 = Bearish (rotate to risk-free). States are aligned after fitting so State 0 always has higher cumulative excess return.
- Forecast signal is shifted +1 day before applying to returns to prevent look-ahead bias.
- Binary 0/1 allocation is the strategy's core strength. Experiments proved continuous allocation and higher thresholds destroy performance.
- `app.py` sidebar has an experiment preset selector (all 9 experiments + Custom). Selecting a preset auto-fills all StrategyConfig params. The backtest uses `walk_forward_backtest()` from main.py, ensuring all config options take effect.
- `app.py` mutates `backend.LAMBDA_GRID` and other module-level constants directly to pass data/grid configuration from sidebar controls.

## Configuration

### Module-Level Constants (`main.py` lines 45-70)
- `TARGET_TICKER = '^SP500TR'`, `BOND_TICKER = 'VBMFX'`, `RISK_FREE_TICKER = '^IRX'`, `VIX_TICKER = '^VIX'`
- `START_DATE_DATA = '1987-01-01'` -- Data start (accommodates 11-year lookback)
- `OOS_START_DATE = '2007-01-01'` -- Out-of-sample start (paper: 2007-2023)
- `TRANSACTION_COST = 0.0005` -- 5 basis points
- `LAMBDA_GRID = [0.0] + list(np.logspace(0, 2, 10))` -- 11 candidates (0 + log-spaced 1 to 100)
- `EWMA_HL_GRID = [0, 2, 4, 8]` -- EWMA halflife candidates for probability smoothing

### StrategyConfig (`config.py`)
Dataclass controlling experiment variants:
- `tuning_metric`: "sharpe" (default/paper) or "sortino"
- `lambda_smoothing`: bool -- EWMA smooth lambda selection across periods (recommended: True)
- `lambda_ensemble_k`: int -- if > 1, average top-K lambda forecasts
- `validation_window_type`: "rolling" (default/paper, 5yr) or "expanding"
- `prob_threshold`: float (default 0.50, paper: 0.50)
- `allocation_style`: "binary" (default/paper) or "continuous"
- `xgb_params`: dict with XGBoost hyperparameters (default: XGBoost defaults per paper -- max_depth=6, learning_rate=0.3, no regularization)

## Experiment Framework

### Available Experiments (in `run_experiments.py`)
1. Paper Baseline (reference)
2. Sortino Tuned
3. Conservative Threshold (0.6)
4. Continuous Allocation
5. Lambda Smoothing
6. Expanding Window
7. Lambda Ensemble (Top 3)
8. The Ultimate Combo
9. Expanding + Lambda Smoothing

### Report Output
Each run generates `benchmarks/experiment_report_YYYYMMDD_HHMMSS.md` containing:
- Results vs B&H table with Sharpe/Sortino/MDD deltas and WIN/LOSE verdicts
- Sub-period analysis (GFC, Recovery, Late Cycle, COVID, Post-COVID)
- Lambda stability tracking per experiment (mean, std, CV, timeline)
- Experiment configurations
- Future enhancements backlog with priority/risk ratings

### Key Findings (as of 2026-03-03)
- **Paper Baseline now beats B&H** -- Sharpe ~0.57 vs B&H ~0.54 after switching to default XGB params (Session 2 fix)
- **Binary 0/1 signal is essential** -- experiments #3, #4, #8 proved continuous/threshold changes destroy performance
- **Strategy wins in crises, loses in bull markets** -- GFC: strong outperformance; Recovery/COVID: underperformance
- **Lambda stability improved with default XGB** -- CV reduced from 1.24 to ~1.07
- **Time period (2007-2026 vs 2007-2023) has negligible impact** on relative performance (-0.003 Sharpe delta)
- See `benchmarks/experiment_selection_20260303.md` for earlier analysis with regularized XGB

## Feature Set

Two groups fed to XGBoost (defined in `run_period_forecast()`):
- **Return features** (used by both JM and XGB): `DD_log_5/21`, `Avg_Ret_5/10/21`, `Sortino_5/10/21`
- **Macro features** (XGB only): `Yield_2Y_EWMA_diff`, `Yield_Slope_EWMA_10`, `Yield_Slope_EWMA_diff_21`, `VIX_EWMA_log_diff`, `Stock_Bond_Corr`

Return features are standardized (z-score) before feeding to the Jump Model. XGBoost receives raw feature values.

## Known Issues / Paper Gaps

- **XGBoost hyperparameters:** Now using default XGB params (max_depth=6, learning_rate=0.3, no regularization) as the paper specifies. Previous custom regularized params (max_depth=4, reg_alpha=1.0, reg_lambda=5.0) caused -0.174 Sharpe delta and made the strategy LOSE to B&H. Switching to defaults fixed this (Session 2, 2026-03-03).
- **Remaining Sharpe gap:** Paper reports 0.79 for LargeCap (2007-2023); our paper-comparable config gets ~0.56. Gap likely from data source (Yahoo vs Bloomberg).
- **Data source:** Paper uses Bloomberg total return indices; we use Yahoo Finance. May affect feature quality.
- **OOS period:** We test 2007-2026 vs paper's 2007-2023. Diagnostic showed time period effect is negligible (-0.003 Sharpe delta).

## Caching

- `cache/data_cache.pkl` -- Persisted fetched+engineered data (delete to re-fetch from APIs)
- `cache/data_cache_{ticker}.pkl` -- Per-ticker caches for multi-asset benchmark
- `_forecast_cache` -- In-memory dict keyed by `(date, lambda, include_xgboost, constrain_xgb)`, lives only during script execution
- `cache/backtest_cache.pkl` -- Used by `app.py` for dashboard session persistence

## Session Memory & Experiment Tracking

Files in `.claude/`:
- **MEMORY.md** - Quick reference (auto-loaded each session, keep <200 lines)
- **experiments.md** - Chronological log of experiments, findings, and changes (newest first)
- **performance_gaps.md** - Gap analysis vs paper and improvement priorities

**Instructions for Claude:** When running experiments or making improvements:
1. Read these files at session start to understand prior context
2. Update them after each significant finding or change (experiments completed, root causes found, improvements applied)
3. Keep MEMORY.md as a concise index; use other files for details
4. Ask user before major updates if unsure

## Python Compatibility

The top of `main.py` includes a distutils.version compatibility shim for Python 3.12+ (required by pandas-datareader). This must remain at the top of the file before other imports.
