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

# Multi-asset benchmark (configurable asset lists, parallel execution)
python misc_scripts/benchmark_assets.py              # Default ETFs (12 ETFs)
python misc_scripts/benchmark_assets.py "Long History" # Long-history mutual fund proxies
python misc_scripts/benchmark_assets.py list           # Show available asset lists

# Run pipeline diagnostics
python misc_scripts/diagnose_pipeline.py              # Full diagnostics
python misc_scripts/diagnose_pipeline.py --quick      # Skip slow tests (permutation)
```

There is no formal test framework (pytest/unittest). Tests are standalone scripts in `misc_scripts/`.

## Architecture

### Entry Points
- `main.py` (~815 lines) -- Core backtest engine. Fetches data, fits models, runs walk-forward simulation, outputs PDF report.
- `app.py` & `pages/` -- Streamlit multi-page dashboard. `app.py` is the landing page.
  - `pages/1_🚀_Performance_Tracker.py`: Fast parameter tuning and strategy performance metrics via `st.form`. Skips heavy ML/SHAP operations.
  - `pages/2_📊_Model_Analysis.py`: Deep evaluation of the active model, SHAP values, feature charts and diagnostics.
  - `pages/3_🛠️_Diagnostics_Launcher.py`: Control center for background scripts and MD report viewer.
  - `pages/4_🔍_Data_Quality_Audit.py`: Go/no-go data quality checks. Reads from existing caches (no pipeline re-run). Three sections: Raw Data Health (Yahoo + FRED), Feature Health (z-score extremes, Sortino clipping, Stock-Bond Corr), Regime Labeling Health (bear fraction, label imbalance, lambda stability). Ticker selector at top; Section 3 uses last cached backtest.
  - Sidebar parameters use `StrategyConfig`. Uses `walk_forward_backtest()` for execution through `main.py` alias `backend`.
- `run_experiments.py` (~300 lines) -- Experiment runner. Tests strategy variants via `StrategyConfig`, compares vs B&H, generates timestamped MD reports with sub-period analysis and lambda stability tracking.
- `misc_scripts/benchmark_assets.py` (~880 lines) -- Multi-asset benchmark. Tests configurable asset lists across 5 market periods with parallel execution. Asset lists defined in `misc_scripts/asset_lists.md`.
- `misc_scripts/diagnose_pipeline.py` (~800 lines) -- Pipeline health diagnostics. Generates a minimalist MD report assessing ML model quality and regime structural integrity.

### Supporting Files
- `config.py` -- `StrategyConfig` dataclass with all tunable strategy parameters.
- `refcard.md` -- **Implementation reference card** extracted from the paper. Contains all formulas, hyperparameters, feature definitions, data splits, and numerical results. Use this instead of the full paper when verifying implementation correctness. Includes an "Undisclosed" section listing gaps the paper does not resolve.
- `misc_scripts/asset_lists.md` -- Named asset lists for multi-asset benchmark (tickers, asset classes, data_start dates). Parsed by `benchmark_assets.py`.
- `benchmarks/` -- Timestamped experiment reports (MD) and benchmark results (CSV).
- `cache/` -- Data caches (`data_cache.pkl`, per-ticker caches for multi-asset).
- `paper_text.txt` -- Full text of the reference paper for comparison.

### Algorithm Pipeline (in `main.py`)
1. `fetch_and_prepare_data()` -- Downloads from Yahoo Finance + FRED, engineers features (EWMA-based return/macro features), caches to `cache/data_cache.pkl`
2. `StatisticalJumpModel` class -- 2-state regime model using alternating optimization (K-means++ init + Viterbi, n_init=10, max_iter=1000, tol=1e-8). Matches the paper's `jumpmodels` library exactly (100% state agreement, identical objective values). `predict_online()` uses forward-only Viterbi (accumulated DP costs, no backtracking).
3. `run_period_forecast()` -- For a given date: fits JM on 11-year lookback, trains XGBClassifier on regime labels + macro features, predicts 6-month OOS window. Computes SHAP values. Results cached in `_forecast_cache` dict.
4. `simulate_strategy()` -- Chains 6-month forecast periods. Applies EWMA smoothing to raw probabilities, thresholds at 0.5, shifts signals by 1 day (look-ahead bias prevention), applies transaction costs.
5. `walk_forward_backtest(df, config)` -- Walk-forward loop: every 6 months tunes lambda on validation window (maximize Sharpe), then runs OOS chunk with best lambda. Returns DataFrame with `.attrs` metadata (lambda_history, lambda_dates, ewma_halflife).

### Key Design Decisions
- State 0 = Bullish (invest in target asset), State 1 = Bearish (rotate to risk-free). States are aligned after fitting so State 0 always has higher cumulative excess return.
- Forecast signal is shifted +1 day before applying to returns to prevent look-ahead bias.
- Binary 0/1 allocation is the strategy's core strength. Experiments proved continuous allocation and higher thresholds destroy performance.
- `1_🚀_Performance_Tracker.py` and `2_📊_Model_Analysis.py` sidebars both have an experiment preset selector (all 11 experiments + Custom). Selecting a preset auto-fills all StrategyConfig params. The backtest uses `walk_forward_backtest()` from main.py, ensuring all config options take effect.
- The dashboard mutates `backend.LAMBDA_GRID` and other module-level constants directly to pass data/grid configuration from sidebar controls.

### Dashboard Deployment Rule (MANDATORY)
**When any algorithm improvement is made (new StrategyConfig fields, new experiment presets, new walk-forward logic, etc.), it MUST be deployed to ALL Streamlit pages simultaneously.** Specifically:
1. Add new `StrategyConfig` fields to `config.py`.
2. Implement the logic in `main.py` (`walk_forward_backtest`) and `benchmark_assets.py`.
3. Add sidebar controls for the new options in BOTH `pages/1_🚀_Performance_Tracker.py` AND `pages/2_📊_Model_Analysis.py`.
4. Add session state defaults, `on_preset_change()` sync, `on_strategy_param_change()` detection, and `StrategyConfig()` construction in BOTH pages.
5. Add a new experiment preset in `EXPERIMENT_PRESETS` dict (in both pages) and in `run_experiments.py` `EXPERIMENTS` list.
6. If the new technique is the best-performing, make it the **default preset** (index=0 in the selectbox) so users get it by default.
7. The dashboard pages share an identical `EXPERIMENT_PRESETS` dict — keep them in sync at all times.

## Configuration

### Module-Level Constants (`main.py` lines 45-70)
- `TARGET_TICKER = '^SP500TR'`, `BOND_TICKER = 'VBMFX'`, `RISK_FREE_TICKER = '^IRX'`, `VIX_TICKER = '^VIX'`
- `START_DATE_DATA = '1987-01-01'` -- Data start (accommodates 11-year lookback)
- `OOS_START_DATE = '2007-01-01'` -- Out-of-sample start (paper: 2007-2023)
- `TRANSACTION_COST = 0.0005` -- 5 basis points
- `LAMBDA_GRID = [4.64, 10.0, 15.0, 21.54, 30.0, 46.42, 70.0, 100.0]` -- Dense 8-pt mid-range grid (Session 5: fills gaps for multi-asset; avoids low λ<4.6 that WF overpicks)
- `EWMA_HL_GRID = [0, 2, 4, 8]` -- EWMA halflife candidates for probability smoothing (fallback for unknown tickers)
- `PAPER_EWMA_HL` -- Dict mapping tickers to paper-prescribed halflives (hl=8: equity/bond/REIT, hl=4: commodity/gold, hl=2: corporate/NAESX, hl=0: EM/EAFE/HY). Used instead of auto-tuning.

### StrategyConfig (`config.py`)
Dataclass controlling experiment variants:
- `tuning_metric`: "sharpe" (default/paper) or "sortino"
- `lambda_smoothing`: bool -- EWMA smooth lambda selection across periods
- `lambda_ensemble_k`: int -- if > 1, average top-K lambda forecasts
- `validation_window_type`: "rolling" (default/paper, 5yr) or "expanding"
- `prob_threshold`: float (default 0.50, paper: 0.50)
- `allocation_style`: "binary" (default/paper) or "continuous"
- `lambda_selection`: "best" (default, argmax) or "median_positive" (median of positive-Sharpe lambdas, most stable CV=0.26)
- `lambda_subwindow_consensus`: bool -- split validation into 3 overlapping sub-windows, take median best-lambda (best avg Sharpe across assets)
- `xgb_params`: dict with XGBoost hyperparameters (default: XGBoost defaults per paper -- max_depth=6, learning_rate=0.3, no regularization)

## Experiment Framework

### Available Experiments (in `run_experiments.py` and dashboard EXPERIMENT_PRESETS)
1. Paper Baseline (reference)
2. Sortino Tuned
3. Conservative Threshold (0.6)
4. Continuous Allocation
5. Lambda Smoothing
6. Expanding Window
7. Lambda Ensemble (Top 3)
8. The Ultimate Combo
9. Expanding + Lambda Smoothing
10. Median-Positive Lambda (most stable, CV=0.26)
11. Sub-Window Consensus (best avg Sharpe across multi-asset)

### Report Output
Each run generates `benchmarks/experiment_report_YYYYMMDD_HHMMSS.md` containing:
- Results vs B&H table with Sharpe/Sortino/MDD deltas and WIN/LOSE verdicts
- Sub-period analysis (GFC, Recovery, Late Cycle, COVID, Post-COVID)
- Lambda stability tracking per experiment (mean, std, CV, timeline)
- Experiment configurations
- Future enhancements backlog with priority/risk ratings

### Key Findings (as of 2026-03-11)
- **LargeCap Sharpe 0.698** (Sub-Window Consensus) vs B&H 0.541 with dense 8pt grid (Session 9)
- **Multi-asset baseline: 7/11 WIN (64%)**, Sub-Window Consensus: 6/11 WIN but best avg Sharpe 0.560 (Session 9)
- **Binary 0/1 signal is essential** -- experiments #3, #4, #8 proved continuous/threshold changes destroy performance
- **Strategy wins in crises, loses in bull markets** -- GFC: strong outperformance; Recovery/COVID: underperformance
- **No single global lambda grid is optimal for all assets** -- different asset classes need different lambda ranges (Session 5)
- **Time period (2007-2026 vs 2007-2023) has negligible impact** on relative performance (-0.003 Sharpe delta)
- See `benchmarks/experiment_selection_20260303.md` for earlier analysis with regularized XGB

## Feature Set

Two groups fed to XGBoost (defined in `run_period_forecast()`):
- **Return features** (used by both JM and XGB): `DD_log_5/21`, `Avg_Ret_5/10/21`, `Sortino_5/10/21`
- **Macro features** (XGB only): `Yield_2Y_EWMA_diff`, `Yield_Slope_EWMA_10`, `Yield_Slope_EWMA_diff_21`, `VIX_EWMA_log_diff`, `Stock_Bond_Corr`

Return features are standardized (z-score) before feeding to the Jump Model. XGBoost receives raw feature values.

## Known Issues / Paper Gaps

- **XGBoost hyperparameters:** Now using default XGB params (max_depth=6, learning_rate=0.3, no regularization) as the paper specifies. Previous custom regularized params (max_depth=4, reg_alpha=1.0, reg_lambda=5.0) caused -0.174 Sharpe delta and made the strategy LOSE to B&H. Switching to defaults fixed this (Session 2, 2026-03-03).
- **Remaining Sharpe gap:** Paper reports 0.79 for LargeCap (2007-2023); we get ~0.645 with dense 8pt grid. JM-only baseline now matches paper (0.607 vs 0.59). Remaining JM-XGB gap from data source (Yahoo vs Bloomberg) and lambda grid effects.
- **JM fit_predict fixed (Session 8):** Was using 1 random init, 20 max iters. Now uses k-means++ init, n_init=10, max_iter=1000, tol=1e-8 — matching the paper's `jumpmodels` library exactly (100% state agreement, identical objective values). JM-only Sharpe improved from 0.534 to 0.607 at λ=100.
- **predict_online fixed (Session 4):** Previous greedy implementation only considered previous state cost, producing sticky regimes (18 shifts vs paper's 46). Now uses forward-only Viterbi (accumulated DP costs) matching the paper's `jumpmodels` library.
- **Lambda grid fixed (Sessions 4+5):** Wide grid [0, logspace(1,100)] caused walk-forward overfitting. Dense 8pt grid [4.64, 10, 15, 21.54, 30, 46.42, 70, 100] fills gaps for multi-asset coverage. Dashboard presets allow testing alternatives.
- **Multi-asset benchmark (Session 5):** 7/11 WIN (64%) on Long History assets vs paper's 11/12 (92%). Root causes: per-asset lambda sensitivity (no single global grid is optimal for all asset classes), FDIVX broken proxy (loses at ALL lambdas), Yahoo vs Bloomberg data gaps. Dense 8pt grid + NAESX hl=2 override improved from 5/11 to 7/11 WIN.
- **Data source:** Paper uses Bloomberg total return indices; we use Yahoo Finance. Causes ~0.14 avg Sharpe gap across assets (Session 3 finding).
- **OOS period:** We test 2007-2026 vs paper's 2007-2023. Diagnostic showed time period effect is negligible (-0.003 Sharpe delta).
- **EWMA halflife:** Auto-tuning on Yahoo data overfits validation window for some assets. Now using paper-prescribed halflives via `PAPER_EWMA_HL` dict (Sessions 3+5). NAESX overridden to hl=2 (Yahoo needs lower smoothing than Bloomberg). Falls back to auto-tuning for tickers not in the dict.

## Caching

- `cache/data_cache.pkl` -- Persisted fetched+engineered data (delete to re-fetch from APIs)
- `cache/data_cache_{ticker}_{date}.pkl` -- Per-ticker/per-list caches for multi-asset benchmark (date from asset list's data_start)
- `_forecast_cache` -- In-memory dict keyed by `(date, lambda, include_xgboost, constrain_xgb)`, lives only during script execution
- `cache/fred_cache.pkl` -- FRED Treasury yields (DGS2, DGS10), ticker-independent, shared across all backtest runs
- `cache/backtest_cache.pkl` -- Used by `app.py` for dashboard session persistence (also read by Data Quality Audit page for regime health checks)

## Session Memory & Experiment Tracking

Claude Code has an auto-memory directory that persists across conversations. The memory files are:
- **MEMORY.md** - Quick reference index (auto-loaded each session, keep <200 lines)
- **experiments.md** - Chronological log of experiments, findings, and changes (newest first)
- **performance_gaps.md** - Gap analysis vs paper and improvement priorities

**MANDATORY for Claude:** You MUST update memory files after significant work:
1. **At session start:** Read memory files to understand prior context before doing any work.
2. **After experiments:** Log experiment parameters, results, and conclusions in `experiments.md`. Update `MEMORY.md` if key metrics changed.
3. **After code changes:** Record what changed and why in `experiments.md`. Update `MEMORY.md` architecture decisions if applicable.
4. **After findings:** Update `performance_gaps.md` if gap analysis changed. Update `MEMORY.md` current performance numbers.
5. Keep `MEMORY.md` as a concise index (<200 lines); use topic files for details.
6. Ask user before major restructuring of memory files.

## Paper Reference

When verifying implementation against the paper, use `refcard.md` (in project root) as the primary reference. It contains all formulas, hyperparameters, pipeline steps, feature definitions, data splits, and numerical results from the paper in a compact format optimized for LLM context. The "Undisclosed" section at the end lists gaps where the paper is ambiguous.

## Python Compatibility

The top of `main.py` includes a distutils.version compatibility shim for Python 3.12+ (required by pandas-datareader). This must remain at the top of the file before other imports.
