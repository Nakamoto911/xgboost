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
python misc_scripts/benchmark_assets.py                      # Yahoo ETFs (12 investable ETFs, ~1993+)
python misc_scripts/benchmark_assets.py "Yahoo Mutual Funds" # Mutual fund proxies, ~1975+
python misc_scripts/benchmark_assets.py "Bloomberg Indices"  # Paper-aligned series from DATA PAUL.xlsx, 1989+
python misc_scripts/benchmark_assets.py "BBG+Yahoo ETF Hybrid"  # Paper-accurate lookback + investable OOS (splice in return-space)
python misc_scripts/benchmark_assets.py list                 # Show available asset lists

# Run pipeline diagnostics
python misc_scripts/diagnose_pipeline.py              # Full diagnostics
python misc_scripts/diagnose_pipeline.py --quick      # Skip slow tests (permutation)
```

There is no formal test framework (pytest/unittest). All tests, diagnostics, benchmarks, and debug scripts live in `misc_scripts/`. See `misc_scripts/TESTS.md` for a registry of all scripts and their purposes. **When creating a new test or diagnostic script, always place it in `misc_scripts/` and add an entry to `misc_scripts/TESTS.md`.**

## Architecture

### Entry Points
- `main.py` (~815 lines) -- Core backtest engine. Fetches data, fits models, runs walk-forward simulation, outputs PDF report.
- `app.py` & `pages/` -- Streamlit multi-page dashboard. `app.py` is the landing page.
  - `pages/1_📊_Model_Analysis.py`: Full backtest portal with parameter tuning via `st.form`, performance metrics, SHAP analysis, feature charts, JM audit, XGBoost evaluation, and PDF/JSON export. Tabs: Performance & Tracking, Feature Impact Analysis, Feature Charts, JM Audit, XGBoost Eval.
  - `pages/2_🛠️_Diagnostics_Launcher.py`: Control center for background scripts and MD report viewer.
  - `pages/3_🔍_Data_Quality_Audit.py`: Go/no-go data quality checks. Reads from existing caches (no pipeline re-run). Cache Freshness banner at top (os.path.getmtime for all cache files). Three sections: Raw Data Health (Yahoo + FRED, stale streaks, outliers, coverage, FRED yields), Feature Health (missing data summary with NaN% bar chart, z-score extremes, Sortino clipping, Stock-Bond Corr), Proxy Reliability (mutual fund vs ETF comparison — correlation, tracking error, return drift for all Yahoo Mutual Funds proxy pairs). Ticker selector at top.
  - `pages/4_📈_Portfolio_Construction.py`: Multi-asset MVO portfolio page reproducing paper Tables 6/7 and Figure 3. Universe selector (Bloomberg default / Yahoo ETFs), configurable rebalance frequency (daily default), MVO params (γ_risk, γ_trade, w_ub, covariance hl, μ lookbacks), in-sample μ toggle. Runs 7 paper strategies (MV/MinVar/EW × baseline/JM-XGB + ERC). Displays Table 6 (ours vs paper), Figure 3 (log-scale cumulative wealth), Table 7 (forecast correlation), diagnostics, and structural-gap explanation.
  - Sidebar parameters use `StrategyConfig`. Uses `walk_forward_backtest()` for execution through `main.py` alias `backend`. Experiment preset selector (all 11 experiments + Custom) auto-fills all StrategyConfig params.
- `run_experiments.py` (~300 lines) -- Experiment runner. Tests strategy variants via `StrategyConfig`, compares vs B&H, generates timestamped MD reports with sub-period analysis and lambda stability tracking.
- `misc_scripts/benchmark_assets.py` (~880 lines) -- Multi-asset benchmark. Tests configurable asset lists across 5 market periods with parallel execution. Asset lists defined in `misc_scripts/asset_lists.md`.
- `misc_scripts/diagnose_pipeline.py` (~800 lines) -- Pipeline health diagnostics. Generates a minimalist MD report assessing ML model quality and regime structural integrity.

### Supporting Files
- `config.py` -- `StrategyConfig` dataclass with all tunable strategy parameters.
- `portfolio.py` -- Multi-asset MVO portfolio engine (paper Section 4). `compute_asset_signals()` runs walk_forward_backtest per asset; `compute_insample_regime_means()` refits JM at each biannual anchor to get in-sample μ; `build_asset_panel()` aligns + shifts forecasts +1; `solve_mvo()` uses SLSQP + cvxpy/CLARABEL fallback; `run_all_portfolios()` runs 7 strategies. **μ and Σ are excess-return forecasts** (paper Section 4.1). In-sample JM regime means (~30%/yr bull) are the correct μ source for MV(JM-XGB) — OOS-conditioned μ (~5%/yr) collapses leverage due to γ_trade L1 penalty.
- `refcard.md` -- **Implementation reference card** extracted from the paper. Contains all formulas, hyperparameters, feature definitions, data splits, and numerical results. Use this instead of the full paper when verifying implementation correctness. Includes an "Undisclosed" section listing gaps the paper does not resolve.
- `misc_scripts/TESTS.md` -- **Test registry.** Documents all scripts in `misc_scripts/` with their purpose. Must be updated when adding new scripts.
- `misc_scripts/run_portfolio_paper.py` -- CLI script: loads signals + insample_mu, builds panel, runs all 7 portfolios, prints Tables 6/7 vs paper reference.
- `misc_scripts/smoke_test_portfolio.py` -- Fast smoke test: 3 BBG assets, 4-year window, validates pipeline end-to-end.
- `misc_scripts/asset_lists.md` -- Four named asset lists for multi-asset benchmark. Parsed by `benchmark_assets.py`:
  - **Yahoo ETFs** (~1993+): IVV, IJH, IWM, EFA, EEM, AGG, SPTL, HYG, SPBO, IYR, DBC, GLD (investable ETFs only — DBC's short 2006-02 inception is handled by per-asset partial-window logic)
  - **Yahoo Mutual Funds** (~1975+): ^SP500TR, VIMSX, NAESX, FDIVX, VEIEX, VBMFX, VUSTX, VWEHX, VWESX, FRESX, ^SPGSCI, GC=F
  - **Bloomberg Indices** (1989+): SPTR, SPTRMDCP, RU20INTR, NDDUEAFE, NDUEEGF, LBUSTRUU, LUTLTRUU, IBOXHY, LUACTRUU, DJUSRET, DBLCDBCE, GOLDLNPM — loaded from `cache/DATA PAUL.xlsx`
  - **BBG+Yahoo ETF Hybrid** (1989+): composite tickers `<BBG>+<ETF>` (e.g. `SPTR+IVV`). BBG drives history up to and including ETF inception date; Yahoo ETF drives returns afterwards. Splice is in return-space → continuous synthetic price. Purpose: paper-accurate JM lookback + investable OOS. Fixes the Commodity proxy issue (DBLCDBCE+DBC). Does NOT fix HYG/SPBO/AGG ETF tracking-error — those issues are inherited from the ETF leg by design.
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
- `1_📊_Model_Analysis.py` sidebar has an experiment preset selector (all 11 experiments + Custom). Selecting a preset auto-fills all StrategyConfig params. The backtest uses `walk_forward_backtest()` from main.py, ensuring all config options take effect.
- The dashboard mutates `backend.LAMBDA_GRID` and other module-level constants directly to pass data/grid configuration from sidebar controls.

### Dashboard Deployment Rule (MANDATORY)
**When any algorithm improvement is made (new StrategyConfig fields, new experiment presets, new walk-forward logic, etc.), it MUST be deployed to the dashboard and experiment runner.** Specifically:
1. Add new `StrategyConfig` fields to `config.py`.
2. Implement the logic in `main.py` (`walk_forward_backtest`) and `benchmark_assets.py`.
3. Add sidebar controls for the new options in `pages/1_📊_Model_Analysis.py`.
4. Add session state defaults, `on_preset_change()` sync, `on_strategy_param_change()` detection, and `StrategyConfig()` construction.
5. Add a new experiment preset in `EXPERIMENT_PRESETS` dict and in `run_experiments.py` `EXPERIMENTS` list.
6. If the new technique is the best-performing, make it the **default preset** (index=0 in the selectbox) so users get it by default.

## Configuration

### Module-Level Constants (`main.py` lines 45-70)
- `TARGET_TICKER = '^SP500TR'`, `BOND_TICKER = 'VBMFX'`, `RISK_FREE_TICKER = '^IRX'`, `VIX_TICKER = '^VIX'`
- `START_DATE_DATA = '1987-01-01'` -- Data start (accommodates 11-year lookback)
- `OOS_START_DATE = '2007-01-01'` -- Out-of-sample start (paper: 2007-2023)
- `TRANSACTION_COST = 0.0005` -- 5 basis points
- `LAMBDA_GRID = [4.64, 10.0, 15.0, 21.54, 30.0, 46.42, 70.0, 100.0]` -- Dense 8-pt mid-range grid (Session 5: fills gaps for multi-asset; avoids low λ<4.6 that WF overpicks)
- `EWMA_HL_GRID = [0, 1, 2, 4, 8, 12, 16]` -- Fixed auto-tune grid for EWMA halflife (not user-configurable)
- `PAPER_EWMA_HL` -- Dict mapping tickers to paper-prescribed halflives (hl=8: equity/bond/REIT, hl=4: commodity/gold, hl=2: corporate/NAESX, hl=0: EM/EAFE/HY). Used only when `ewma_mode="paper"`.

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
- `ewma_mode`: "auto" (default, tune on pre-OOS window using grid [0,1,2,4,8,12,16]) or "paper" (use PAPER_EWMA_HL dict values directly)
- `xgb_params`: dict with XGBoost hyperparameters (default: XGBoost defaults per paper -- max_depth=6, learning_rate=0.3, no regularization)

## Experiment Framework

### Available Experiments (in `run_experiments.py` and dashboard EXPERIMENT_PRESETS)
1. Paper Baseline -- reference implementation matching the paper (OOS 2007-2023, data start 1991, Dense 8pt lambda grid)
2. Optimized (default) -- Sub-Window Consensus, Focused No-100 lambda grid, end date = yesterday

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
- **Multi-asset benchmark (Session 5):** 7/11 WIN (64%) on Yahoo Mutual Funds assets vs paper's 11/12 (92%). Root causes: per-asset lambda sensitivity (no single global grid is optimal for all asset classes), FDIVX broken proxy (loses at ALL lambdas), Yahoo vs Bloomberg data gaps. Dense 8pt grid + NAESX hl=2 override improved from 5/11 to 7/11 WIN.
- **Data source:** Paper uses Bloomberg total return indices; we use Yahoo Finance. Causes ~0.14 avg Sharpe gap across assets (Session 3 finding).
- **OOS period:** We test 2007-2026 vs paper's 2007-2023. Diagnostic showed time period effect is negligible (-0.003 Sharpe delta).
- **EWMA halflife:** Controlled by `ewma_mode` on StrategyConfig. `"auto"` (default) tunes on pre-OOS window using fixed grid [0,1,2,4,8,12,16]. `"paper"` uses prescribed values from `PAPER_EWMA_HL` dict. Paper Baseline preset uses `"paper"`, all others use `"auto"`.

## Caching

All data caches **auto-refresh** when stale — no manual deletion needed.

### Cache Files
- `cache/data_cache_{ticker}_{start}_{end}.pkl` -- Per-ticker fetched+engineered data. Auto-refreshes if cached end date is >7 days behind requested `END_DATE`. Old cache files for the same ticker are cleaned up when a new one is saved.
- `cache/fred_cache.pkl` -- FRED Treasury yields (DGS2, DGS10), ticker-independent, shared across all backtest runs. Auto-refreshes if >7 days behind `END_DATE`.
- `cache/data_cache_{ticker}_{date}_v2.pkl` -- Per-ticker caches for multi-asset benchmark (`benchmark_assets.py`). Auto-refreshes if >30 days stale.
- `cache/backtest_cache.pkl` -- Used by `app.py` for dashboard session persistence (also read by Data Quality Audit page).
- `cache/portfolio_signals_{universe}_{oos_start}_{oos_end}.pkl` -- Per-universe portfolio signals cache (`portfolio.py`). Contains walk_forward_backtest results for all 12 assets. Refreshed via force-refresh button in page 4.
- `cache/portfolio_insample_mu_{universe}_{oos_start}_{oos_end}.pkl` -- In-sample JM regime means cache (`portfolio.py`). Contains μ_bull/μ_bear per asset per biannual anchor. Refreshed via force-refresh button in page 4.
- `_forecast_cache` -- In-memory dict keyed by `(date, lambda, include_xgboost, constrain_xgb)`, lives only during script execution.

### Staleness Rules
- `main.py` (`fetch_and_prepare_data`, `_fetch_fred_data`): compares cached data end date vs `END_DATE`; re-fetches if gap >7 calendar days (tolerates weekends/holidays).
- `benchmark_assets.py` (`fetch_etf_data`): compares cached end date vs today; re-fetches if >30 days stale. Uses dynamic `fetch_end = today + 1` instead of hard-coded dates.
- **Note:** ^IRX and ^VIX don't have standalone caches — they're fetched inline by `fetch_and_prepare_data()` and merged into the target ticker's cache as derived columns (`RF_Rate`, `VIX_EWMA_log_diff`).

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

## Bloomberg Data Source

`cache/DATA PAUL.xlsx` contains all 12 paper total-return series exported from Bloomberg (1989-01-02 to 2026-03-20). Parsed with `skiprows=6`; columns: `['Date', 'SPTR', 'SPTRMDCP', 'RU20INTR', 'NDDUEAFE', 'NDUEEGF', 'LBUSTRUU', 'IBOXHY', 'LUACTRUU', 'DJUSRET', 'DBLCDBCE', 'GOLDLNPM', 'LUTLTRUU']`. Loading is handled by `_load_bbg_raw()` / `_load_bbg_price_series()` in `benchmark_assets.py` — called automatically when ticker is in `BBG_PRICE_COLS`.

**^SPGSCI vs DBLCDBCE:** Both represent Commodity but are NOT interchangeable across lists. ^SPGSCI is the Yahoo Finance index proxy used in "Yahoo Mutual Funds" (replaces delisted PCASX/PCRAX, daily-return corr 0.928 with DBLCDBCE, 1984+ history). DBLCDBCE is the Bloomberg column name used in "Bloomberg Indices". Replacing one with the other in the wrong list will silently break data loading.

**PAPER_EWMA_HL for Bloomberg tickers:** hl=8 for SPTR/SPTRMDCP/RU20INTR/LBUSTRUU/LUTLTRUU/DJUSRET; hl=4 for DBLCDBCE/GOLDLNPM; hl=2 for LUACTRUU; hl=0 for NDUEEGF/NDDUEAFE/IBOXHY. DD excluded: LBUSTRUU, LUTLTRUU, GOLDLNPM.

## Data Source Divergence (Yahoo vs Bloomberg)

Generated by `python misc_scripts/compare_data_sources.py` (or via Diagnostics Launcher → "Compare Data Sources"). The 12 paper assets fall into four divergence tiers when comparing Yahoo proxies to Bloomberg total-return indices on the 2007-2023 OOS window:

| Tier | Assets | Cause | Direction |
|---|---|---|---|
| 1. Clean (6/12) | LargeCap, MidCap, SmallCap, REIT, Treasury, Gold | Yahoo ≈ Bloomberg (ρ ≥ 0.94) | n/a |
| 2. Credit-ETF liquidity premium (3/12) | **HighYield (HYG: vol +5.3%, Sharpe −0.30)**, Corporate (SPBO: vol +1.8%), AggBond (AGG: vol +1.2%) | ETF prices include bid-ask spread + premium/discount to NAV; BBG indices are NAV-based bond marks | ETF inflates vol; mutual fund proxies (VWEHX/VWESX/VBMFX) transact at NAV → tighter tracking |
| 3. International timezone mismatch (2/12) | **EM (EEM: vol +9.7%, ρ=0.64)**, EAFE (EFA: vol +4.5%, ρ=0.69) | US-listed ETFs trade past local-market close; BBG NDDUEAFE/NDUEEGF mark each constituent at local close + FX | Daily Yahoo returns embed post-close US activity → vol inflation, low correlation with BBG |
| 4. Wrong-index proxy (1/12) | **Commodity** (^SPGSCI in Yahoo MF list) | GSCI ≈ 60% energy; BCOM (DBLCDBCE) ≈ 30% energy — different baskets, not proxies | ^SPGSCI return drift +1.4%, vol +4.5% vs BBG. Fixable via DBC (inception 2006-02) in the Hybrid list |

**Practical impact:** The MVO optimizer in `portfolio.py` reads μ and σ from these series. Tier-2 and Tier-3 inflated σ causes the optimizer to underweight those assets on Yahoo runs. The Bloomberg run eliminates the data noise but a residual leverage gap (0.71 ours vs paper 0.86) persists — that is a joint-forecast-distribution issue, not a single-asset data quality issue.

**Pre-OOS-inception note:** All ETFs have inception ≥ 2000; SPBO inception 2011-04-07 is the latest. The "Hybrid" asset list (below) was added to combine BBG history pre-inception with Yahoo ETF returns post-inception.

## BBG+Yahoo ETF Hybrid Asset List

Added Session 22 (2026-05-13). The 4th asset list, beside Yahoo ETFs, Yahoo Mutual Funds, and Bloomberg Indices.

**Composite ticker format:** `<BBG>+<ETF>` (e.g. `SPTR+IVV`). The BBG leg drives daily returns up to AND INCLUDING the ETF's first available date; Yahoo ETF leg drives returns from the day after. Splice is performed in return-space then a synthetic continuous price is rebuilt — JM/XGB downstream sees one seamless series, no scale discontinuity.

**Why:** Paper-accurate JM lookback + investable OOS execution. Also fixes the ^SPGSCI/DBLCDBCE mismatch (Hybrid uses DBLCDBCE pre-2006-02, DBC after).

**What it does NOT fix:** HYG/SPBO/AGG ETF tracking-error issues are inherited from the ETF leg by design (the splice happens before OOS for these). EFA/EEM timezone mismatch likewise survives.

**Smoke-test results (B&H Sharpe 2007-2023 vs paper Table 4):**
- 8/12 match paper within ±0.03 (LargeCap, MidCap, SmallCap, EAFE, EM, Treasury, REIT, Gold)
- **Commodity FIXED**: DBLCDBCE+DBC = 0.047 (paper 0.03) — was the main motivation
- HYG (-0.31), SPBO (-0.13), AGG (-0.10) inherit ETF-leg drift — predicted and expected

**Implementation hooks:**
- `misc_scripts/asset_lists.md` → "BBG+Yahoo ETF Hybrid" section
- `misc_scripts/benchmark_assets.py` → `_parse_hybrid_ticker()` + `_load_hybrid_price_series()`; routes from `fetch_etf_data` when ticker contains `+`. Entries added to `PAPER_EWMA_HL`, `DD_EXCLUDE_TICKERS`, `TICKER_TO_PAPER_ASSET`.
- `portfolio.py` → `HYBRID_ASSETS` list (5-tuple specs), `_build_hybrid_price_series()`, `_build_hybrid_features()`. `_asset_specs_for('hybrid')`, `_start_date_for('hybrid')='1991-01-01'`.
- Cache filenames: `+` is sanitized to `_PLUS_`.

**Where to select the Hybrid:**
- CLI: `python misc_scripts/benchmark_assets.py "BBG+Yahoo ETF Hybrid"`
- CLI: `python misc_scripts/run_portfolio_paper.py hybrid`
- CLI: `XGB_DATA_SOURCE=hybrid python misc_scripts/run_multi_asset_ablation.py`
- Streamlit page 2 (Diagnostics Launcher): Benchmark Assets and Multi-Asset Macro Ablation selectors
- Streamlit Portfolio Construction page (`portfolio_construction.py`): universe dropdown
- Python: `portfolio.compute_asset_signals(universe='hybrid')`

## Paper Reference

When verifying implementation against the paper, use `refcard.md` (in project root) as the primary reference. It contains all formulas, hyperparameters, pipeline steps, feature definitions, data splits, and numerical results from the paper in a compact format optimized for LLM context. The "Undisclosed" section at the end lists gaps where the paper is ambiguous.

## Python Compatibility

The top of `main.py` includes a distutils.version compatibility shim for Python 3.12+ (required by pandas-datareader). This must remain at the top of the file before other imports.
