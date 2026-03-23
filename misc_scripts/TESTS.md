# Test & Script Registry

All standalone tests, diagnostics, and benchmarks live in `misc_scripts/`. This file documents each script's purpose. **When adding a new script, add an entry here.**

## Benchmarks

| Script | Purpose |
|---|---|
| `benchmark_assets.py` | Multi-asset benchmark: tests JM-XGB vs B&H across configurable asset lists (12 ETFs, mutual fund proxies) with parallel execution and 5 market periods. Main benchmark tool. |
| `benchmark_2007_2023.py` | Multi-asset benchmark restricted to 2007-2023 to match paper's Table 4 for direct comparison. |
| `benchmark.py` | Micro-benchmark comparing two Viterbi implementations (loop-based vs vectorized matrix operations). |

## Diagnostics

| Script | Purpose |
|---|---|
| `diagnose_pipeline.py` | Full pipeline health diagnostics. Generates MD report assessing ML model quality and regime structural integrity. |
| `diagnose_baseline.py` | Diagnoses Paper Baseline underperformance by testing time period and XGBoost hyperparameter effects. |
| `diagnose_fast.py` | Fast diagnostic suite (skips SHAP) to quickly diagnose signal quality issues. |
| `diagnose_gap.py` | Investigates JM-XGB Sharpe gap vs paper by decomposing pipeline quality (regimes, forecasts, lambda tuning). |
| `diagnose_gap2.py` | Lean diagnostic testing alternative grid/window configurations via walk_forward_backtest. |
| `diagnose_jm.py` | Identifies implementation differences (predict_online, n_init, max_iter, state sorting) between our JM and the paper's reference. |
| `diagnose_jm_lib.py` | Compares our JM implementation against paper's official `jumpmodels` library across lambda sweep. |
| `diagnose_jm_lib2.py` | Fine-grain lambda sweep (60-100) where `jumpmodels` showed best results, tests walk-forward tuning. |
| `diagnose_multi_asset.py` | Diagnoses multi-asset benchmark gaps vs paper by testing time period, lambda sensitivity, and WF choices. |
| `diagnose_performance.py` | Comprehensive diagnostic: signal quality, time-in-market, sub-periods, EWMA sensitivity, lambda tuning. |
| `diagnose_regimes.py` | Compares our JM regime dates against paper's Figure 2 using both our implementation and `jumpmodels` library. |
| `diagnose_regimes_viterbi.py` | Uses full Viterbi (fit_predict/predict) to extract bear regime dates and compare against paper's Figure 2. |
| `quick_diag_vwesx_fresx.py` | Quick diagnostic for VWESX and FRESX with custom lambda grids and EWMA halflives. |

## Tests

| Script | Purpose |
|---|---|
| `test_before_after.py` | Compares BEFORE (default XGB, fixed hl=8) vs AFTER (regularized XGB, tuned hl) configurations. |
| `test_download.py` | Streamlit component test for PDF export via base64-encoded download link. |
| `test_initial_hl_tuning.py` | Tests paper's approach of jointly tuning (HL, lambda) on initial validation window (2002-2007). |
| `test_joint_tuning.py` | Tests joint tuning of lambda + EWMA halflife + threshold following paper methodology. |
| `test_jm_only_multiasset.py` | Tests JM-only (no XGBoost) performance across multiple assets. |
| `test_legend.py` | Test script for Plotly legend positioning with multiple legend groups. |
| `test_one_class.py` | Edge case: SHAP computation when XGBoost classifier is trained on single-class data. |
| `test_shap.py` | Checks for missing SHAP values in backtest_cache.pkl. |
| `test_walkforward.py` | Walk-forward lambda trace: validates lambda selection across 6-month rolling periods. |
| `test_warm_predict.py` | Tests warm-start Viterbi (training+OOS data) vs cold-start (OOS-only) performance. |
| `test_wf_robustness_multiasset.py` | Multi-asset walk-forward robustness: compares lambda selection strategies across 11 assets. |
| `test_xgb_single.py` | Edge case: XGBoost predict_proba behavior when trained on single-class data. |
| `test_yf.py` | Basic Yahoo Finance download test (^GSPC ticker). |
| `test_yf_treasury.py` | Tests Yahoo Finance Treasury yield data (TNX ticker). |
| `test_yf_treasury2.py` | Tests different Treasury yield ticker formats (US2Y=X, US10Y=X, US2Y). |

## Ablation Studies

| Script | Purpose |
|---|---|
| `run_macro_ablation.py` | Macro feature ablation: compares all features vs return-only vs macro-only XGBoost configurations. |

## Bloomberg Replication

| Script | Purpose |
|---|---|
| `test_bloomberg_data.py` | Bloomberg SPTR data validation: grid sweep (λ=0 variants, regime stats), bear period analysis, λ=0 hypothesis test. Uses `cache/DATA PAUL.xlsx`. |
| `investigate_gap.py` | Systematic investigation of remaining 0.09 Sharpe gap vs paper. Tests: XGBoost tree_method (A), lambda grid fine sweep N=20-32 (B), sub-window consensus + finer grids (C), predict_online DP initialization (D), fixed-λ XGB decomposition (E), Sortino clipping (F), per-period EWMA smoothing (G). |
| `test_bbg_assets.py` | Tests n_estimators=100 vs 200 on Bloomberg data for REIT (DJUSRET) and AggBond (LBUSTRUU). Validates generalizability of n_est=200 improvement across assets. Compares vs paper Table 4 targets. |

## Data Checks

| Script | Purpose |
|---|---|
| `check_dates.py` | Tests current data availability from Yahoo Finance (^SP500TR) and FRED Treasury yields. |

## Debug Helpers

| Script | Purpose |
|---|---|
| `debug_plotly.py` | Creates mock Plotly subplot layout to test dashboard visualization structure. |
| `debug_shap.py` | Inspects SHAP values from backtest_cache.pkl. |
