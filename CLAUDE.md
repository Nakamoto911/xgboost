# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quantitative trading strategy combining a Statistical Jump Model (JM) for market regime identification with XGBoost for return forecasting. Based on arXiv paper 2406.09578v2. Walk-forward backtested from 2007-2026 on S&P 500 Total Return data.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full backtest (generates timestamped PDF report)
python main.py

# Launch interactive Streamlit dashboard
streamlit run app.py

# Run ad-hoc test scripts
python misc_scripts/test_download.py
python misc_scripts/benchmark.py
```

There is no formal test framework (pytest/unittest). Tests are standalone scripts in `misc_scripts/`.

## Architecture

**Two entry points:**
- `main.py` (651 lines) — Core backtest engine. Self-contained: fetches data, fits models, runs walk-forward simulation, outputs PDF report.
- `app.py` (2059 lines) — Streamlit dashboard. Imports `main.py` as `backend` and exposes all parameters via sidebar controls with interactive Plotly charts.

**Algorithm pipeline (in `main.py`):**
1. `fetch_and_prepare_data()` — Downloads from Yahoo Finance + FRED, engineers features (EWMA-based return/macro features), caches to `data_cache.pkl`
2. `StatisticalJumpModel` class — 2-state regime model using alternating optimization (K-means + Viterbi). Optimized fast-path for 2 states in the Viterbi forward pass.
3. `run_period_forecast()` — For a given date: fits JM on 11-year lookback, trains XGBClassifier on regime labels + macro features, predicts 6-month OOS window. Computes SHAP values. Results cached in `_forecast_cache` dict.
4. `simulate_strategy()` — Chains 6-month forecast periods. Applies EWMA smoothing (halflife=8) to raw probabilities, thresholds at 0.5, shifts signals by 1 day (look-ahead bias prevention), applies transaction costs.
5. `main()` — Walk-forward loop: every 6 months tunes lambda on 5-year validation window (maximize Sharpe), then runs OOS chunk with best lambda.

**Key design decisions:**
- State 0 = Bullish (invest in target asset), State 1 = Bearish (rotate to risk-free). States are aligned after fitting so State 0 always has higher cumulative excess return.
- Forecast signal is shifted +1 day before applying to returns to prevent look-ahead bias.
- `app.py` mutates `backend.LAMBDA_GRID` and other module-level constants directly to pass configuration from sidebar controls.

## Configuration

All key parameters are module-level constants in `main.py` (lines 45-66):
- `TARGET_TICKER`, `BOND_TICKER`, `RISK_FREE_TICKER`, `VIX_TICKER` — Asset tickers
- `START_DATE_DATA = '1987-01-01'` — Data start (accommodates 11-year lookback)
- `OOS_START_DATE = '2007-01-01'` — Out-of-sample start (paper: 2007-2023)
- `TRANSACTION_COST = 0.0005` — 5 basis points
- `LAMBDA_GRID = list(np.logspace(0, 2, 20))` — 20 log-spaced jump penalty candidates (1 to 100)

## Caching

- `data_cache.pkl` — Persisted fetched+engineered data (delete to re-fetch from APIs)
- `_forecast_cache` — In-memory dict keyed by `(date, lambda, include_xgboost, constrain_xgb)`, lives only during script execution
- `backtest_cache.pkl` — Used by `app.py` for dashboard session persistence

## Feature Set

Two groups fed to XGBoost (defined in `run_period_forecast()`):
- **Return features** (used by both JM and XGB): `DD_log_5/21`, `Avg_Ret_5/10/21`, `Sortino_5/10/21`
- **Macro features** (XGB only): `Yield_2Y_EWMA_diff`, `Yield_Slope_EWMA_10`, `Yield_Slope_EWMA_diff_21`, `VIX_EWMA_log_diff`, `Stock_Bond_Corr`

Return features are standardized (z-score) before feeding to the Jump Model. XGBoost receives raw feature values.

## Python Compatibility

The top of `main.py` includes a distutils.version compatibility shim for Python 3.12+ (required by pandas-datareader). This must remain at the top of the file before other imports.
