# XGBoost Regime Switching Strategy

This repository contains an implementation of a quantitative trading strategy that combines a Statistical Jump Model (JM) for market regime identification with an XGBoost classifier for return forecasting.

## Main Algorithm

![Strategy Flow Diagram](images/strategy_flow.jpg)

The core algorithm operates in two primary phases: Regime Identification and Forecasting, which are evaluated iteratively over time using a rolling window approach.

### 1. Market Regime Identification (Statistical Jump Model)
The strategy employs a discrete 2-state Statistical Jump Model (JM) to classify the market environment into distinct regimes (e.g., typically modeling "Bull" / low-volatility and "Bear" / high-volatility states).
- **Optimization Method**: The model uses an alternating optimization approach, combining K-means style state updates with the Viterbi algorithm to determine the most likely sequence of hidden states.
- **Jump Penalty**: A jump penalty ($\lambda$) is applied to the objective function to constrain the sequence of states. This penalizes frequent state transitions, preventing excessive and unrealistic switching between market regimes due to short-term noise.
- **Online Prediction**: As new data arrives, the model assigns data points to clusters in real-time, considering the last known state and the jump penalty constraint.

### 2. Return Forecasting (XGBoost)
Once the market regime framework is established, an XGBoost classifier is utilized to forecast future price movements.
- **Lookback Window**: The model is trained on a rolling historical observation window (e.g., an 11-year lookback).
- **Regime-Conditioned Predictions**: By incorporating the regime identified by the JM and additional engineered features, the XGBoost model learns non-linear relationships to generate actionable market predictions.

### 3. Strategy Simulation
The overall strategy is simulated in discrete steps (e.g., 6-month forward-rolling chunks).
1. **Model Fitting**: For each period, the algorithm fits the Jump Model to identify continuous market states over the lookback window.
2. **Training Forecasting Model**: The XGBoost model is trained on the identified regimes and related features.
3. **Execution**: The strategy makes allocation decisions (e.g., investing in the risky asset or a risk-free alternative) over the forecast horizon based on the model's predictions, aiming to maximize risk-adjusted returns and avoid large drawdowns during turbulent periods.

## Data and Feature Engineering

### Data Sources
The primary data is fetched using Yahoo Finance and FRED APIs:
- **Target Asset**: S&P 500 Total Return (`^SP500TR`)
- **Bond Proxy**: Vanguard Total Bond Market (`VBMFX`)
- **Risk-Free Rate**: 13-Week Treasury Bill (`^IRX`)
- **Volatility Index**: VIX (`^VIX`)
- **Macro Drivers**: 2-year (`DGS2`) and 10-year (`DGS10`) Treasury Yields from FRED.

Data is fetched from 1987 onwards to accommodate the 11-year training lookback period for the earliest out-of-sample test date.

### Engineered Features
The algorithm computes both Asset-Specific and Cross-Asset Macro features:
- **Asset Specifics**: Log downside deviation (5, 21-day EWMA), Average Returns (5, 10, 21-day EWMA), and Sortino Ratio variants. Return calculation factors in the calculated risk-free daily rate.
- **Macro Features**: 2-Year Yield differentials, Yield Curve Slope (10yr - 2yr with EWMAs), VIX log differences, and 252-day correlation sequences between Stock and Bond returns.

## Backtesting Results

The strategy evaluates backtesting in a fully out-of-sample, walk-forward basis (e.g. from 2007 to early 2026), simulating a real-time trading environment:
- **Walk-Forward Tuning**: Every 6 months, the model tunes the lambda penalty parameter over a 5-year validation window, maximizing the Sharpe ratio. The optimally identified $\lambda$ is then fixed for the next 6 months of out-of-sample forward trading.
- **Evaluation Points**: The simulation records Returns, Volatility, Sharpe Ratio, Sortino Ratio, Max Drawdown, and number of trades generated for the standard Buy-and-Hold versus the complete JM-XGB framework.
- **Report Generation**: The script outputs comprehensive performance metrics and charts plotting wealth curves across the test period to a timestamped PDF, highlighting identified "Bear Regimes" where the strategy rotated into the risk-free asset

## Project Structure

```
.
├── main.py                  # Core backtest engine (S&P 500, generates PDF report)
├── app.py                   # Streamlit interactive dashboard
├── run_experiments.py        # Experiment runner (strategy hypothesis testing)
├── config.py                # StrategyConfig dataclass for parameterizing variants
├── requirements.txt
├── cache/                   # Auto-generated data caches (gitignored)
│   ├── data_cache.pkl       # Fetched + engineered S&P 500 data (main.py)
│   ├── backtest_cache.pkl   # Last dashboard run results (app.py)
│   └── data_cache_*_*.pkl   # Per-asset caches (benchmark_assets.py)
├── benchmarks/              # Timestamped benchmark outputs (gitignored)
│   ├── experiment_report_YYYYMMDD_HHMMSS.md   # run_experiments.py output
│   ├── benchmark_report_YYYYMMDD_HHMMSS.md    # benchmark_assets.py output
│   └── benchmark_results_YYYYMMDD_HHMMSS.csv  # benchmark_assets.py output
└── misc_scripts/
    ├── benchmark_assets.py  # Multi-asset robustness testing
    ├── diagnose_pipeline.py # Pipeline health diagnostics
    ├── asset_lists.md       # Configurable asset lists (tickers, classes, data_start)
    └── ...                  # Other debug/test scripts
```

## Environment Setup

1. Standard Python environment (3.8+ recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running

There are several entry points depending on your goal:

### 1. Strategy Research: Experiment Runner (Recommended for testing hypotheses)
Tests a controlled set of 9 pre-defined strategy variants and compares each against the paper baseline. This is the primary tool for validating improvements.

```bash
python run_experiments.py [all|list|N|N,M|N-M]
```

**Options:**
- `python run_experiments.py all` — Run all 9 experiments
- `python run_experiments.py list` — List available experiments
- `python run_experiments.py 1,5,6` — Run experiments #1, #5, #6
- `python run_experiments.py 5-7` — Run experiments #5 through #7

**Output:** Timestamped markdown report (`benchmarks/experiment_report_YYYYMMDD_HHMMSS.md`) with:
- Side-by-side comparison table (Sharpe, Sortino, Max DD vs B&H)
- Sub-period performance breakdown (GFC, Recovery, Late Cycle, COVID, Post-COVID)
- Lambda stability diagnostics (mean, std, coefficient of variation)
- Integration recommendations for each experiment
- Future enhancement backlog

**Experiments tested:**
1. Paper Baseline (reference)
2. Sortino Tuned (optimize for downside volatility)
3. Conservative Threshold (0.6 instead of 0.5)
4. Continuous Allocation (fractional instead of binary)
5. Lambda Smoothing (smooth lambda selection across periods)
6. Expanding Window (growing validation window instead of rolling)
7. Lambda Ensemble (average top-3 lambdas)
8. The Ultimate Combo (all experimental features)
9. Expanding + Lambda Smoothing (combination)

### 2. Interactive Exploration: Dashboard
Launches the Streamlit app for interactive exploration of strategy parameters and results. Select an experiment preset or customize parameters manually, run the backtest, and view live charts:

```bash
streamlit run app.py
```

**Features:**
- Experiment preset selector (auto-fills all parameters)
- Customizable Strategy Parameters (tuning metric, validation window type, lambda smoothing, etc.)
- Data & Execution settings (tickers, dates, lambda grid, XGB hyperparameters)
- Interactive Plotly charts (wealth curve, drawdowns, sub-periods)
- SHAP importance analysis
- Results cached for fast reload

Opens in your default browser. The last run is cached in `cache/backtest_cache.pkl`.

### 3. Full Backtest (S&P 500)
Runs the walk-forward backtest with current parameters and saves a timestamped PDF report:

```bash
python main.py
```

**Output:** PDF report with:
- Performance metrics table (Sharpe, Sortino, Max DD, trades)
- Wealth curve chart with annotated bear regimes
- Feature importance rankings

Data is cached in `cache/data_cache.pkl` to avoid re-fetching on subsequent runs. Delete it to force a fresh download from Yahoo Finance and FRED.

### 4. Multi-Asset Robustness Testing (Optional)
Tests the finalized strategy across configurable asset lists to verify the edge generalizes beyond S&P 500:

```bash
python misc_scripts/benchmark_assets.py                # Default ETFs (12 ETFs)
python misc_scripts/benchmark_assets.py "Long History"  # Long-history mutual fund proxies
python misc_scripts/benchmark_assets.py list            # Show available asset lists
```

**Output:** Two files in `benchmarks/`:
- `benchmark_report_YYYYMMDD_HHMMSS.md` — full results with per-period tables
- `benchmark_results_YYYYMMDD_HHMMSS.csv` — raw data for further analysis

**Asset lists** are defined in `misc_scripts/asset_lists.md`. Each list has a name, tickers with asset class groupings, and a `data_start` date. Two lists are included:
- **Default ETFs** (12 ETFs: IVV, IJH, IWM, EFA, EEM, AGG, SPTL, HYG, SPBO, IYR, DBC, GLD)
- **Long History** (12 mutual fund proxies with data back to 1975-1998: ^SP500TR, VIMSX, NAESX, FDIVX, VEIEX, VBMFX, VUSTX, VWEHX, VWESX, FRESX, PCASX, GC=F)

To add a new list, add a `## List Name` section with `data_start:` and a ticker table to `misc_scripts/asset_lists.md`.

**Optimization notes:** Reduced lambda grid (5 points), no SHAP, multiprocessing for speed.

Per-asset data cached in `cache/data_cache_<TICKER>_<DATE>.pkl`. Delete individual files to re-fetch.

### 5. Pipeline Health Diagnostics
Analyzes the ML health of the pipeline independent of financial returns, generating a comprehensive markdown report. Useful for verifying the structural integrity of the regime classifications and predictive signals.

```bash
python misc_scripts/diagnose_pipeline.py
python misc_scripts/diagnose_pipeline.py --quick  # Skip slow permutation tests
```

**Output:** A timestamped markdown report in `benchmarks/` containing:
- 14-point "Gate Checklist" to quickly identify pipeline failures
- JM Regime Quality (Silhouette, Davies-Bouldin, Return separation)
- XGBoost OOS Classification metrics (Accuracy, ROC-AUC, Log-loss, MCC)
- Signal calibration and overfitting analysis
- Rank IC / Information Ratio across multiple horizons

---

### Typical Workflow

1. **Use `run_experiments.py`** to test hypothesis-driven strategy modifications and generate comparison reports
2. **Use `app.py`** to interactively explore promising variants and visualize results
3. **Use `main.py`** to generate a final PDF report of your chosen configuration
4. **Use `benchmark_assets.py`** to verify the strategy works across other asset classes (optional, for robustness)
5. **Use `diagnose_pipeline.py`** to verify the underlying structural health of the ML models
