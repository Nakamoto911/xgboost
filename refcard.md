# Shu, Yu & Mulvey (2024) — Dynamic Asset Allocation with Asset-Specific Regime Forecasts — Reference Card

> **What this document is:** A compressed, implementation-focused reference card extracted from the research paper *"Dynamic Asset Allocation with Asset-Specific Regime Forecasts"* by Yizhan Shu, Chenyu Yu, and John M. Mulvey (arXiv:2406.09578v2, August 2024). It is intended to replace the full PDF in LLM context when verifying implementation code against the paper. All formulas, hyperparameters, pipeline steps, feature definitions, data splits, and numerical results are reproduced faithfully from the paper. Qualitative discussion, literature review, and narrative have been omitted. Use the **Undisclosed** section at the end to identify gaps the paper does not resolve.

---

## Data

| Item | Detail |
|---|---|
| Assets | 12 risky + 1 risk-free (US T-bill 3M CMY from FRED) |
| Period | 1991–2023, daily total return, USD, Bloomberg Terminal |
| Frequency | Daily |
| Risk-free proxy | US Treasury 3-month constant maturity yield |

**Asset list:**

| Abbrev | Index | ETF |
|---|---|---|
| LargeCap | S&P 500 TR (SPTR) | IVV |
| MidCap | S&P MidCap 400 TR (SPTRMDCP) | IJH |
| SmallCap | Russell 2000 TR (RU20INTR) | IWM |
| EAFE | MSCI EAFE Net TR USD (NDDUEAFE) | EFA |
| EM | MSCI EM Net TR USD (NDUEEGF) | EEM |
| AggBond | Bloomberg US Agg Bond TR (LBUSTRUU) | AGG |
| Treasury | Bloomberg US Long Treasury TR (LUTLTRUU) | SPTL* |
| HighYield | iBoxx USD Liquid HY TR (IBOXHY) | HYG† |
| Corporate | Bloomberg US Corporate TR (LUACTRUU) | SPBO |
| REIT | Dow Jones US RE TR (DJUSRET) | IYR‡ |
| Commodity | DB DBIQ Optimum Yield Div Cmdty ER (DBLCDBCE) | DBC |
| Gold | LBMA Gold Price PM USD (GOLDLNPM) | GLD |

*Prior to 1994-03-02: S&P US Treasury Bond 20+ Year TR (SPBDUSLT)  
†Prior to 1999-01-04: Vanguard High-Yield Corporate Fund TR (VWEHX)  
‡Prior to 1992-01-03: Dow Jones Equity REIT TR (REIT index)

---

## Features

### JM Features (return features — Table 2)
Computed per asset from **excess return series** (vs risk-free). 8 features total. All EWM-smoothed.

| Feature | Halflives (trading days) |
|---|---|
| Downside Deviation — log scale, i.e. `log(EWM_DD(hl))` where `DD = sqrt(EWM(min(r,0)^2, hl))` | 5, 21 |
| Average Return — `EWM_avg(r_excess, hl)` | 5, 10, 21 |
| Sortino Ratio — `EWM_avg(r_excess, hl) / EWM_DD(hl)` | 5, 10, 21 |

**Exception:** AggBond, Treasury, Gold — DD features EXCLUDED (only 6 features: avg_return ×3, Sortino ×3). Reason: DD features don't separate regimes in-sample for these assets.

### XGBoost Features (return features + macro features — Tables 2 & 3)
All 8 JM return features above PLUS:

| Feature | Transformation |
|---|---|
| US Treasury 2Y Yield | `diff(EWM(yield_2y, hl=21))` — EWMA of daily difference |
| Yield Curve Slope (10Y−2Y) | `EWM(slope, hl=10)` |
| Yield Curve Slope (10Y−2Y) | `diff(EWM(slope, hl=21))` — EWMA of daily change |
| VIX Index | `EWM(log_diff(VIX), hl=63)` |
| Stock-Bond Correlation | Rolling corr(LargeCap_ret, AggBond_ret), 1-year lookback (≈252 days) |

Source for macro features: FRED (daily, real-time available). All 8 return features are asset-specific; macro features identical for all assets.

---

## Model Spec

### Step 1 — Statistical Jump Model (JM)

**Objective:**
$$\min_{\Theta, S} \sum_{t=0}^{T-1} \ell(x_t, \theta_{s_t}) + \lambda \sum_{t=1}^{T-1} \mathbf{1}[s_{t-1} \neq s_t]$$

- Loss: scaled squared ℓ2-distance: `ℓ(x, θ) = 0.5 * ||x - θ||²`
- K = 2 states (bullish=0, bearish=1)
- Labeling: regime with higher cumulative excess return → bullish (s=0)
- Solver: coordinate descent (alternating optimization of Θ and S)
- Implementation reference: https://github.com/Yizhan-Oliver-Shu/jump-models (scikit-learn API)

### Step 2 — XGBoost Classifier

- Model: `XGBClassifier` with **default hyperparameters** (no tuning)
- Target: `ŝ_{t+1}` (JM labels shifted forward 1 day)
- Output: predicted probability → threshold 0.5 → binary forecast
- Post-processing: EWM smoothing of probability series (halflife selected per asset, see Hyperparameters)

---

## Pipeline

```
For each asset, every 6 months:
  1. Fit JM:
     - Data: 11-year lookback training window
     - Features: Table 2 (return features only)
     - Output: optimal state sequence ŝ_0…ŝ_{T-1}
  2. Shift labels: target = ŝ_{t+1}
  3. Fit XGBClassifier:
     - Same 11-year training window
     - Features: Tables 2 + 3 (return + macro)
     - Target: shifted labels
  4. Daily online forecasting for next 6 months:
     - XGBClassifier outputs P(bullish) for t+1
     - Apply EWM smoothing to probability series
     - Threshold at 0.5 → f_{t+1} ∈ {bull, bear}
  5. Every 6 months from start of testing period:
     - Run Algorithm 1 over 5-year validation window for λ candidates [0.0–100.0, log-uniform]
     - Compute Sharpe of 0/1 strategy per λ
     - Select optimal λ̂ → use for next 6 months of live forecasting
```

---

## Hyperparameters

| Param | Value/Range | Selection Method |
|---|---|---|
| λ (jump penalty) | [0.0, 100.0], log-uniform grid | Time-series CV: maximize Sharpe of 0/1 strategy on 5-year validation window; updated biannually |
| JM K (states) | 2 | Fixed |
| JM lookback window | 11 years | Fixed |
| JM update frequency | Biannual (every 6 months) | Fixed |
| XGBoost params | All defaults | No tuning |
| XGBoost update frequency | Biannual | Fixed |
| Probability smoothing halflife | Asset-specific (see below) | Selected on initial validation window 2002–2007 using 0/1 strategy Sharpe |
| Covariance EWM halflife | 252 days | Fixed |

**Per-asset probability smoothing halflives:**

| hl=8 days | hl=4 days | hl=2 days | hl=0 (none) |
|---|---|---|---|
| LargeCap, MidCap, SmallCap, REIT, AggBond, Treasury | Commodity, Gold | Corporate | EM, EAFE, HighYield |

---

## Splits & Walk-Forward

| Period | Dates | Purpose |
|---|---|---|
| Initial training window | 1991–2002 (11 years) | First JM + XGB fit |
| First validation window | 2002–2007 (5 years) | λ selection + smoothing hl selection |
| Testing period | 2007–2023 | Out-of-sample results reported |

- **Biannual updates:** every 6 months from start of testing period
- λ updated biannually using 5-year lookback validation window (rolling)
- JM and XGB both re-fit on same 11-year rolling training window

---

## Portfolio Construction

**MVO formulation:**
$$\max_w \; w^T\mu - \gamma_{\text{risk}} w^T\Sigma w - \gamma_{\text{trade}} \cdot a \|w - w_{\text{pre}}\|_1$$
$$\text{s.t.} \quad 0 \leq w \leq w^{ub}, \quad \mathbf{1}^T w \leq L$$

- `a` = one-way transaction cost = **5 bps**
- `w^{ub}` = 40% per asset
- `L` = 1 (no leverage; cash = `1 - 1ᵀw`)
- Solver: Gurobi via `gurobipy`

**Risk-free fallback rule (ALL models):** if ≤3 assets forecast as bullish → allocate 100% to risk-free asset.

### MinVar (JM-XGB)
- Input returns: `μ_j = 10 bps` if bullish forecast, `μ_j = 0` if bearish
- `γ_risk = 10.0`, `γ_trade = 1.0`
- Covariance: EWM historical, hl=252 days

### MinVar (baseline)
- Input: constant `μ ∝ 1` (equivalent to minimizing variance)
- `γ_risk = 10.0`, `γ_trade = 0.0` (no trading cost term)

### MV (JM-XGB)
- Return forecast: average return of training-window periods in the **same forecasted regime** under λ̂
- Cap bearish regime return forecasts at **−10 bps**
- `γ_risk = 10.0`, `γ_trade = 1.0`
- Covariance: EWM historical, hl=252 days
- If ≤3 bullish → 100% risk-free

### MV (baseline)
- Return forecast: EWM of historical returns, hl=5 years
- `γ_risk = 5.0`, `γ_trade = 0.0`

### EW (JM-XGB)
- Equal weight across all bullish-forecast assets; 0 to bearish
- Total weight = 100% (fully invested among bullish set); rebalance daily
- If ≤3 bullish → 100% risk-free

### EW (baseline)
- Fixed 1/12 to each of 12 risky assets; rebalance daily

### 60/40 Fix-Mix Benchmark

| LargeCap | MidCap | SmallCap | EAFE | EM | REIT | HighYield | Commodity | Gold | Treasury | Corporate | AggBond |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10% | 5% | 5% | 5% | 5% | 10% | 10% | 5% | 5% | 10% | 10% | 20% |

Rebalanced daily.

---

## Results

### Table 4 — 0/1 Strategy Performance (2007–2023)

**Sharpe Ratio:**

| Strategy | LargeCap | MidCap | SmallCap | EAFE | EM | REIT | AggBond | Treasury | HighYield | Corporate | Commodity | Gold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B&H | 0.50 | 0.45 | 0.36 | 0.20 | 0.20 | 0.27 | 0.46 | 0.26 | 0.67 | 0.54 | 0.03 | 0.43 |
| JM | 0.59 | 0.49 | 0.28 | 0.28 | 0.65 | 0.39 | 0.43 | 0.21 | 1.49 | 0.83 | 0.08 | 0.12 |
| JM-XGB | 0.79 | 0.59 | 0.51 | 0.56 | 0.85 | 0.56 | 0.67 | 0.38 | 1.88 | 0.76 | 0.23 | 0.31 |

**Max Drawdown:**

| Strategy | LargeCap | MidCap | SmallCap | EAFE | EM | REIT | AggBond | Treasury | HighYield | Corporate | Commodity | Gold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B&H | -55.25% | -55.15% | -58.89% | -60.41% | -65.25% | -74.23% | -18.41% | -46.91% | -32.87% | -22.04% | -75.54% | -44.62% |
| JM | -24.78% | -33.24% | -38.35% | -29.72% | -26.22% | -54.71% | -6.09% | -22.85% | -13.88% | -8.26% | -58.48% | -31.78% |
| JM-XGB | -17.69% | -29.89% | -35.84% | -19.93% | -21.30% | -32.70% | -6.30% | -17.46% | -10.25% | -6.79% | -47.90% | -21.62% |

### Table 6 — Portfolio Performance (2007–2023), annualized excess returns, rf=1.1%/yr

| Metric | 60/40 | MinVar | MinVar(JM-XGB) | MV | MV(JM-XGB) | EW | EW(JM-XGB) |
|---|---|---|---|---|---|---|---|
| Return | 5.0% | 2.8% | 3.9% | 2.6% | 8.9% | 5.5% | 8.2% |
| Volatility | 8.9% | 4.0% | 3.5% | 7.1% | 8.7% | 10.8% | 9.0% |
| Sharpe | 0.57 | 0.70 | 1.12 | 0.37 | 1.02 | 0.51 | 0.91 |
| MDD | -31.5% | -19.3% | -7.1% | -25.6% | -13.5% | -37.5% | -17.6% |
| Calmar | 0.16 | 0.15 | 0.55 | 0.10 | 0.66 | 0.15 | 0.47 |
| Turnover | 0.74 | 0.49 | 2.06 | 3.40 | 9.12 | 0.81 | 11.70 |
| Leverage | 1.00 | 1.00 | 0.91 | 0.95 | 0.86 | 1.00 | 0.92 |

### Table 7 — Return Forecast Correlation with Actual (2007–2023)

| Method | Overall | LargeCap | MidCap | SmallCap | EAFE | EM | REIT | AggBond | Treasury | HighYield | Corporate | Commodity | Gold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EWMA | -1.04% | -1.58% | -3.86% | -3.72% | -3.73% | -2.03% | -5.09% | 1.25% | -1.17% | -0.16% | -0.06% | -1.05% | -1.59% |
| JM-XGB | 2.43% | 1.66% | 0.90% | 1.03% | 4.53% | 6.02% | 2.10% | 3.22% | 1.64% | 10.54% | 2.62% | 3.39% | 0.32% |

### Tables 8–9 — Sensitivity Analysis

**MinVar(JM-XGB) vs γ_trade:**

| Metric | γ_trade=0.0 | γ_trade=1.0 (default) |
|---|---|---|
| Return | 4.3% | 3.9% |
| Volatility | 4.2% | 3.5% |
| Sharpe | 1.02 | 1.12 |
| MDD | -12.8% | -7.1% |
| Calmar | 0.34 | 0.55 |
| Turnover | 11.80 | 2.06 |
| Leverage | 0.91 | 0.91 |

**MV(JM-XGB) vs γ_risk (γ_trade=1.0 fixed):**

| Metric | γ_risk=5.0 | γ_risk=10.0 (default) | γ_risk=20.0 |
|---|---|---|---|
| Return | 10.1% | 8.9% | 6.7% |
| Volatility | 10.0% | 8.7% | 6.9% |
| Sharpe | 1.01 | 1.02 | 0.96 |
| MDD | -15.4% | -13.5% | -13.5% |
| Calmar | 0.65 | 0.66 | 0.49 |
| Turnover | 10.03 | 9.12 | 7.67 |
| Leverage | 0.89 | 0.86 | 0.76 |

---

## Figure-Only Data

### Figure 2 — Regime forecasts (2007–2023), LargeCap, REIT, AggBond

| Asset | % Bear Market | # Regime Shifts |
|---|---|---|
| LargeCap | 20.9% | 46 |
| REIT | 18.4% | 46 |
| AggBond | 41.5% | 97 |

**Approximate bear regime periods (from Figure 2 axis reading):**

*LargeCap (pink shading):*
- ~2008-09 to 2009-06 (GFC main)
- Several short episodes ~2010, ~2011
- Short episodes ~2015–2016
- ~2020-02 to 2020-04 (COVID)
- Multiple short episodes ~2022

*REIT (orange shading):*
- ~2007-07 to 2009-06 (started earlier than LargeCap — subprime)
- Short episodes ~2011
- ~2020-02 to 2020-05
- Multiple short episodes ~2022

*AggBond (blue shading):*
- ~2007-07 to 2008-01
- ~2009-01 to 2009-06
- Multiple episodes ~2013–2014
- ~2018-02 to 2019-06
- ~2021-10 to 2023-12 (rate hike cycle — large block)

---

## Undisclosed

- **DD formula**: paper says "downside deviation" but does not give the exact formula. Assumed: `EWM(min(r_excess, 0)^2, hl)^0.5` — verify exact treatment of zeros.
- **Sortino sign**: paper says ratio = avg_return / DD; does not clarify whether numerator is raw EWM avg or itself EWM of exceedances.
- **EWM convention**: paper does not specify `adjust=True/False` or `min_periods` for pandas EWM. Affects early-sample behavior.
- **Log-uniform λ grid**: "evenly on a logarithmic scale" from 0.0 to 100.0 — exact grid size (number of points) not stated. λ=0.0 is problematic for log scale; likely `logspace` starting near 0 or handling 0 separately.
- **Biannual update dates**: exact calendar anchoring (Jan/Jul? start of data? next business day?) not specified.
- **XGBoost version**: default hyperparameters differ by version; paper does not pin XGBoost version.
- **Probability smoothing implementation**: applied to raw XGBoost output probability P(bull) before thresholding — EWM direction (causal, no look-ahead) implied but not stated.
- **Covariance matrix regularization**: only EWM stated; no shrinkage, clipping, or factor model mentioned.
- **MV return forecast regime assignment**: "average return from historical periods in the same regime under λ̂" — it is unclear whether this is the in-sample JM regime (from the most recent 11-year fit) or a running label.
- **-10 bps cap**: applied to bearish return forecast in MV only — confirm not applied in MinVar or EW.
- **"≤3 bullish" rule**: threshold is 3 or fewer → 100% cash. Applies to all three portfolio models (MinVar, MV, EW) with JM-XGB.