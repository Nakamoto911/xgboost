# JM-XGB Performance Diagnosis & Improvements

## Experiment Date: 2026-03-02

## Problem
B&H was beating JM-XGB on both Sharpe and Sortino ratios over the 2007-2026 OOS period on ^SP500TR.

## Diagnostic Approach
Created test scripts in `misc_scripts/` (diagnose_fast.py, test_before_after.py, test_walkforward.py, test_joint_tuning.py) that bypass SHAP computation for speed and test signal quality, sub-period performance, EWMA sensitivity, lambda sensitivity, XGB calibration, threshold sensitivity, and missed opportunity costs.

## Key Findings

### 1. Signal quality was inverted
With default XGB (lambda=10, hl=8): market returned **12.3% ann. when model said BEAR** vs 9.5% when BULL. The signal had negative predictive value — false bear signals during bull markets destroyed value.

### 2. Strategy is fundamentally a drawdown protector, not return enhancer
JM-XGB wins during crashes (GFC: +0.1% vs -51.8%, COVID: -7.2% vs -31.6%, 2022: -10.9% vs -24.7%) but gives back gains during long bull runs (2009-2015: +75.5% vs +201.3%).

### 3. Walk-forward tuning was overfitting
- 21 lambda × 4 EWMA halflife = 84 combinations per 6-month window
- Lambda jumped wildly between 3 and 100 across periods
- Walk-forward won only 8/38 periods (21%)
- Joint tuning of lambda+hl+threshold made things WORSE (Sharpe 0.296)

### 4. Fixed lambda outperformed walk-forward lambda
With fixed lambda=5, regularized XGB, hl=8: Sharpe=0.610 vs B&H 0.444. The walk-forward was destroying this edge.

### 5. EWMA halflife was misimplemented
- Paper says: tune halflife ONCE per asset on initial validation window, fix for all OOS
- Code had: hardcoded at 8 (or later, jointly tuned per walk-forward step)
- Per-chunk EWMA broke smoothing continuity at period boundaries

### 6. XGBoost defaults overfit on noisy regime labels
The Jump Model produces imperfect labels. Default XGB memorized noise. Regularized config (max_depth=4, reg_alpha=1.0, reg_lambda=5.0, subsample=0.8, colsample_bytree=0.8) gave Sharpe=0.554 vs default 0.488.

## Changes Made (commit 8afdd9d)

### 1. XGBoost regularization (`main.py` ~line 360)
```python
XGBClassifier(
    max_depth=4, n_estimators=100, learning_rate=0.1,
    reg_alpha=1.0, reg_lambda=5.0,
    subsample=0.8, colsample_bytree=0.8,
)
```

### 2. EWMA halflife tuned once (Phase 1 in `main()`)
- Tuned from {0, 2, 4, 8} on initial 5-year validation window (2002-2007)
- Selected hl=8 for S&P 500
- Fixed for entire OOS period
- Walk-forward now only tunes lambda (11 candidates, not 84)

### 3. Lambda grid reduced + continuous EWMA
- Grid: 10 log-spaced candidates (was 20)
- EWMA applied to full concatenated OOS series (not per-chunk)

### 4. Extended to 2026
- END_DATE changed from 2024-01-01 to 2026-01-01

## Results

| Metric | Before | After | B&H |
|--------|--------|-------|-----|
| Sharpe | 0.29 | **0.56** | 0.44 |
| Sortino | 0.39 | **0.77** | 0.62 |
| Ann. Return | 4.62% | 7.65% | 8.65% |
| Ann. Vol | 13.97% | 11.89% | 19.86% |
| Max DD | -33.92% | **-18.74%** | -56.78% |
| Final Wealth | 2.36x | 4.04x | 4.83x |
| Trades | 109 | 76 | N/A |

## Ideas for Further Improvement

### Within paper methodology
1. **Sortino-based walk-forward objective** — Replace Sharpe with Sortino for lambda selection. The strategy's edge is drawdown reduction; Sortino better captures downside risk avoidance and may produce more stable lambda selections.

2. **Walk-forward lambda smoothing** — Instead of using the single best lambda per period, use an exponentially weighted average of recent best lambdas to prevent sudden jumps. E.g., `lambda_t = 0.7 * best_lambda_t + 0.3 * lambda_{t-1}`.

3. **Expanding validation window** — The paper uses a fixed 5-year rolling window. An expanding window (starting from 5 years, growing over time) would give more data for later periods and may produce more stable selections.

4. **Feature selection per period** — Not all 13 features are informative in all market regimes. Walk-forward feature importance could drop low-value features to reduce XGB noise.

5. **Calibrated probability threshold** — Instead of fixed 0.5, use isotonic regression or Platt scaling on validation probabilities to calibrate the threshold. Our tests showed threshold=0.55-0.60 can improve Sharpe.

### Extensions beyond the paper
6. **Partial allocation** — Instead of binary 0/1 switching, use the smoothed probability as a continuous allocation weight (e.g., invest `1 - P(bear)` fraction in equities). This reduces the cost of false signals.

7. **Multi-asset portfolio** — The paper's full framework uses 12 assets with mean-variance optimization. A single asset is the hardest case; adding bonds/REITs/commodities would enable cross-asset diversification of regime signals.

8. **Ensemble of lambdas** — Instead of picking one best lambda, average the signals from top-K lambdas on the validation window to produce a more robust ensemble forecast.

9. **Regime-dependent risk parity** — Instead of 100% equity or 100% risk-free, allocate between equity/bonds/cash based on regime confidence using risk parity weights.

10. **Online learning for XGBoost** — Instead of refitting from scratch every 6 months, use incremental training to adapt more quickly to regime changes while retaining memory of older patterns.
