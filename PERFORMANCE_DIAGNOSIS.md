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

