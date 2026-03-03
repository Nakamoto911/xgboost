# Experiment Log

Each session records experiments run, parameters tested, results, and conclusions.
Entries are in reverse chronological order (newest first).

---

## Session 2026-03-03 (Session 2) - Diagnosing Paper Baseline vs B&H

**Goal:** Understand why Paper Baseline doesn't consistently beat B&H on Sharpe.

### Hypotheses Being Tested
1. Extended OOS period (2007-2026 vs paper's 2007-2023) drags performance
2. XGBoost hyperparams deviate from paper "defaults"
3. Data source (Yahoo vs Bloomberg) affects feature quality

### Experiments Run
| # | Description | Config Changes | Sharpe | B&H Sharpe | Delta | Notes |
|---|---|---|---|---|---|---|
| 1 | Paper Baseline (before) | regularized XGB | 0.392 | 0.541 | -0.150 | LOSES to B&H |
| 2 | Paper period + reg XGB | END_DATE=2024, reg XGB | 0.346 | 0.499 | -0.152 | LOSES to B&H |
| 3 | Current + default XGB | default XGB params | 0.566 | 0.541 | +0.025 | BEATS B&H |
| 4 | Paper period + default XGB | END_DATE=2024, default XGB | 0.556 | 0.499 | +0.058 | BEATS B&H |

### Effect Decomposition
- **XGB params effect: +0.174 Sharpe delta** (dominant factor!)
- Time period effect: -0.003 (negligible)
- Interaction: +0.036

### Findings
1. **XGBoost over-regularization was THE root cause.** Custom params (max_depth=4, reg_alpha=1.0, reg_lambda=5.0) suppressed model learning after EWMA/lambda grid fixes from Session 1.
2. **Time period (2007-2026 vs 2007-2023) has essentially zero impact** on relative performance.
3. Default XGB also improves lambda stability (CV 1.24 → 1.07).
4. Remaining gap to paper (0.566 vs 0.79) likely from data source (Yahoo vs Bloomberg).

### Changes Made
- `config.py`: Switched xgb_params to paper defaults (max_depth=6, lr=0.3, no regularization)
- Updated CLAUDE.md, PERFORMANCE_DIAGNOSIS.md to reflect findings
- Created `misc_scripts/diagnose_baseline.py` for the 4-way diagnostic

---

## Session 2026-03-03 (Session 1) - Initial Setup & Experiment Framework

**Goal:** Build experiment framework, run all 9 experiments, diagnose performance.

### Key Results (from experiment reports)
- Paper Baseline: Sharpe ~0.652, B&H ~0.580, Delta +0.072
- Lambda Smoothing: Sharpe +0.097 vs B&H (recommended)
- Expanding Window: Sharpe +0.117 vs B&H (but lambda degenerates to 0)
- Conservative Threshold, Continuous Allocation, Ultimate Combo all LOSE to B&H

### Changes Made (commit 8afdd9d)
1. XGBoost regularization (max_depth=4, reg_alpha=1.0, reg_lambda=5.0)
2. EWMA halflife tuned once on pre-OOS window
3. Lambda grid reduced from 20 to 10 candidates
4. Extended OOS to 2026

### Key Insight
Strategy is a drawdown protector, not return enhancer. Wins in crises, loses in bulls.
