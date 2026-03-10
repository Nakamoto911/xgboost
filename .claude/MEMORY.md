# JM-XGB Strategy - Session Memory

## Quick Reference
- **Paper:** Shu, Yu, Mulvey (2024), arXiv 2406.09578v2
- **Paper Sharpe (LargeCap):** 0.79 (2007-2023), B&H ~0.50
- **Our current (after Session 4b fixes):** Sharpe 0.675 (default grid) / 0.852 (no-100 grid)
- **Key files:** `main.py` (engine), `config.py` (params), `run_experiments.py` (runner)
- **Diagnostics:** `misc_scripts/diagnose_gap.py`, `diagnose_gap2.py`, `diagnose_jm_lib.py`, `diagnose_regimes_viterbi.py`

## Architecture Decisions
- Binary 0/1 allocation is essential (continuous destroys performance)
- EWMA halflife: use paper-prescribed values from `PAPER_EWMA_HL` dict (Session 3 fix)
- Walk-forward tunes only lambda (not hl+lambda jointly)
- XGBoost uses paper defaults (max_depth=6, lr=0.3, no regularization) — FIXED in Session 2
- `predict_online` uses forward-only Viterbi (accumulated DP costs) — FIXED in Session 4b
- Lambda grid focused mid-range [4.64, 10, 21.54, 46.42, 100] — FIXED in Session 4b

## Fixes Applied (Session 4b, 2026-03-10)

### Fix 1: predict_online → Forward-Only Viterbi
- Old: greedy (only previous state cost) → 18 shifts, sticky regimes, false bears
- New: `values[t] = loss[t] + min_k(values[t-1,k] + penalty[k,:])`, then `argmin(values[t])`
- Matches paper's `jumpmodels` library implementation exactly
- Fixed in both `main.py` and `benchmark_assets.py`

### Fix 2: Lambda Grid → Focused Mid-Range
- Old: `[0.0] + logspace(0,2,10)` → walk-forward picks extreme lambdas (0, 100)
- New: `[4.64, 10.0, 21.54, 46.42, 100.0]` (default) or `[4.64, 10, 21.54, 46.42]` (best)
- Dashboard has 5 presets: Focused 5pt (default), Focused 4pt, Legacy Wide, Expanded, Custom

## Current Performance (2007-2023, ^SP500TR)
- **Default (5pt grid):** Sharpe 0.675 vs B&H 0.499 (+0.177), MDD -22.0%
- **No-100 (4pt grid):** Sharpe 0.852 vs B&H 0.499 (+0.354), MDD -22.0%
- **Paper reference:** Sharpe 0.79, MDD -24.78%, Bear% 20.9%, 46 shifts
- Remaining gap to paper (~0.12): Yahoo vs Bloomberg data source

## PAPER_EWMA_HL (Session 3)
- hl=8: LargeCap, MidCap, SmallCap, REIT, AggBond, Treasury
- hl=4: Commodity, Gold
- hl=2: Corporate
- hl=0: EM, EAFE, HighYield

## Key Patterns
- Strategy wins in crises (GFC), loses in bull markets (Recovery)
- Lambda instability (CV>1.0) is THE primary performance destroyer
- XGB probabilities bimodal: 59.8% below 0.3, 37.1% above 0.5
- Multi-asset avg (before 4b): Sharpe 0.50 vs B&H 0.44 (+0.06) — 5/11 WIN

## User Preferences
- Wants systematic experiment tracking across sessions
- Focus on Paper Baseline first before exploring improvements
- See [experiments.md](experiments.md) for full log
