# JM-XGB Strategy - Session Memory

## Quick Reference
- **Paper:** Shu, Yu, Mulvey (2024), arXiv 2406.09578v2
- **Paper Sharpe (LargeCap):** 0.79 (2007-2023), B&H ~0.50
- **Our best so far:** See [experiments.md](experiments.md)
- **Key files:** `main.py` (engine), `config.py` (params), `run_experiments.py` (runner)

## Architecture Decisions
- Binary 0/1 allocation is essential (continuous destroys performance)
- EWMA halflife tuned once on pre-OOS window, fixed for all OOS
- Walk-forward tunes only lambda (not hl+lambda jointly)
- XGBoost uses paper defaults (max_depth=6, lr=0.3, no regularization) - FIXED in Session 2

## Current Performance (Session 2, 2026-03-03)
- **Paper Baseline: Sharpe 0.566 vs B&H 0.541 (+0.025) -- BEATS B&H**
- MDD: -23.37% vs B&H -55.25% (58% improvement)
- Remaining gap to paper: 0.566 vs 0.79 (likely data source: Yahoo vs Bloomberg)
- See [performance_gaps.md](performance_gaps.md) and [experiments.md](experiments.md)

## Key Patterns
- Strategy wins in crises (GFC: +0.72 Sharpe vs B&H), loses in bull markets
- Lambda instability (CV=1.67 baseline) is a persistent concern
- Post-COVID period (2022-2025) particularly weak for strategy

## User Preferences
- Wants systematic experiment tracking across sessions
- Focus on Paper Baseline first before exploring improvements
