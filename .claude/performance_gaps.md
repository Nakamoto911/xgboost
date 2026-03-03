# Performance Gap Analysis

## Paper vs Implementation Comparison

| Factor | Paper | Our Implementation | Impact |
|---|---|---|---|
| OOS Period | 2007-2023 | 2007-2026 | MODERATE - 2024-25 drags performance |
| Data Source | Bloomberg total return | Yahoo Finance `^SP500TR` | UNKNOWN |
| XGB Params | "default" (max_depth=6, lr=0.3, no reg) | Now using defaults (FIXED Session 2) | RESOLVED (+0.174 Sharpe) |
| Lambda Grid | "log-spaced 0-100" (count unknown) | 11 candidates (0 + logspace(0,2,10)) | LOW |
| EWMA halflife | Tuned once per asset | Tuned once on pre-OOS | MATCHES paper |
| Allocation | Binary 0/1 | Binary 0/1 | MATCHES paper |

## Sub-Period Performance Pattern
| Period | Strategy vs B&H | Why |
|---|---|---|
| GFC (2007-09) | BIG WIN (+0.72) | Correctly identifies bear regime |
| Recovery (2010-15) | BIG LOSE (-0.38) | False bear signals during bull |
| Late Cycle (2016-19) | Slight LOSE | Misses some rally |
| COVID (2020-21) | LOSE (-0.19) | Misses V-shaped recovery |
| Post-COVID (2022-25) | Slight WIN (+0.14) | Mixed regime benefits strategy |

## Top Priorities to Close Gap
1. ~~Test with paper's OOS period (2007-2023)~~ DONE - negligible effect (-0.003)
2. ~~Test with XGBoost default params~~ DONE - **+0.174 Sharpe delta, strategy now beats B&H**
3. **Feature engineering audit** vs paper Tables 2 & 3 (remaining gap: 0.566 vs paper 0.79)
4. **Data source** (Yahoo vs Bloomberg) likely explains remaining ~0.23 gap
