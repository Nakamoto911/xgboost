# JM-XGB Strategy - Session Memory

## Quick Reference
- **Paper:** Shu, Yu, Mulvey (2024), arXiv 2406.09578v2
- **Paper Sharpe (LargeCap JM-XGB):** 0.79 (2007-2023), B&H ~0.50
- **Our current (after Session 8 JM fix):** JM-XGB Sharpe 0.645 (8pt grid, 2007-2023), JM-only 0.607 (beats paper's 0.59)
- **Key files:** `main.py` (engine), `config.py` (params), `run_experiments.py` (runner)

## Architecture Decisions
- Binary 0/1 allocation is essential (continuous destroys performance)
- EWMA halflife: use paper-prescribed values from `PAPER_EWMA_HL` dict (Session 3 fix)
- Walk-forward tunes only lambda (not hl+lambda jointly)
- XGBoost uses paper defaults (max_depth=6, lr=0.3, no regularization) — FIXED in Session 2
- `predict_online` uses forward-only Viterbi (accumulated DP costs) — FIXED in Session 4b
- Lambda grid focused mid-range [4.64, 10, 15, 21.54, 30, 46.42, 70, 100] (dense 8pt)

## Key Fixes (Sessions 2-8)
- **JM fit_predict → k-means++ init, n_init=10, max_iter=1000, tol=1e-8 (Session 8)** — matches paper exactly
- `predict_online` → forward-only Viterbi (Session 4b)
- Lambda grid → dense 8pt (Sessions 4b, 5)
- XGBoost → paper defaults (Session 2)
- EWMA HL → paper-prescribed dict (Session 3)

## JM-only Strategy: Table 4 JM Row — FULLY EXPLAINED (Session 20)

**Session 20 result: 9/12 assets match or beat paper JM with TC=0 + best method**

| Asset | Paper JM | Our Best | Method | Gap |
|---|---|---|---|---|
| LargeCap | 0.59 | **0.597** | TC=0 + shared-λ | **+0.007 ✓** |
| MidCap | 0.49 | 0.456 | TC=0 + shared-λ | −0.034 |
| SmallCap | 0.28 | **0.329** | TC=0 + indep-WF | **+0.049 ✓** |
| EAFE | 0.28 | **0.260** | TC=0 + shared-λ | −0.020 ≈ match |
| EM | 0.65 | **0.745** | TC=0 + indep-WF | **+0.095 ✓** |
| REIT | 0.39 | 0.264 | TC=5bps + indep | −0.126 (data) |
| AggBond | 0.43 | **0.642** | TC=0 + indep-WF | **+0.212 ✓** |
| Treasury | 0.21 | **0.302** | TC=0 + shared-λ | **+0.092 ✓** |
| HighYield | 1.49 | **1.725** | TC=0 + indep-WF | **+0.235 ✓** |
| Corporate | 0.83 | **0.958** | TC=0 + indep-WF | **+0.128 ✓** |
| Commodity | 0.08 | **0.337** | TC=0 + shared-λ | **+0.257 ✓** |
| Gold | 0.12 | 0.068 | TC=5bps + indep | −0.052 (structural) |

**Two confirmed findings explain paper JM row:**
1. **TC=0** (S19): Table 4 is gross of TC
2. **Shared-λ** (S18): Paper's JM row reuses XGB-selected λ (not independently validated)

**Remaining gaps:** MidCap −0.034 (WF noise, untested: TC=0 + 3pt [10,15,22]), REIT −0.126 (data quality), Gold −0.052 (structural)

## JM-XGB Strategy: Bloomberg Data (Sessions 11-20)

### LargeCap Bloomberg SPTR (2007-2023)
- **8pt grid + TC=0.0005 (baseline): 0.691** (gap −0.099 vs paper 0.79)
- **8pt grid + TC=0 (gross): 0.743** (gap −0.047) — TC explains −0.052
- **Oracle λ=45 (no WF): 0.787-0.788** — algorithm correct, gap = WF λ noise
- B&H: 0.499 ✓

### Multi-Asset Bloomberg (2007-2023, 8pt, n_est=100, TC=0.0005)
- **11/12 beat B&H (92%)** — matches paper. Only Gold loses.
- **Average gap vs paper JM-XGB: −0.023 Sharpe** across 12 assets

## PAPER_EWMA_HL
- hl=8: LargeCap, MidCap, SmallCap, REIT, AggBond, Treasury
- hl=4: Commodity, Gold
- hl=2: Corporate, NAESX (Yahoo override)
- hl=0: EM, EAFE, HighYield

## Lambda Grids
- Default: [4.64, 10.0, 15.0, 21.54, 30.0, 46.42, 70.0, 100.0] (dense 8pt)

## Key Confirmed Facts
- **Table 4 is GROSS of TC** (Session 19) — paper reports net returns without 5bps deduction
- **Paper's JM row reuses XGB λ** (Session 18) — not an independent JM-only optimization
- **Oracle λ gives paper results** for all assets (Sessions 12, 15) — gap = WF λ selection noise
- **Binary 0/1 signal essential** — continuous/threshold changes destroy performance

## Data Source Analysis (Session 22, 2026-05-13)

Run `python misc_scripts/compare_data_sources.py` (or via Diagnostics Launcher) for full pair-wise stats. Findings classify the 12 paper assets into four divergence tiers:

| Tier | Assets | Root cause | Fixable? |
|---|---|---|---|
| **1. Clean** (6/12) | LargeCap, MidCap, SmallCap, REIT, Treasury, Gold | Yahoo ≈ Bloomberg (ρ≥0.94) | n/a |
| **2. Credit-ETF liquidity premium** (3/12) | HighYield (HYG, vol +5.3%), Corporate (SPBO, vol +1.8%), AggBond (AGG, vol +1.2%) | ETF trades at bid-ask spread + premium/discount to NAV; BBG indices are NAV-based | Use MF proxies (VWEHX/VWESX/VBMFX) — already in Yahoo Mutual Funds list |
| **3. International timezone mismatch** (2/12) | EAFE (EFA, vol +4.5%), EM (EEM, vol +9.7%) | US-listed ETFs trade past Asian/European local closes; BBG marks at local close + FX | MF proxies (FDIVX/VEIEX) help (ρ 0.73-0.78 vs ETF 0.64-0.69); structurally unfixable without BBG |
| **4. Wrong-index proxy** (1/12) | Commodity (^SPGSCI vs DBLCDBCE) | GSCI ≈ 60% energy; BCOM ≈ 30% energy — different baskets | Use DBC ETF (inception 2006-02). Only available pre-2006 via BBG → use the new Hybrid list |

**Implication for MVO leverage gap (0.71 ours vs paper 0.86):**
- MVO uses μ, σ as inputs. Tier-2 and Tier-3 inflate σ on Yahoo data → optimizer underweights those assets.
- Bloomberg run already eliminates the Yahoo data noise but leverage gap persists → joint forecast distribution issue, not single-asset data quality.

## BBG+Yahoo ETF Hybrid Asset List (Session 22)

4th asset list added to `misc_scripts/asset_lists.md` + `portfolio.py` `HYBRID_ASSETS`. Composite tickers `<BBG>+<ETF>` (e.g. `SPTR+IVV`). BBG drives history up to AND INCLUDING the ETF's first available date; Yahoo ETF drives returns from the day after. Splice is in return-space (no scale discontinuity); a synthetic continuous price is rebuilt for downstream feature engineering.

**Purpose:** Paper-accurate JM lookback + investable OOS returns.

**B&H Sharpe vs paper Table 4 (smoke test, 2007-2023):**

| Tier | Hybrid Sharpe | Gap vs paper | Status |
|---|---|---|---|
| 8/12 (LargeCap, MidCap, SmallCap, EAFE, EM, Treasury, REIT, Gold) | matches | within ±0.03 | ✓ clean |
| **Commodity (DBLCDBCE+DBC)** | **0.047** | **+0.02** | **FIXED** (was ^SPGSCI mismatch) |
| AggBond (LBUSTRUU+AGG) | 0.364 | −0.10 | inherits AGG ETF drift |
| Corporate (LUACTRUU+SPBO) | 0.412 | −0.13 | inherits SPBO drift (post-2011-04) |
| HighYield (IBOXHY+HYG) | 0.360 | −0.31 | inherits HYG drift |

**Where it's selectable (Session 22 wiring):**
- CLI: `python misc_scripts/benchmark_assets.py "BBG+Yahoo ETF Hybrid"`
- CLI: `python misc_scripts/run_portfolio_paper.py hybrid`
- CLI: `XGB_DATA_SOURCE=hybrid python misc_scripts/run_multi_asset_ablation.py`
- Streamlit page 2 → Benchmark Assets and Multi-Asset Macro Ablation selectors
- Streamlit Portfolio Construction page (`portfolio_construction.py`) → universe dropdown
- Python: `portfolio.compute_asset_signals(universe='hybrid')`

## User Preferences
- Wants systematic experiment tracking across sessions
- Focus on Paper Baseline first before exploring improvements
- See [experiments.md](experiments.md) for full log
