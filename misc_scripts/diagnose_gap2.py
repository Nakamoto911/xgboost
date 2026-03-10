"""
Diagnostic Part 2: Why does walk-forward lambda tuning pick suboptimal lambdas?
Lean version — focuses on alternative grid/window tests via walk_forward_backtest().
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['XGB_END_DATE'] = '2024-01-01'

import numpy as np
import pandas as pd

import main as backend
from config import StrategyConfig

backend.END_DATE = '2024-01-01'

config = StrategyConfig()
df = backend.fetch_and_prepare_data()

oos_df = df[(df.index >= '2007-01-01') & (df.index < '2024-01-01')]
_, _, bh_sharpe, _, _ = backend.calculate_metrics(oos_df['Target_Return'], oos_df['RF_Rate'])
print(f"B&H Sharpe: {bh_sharpe:.3f}")

def run_wf(label, grid=None, val_yrs=None, ewma_hl=None):
    """Run walk-forward with overrides, return Sharpe + lambda stats."""
    orig_grid = backend.LAMBDA_GRID
    orig_val = backend.VALIDATION_WINDOW_YRS
    orig_hl = backend.PAPER_EWMA_HL.get('^SP500TR')

    if grid is not None:
        backend.LAMBDA_GRID = grid
    if val_yrs is not None:
        backend.VALIDATION_WINDOW_YRS = val_yrs
    if ewma_hl is not None:
        backend.PAPER_EWMA_HL['^SP500TR'] = ewma_hl

    backend._forecast_cache.clear()
    res = backend.walk_forward_backtest(df, config)

    backend.LAMBDA_GRID = orig_grid
    backend.VALIDATION_WINDOW_YRS = orig_val
    if orig_hl is not None:
        backend.PAPER_EWMA_HL['^SP500TR'] = orig_hl

    if res.empty:
        return

    _, _, sharpe, _, mdd = backend.calculate_metrics(res['Strat_Return'], res['RF_Rate'])
    lambdas = res.attrs.get('lambda_history', [])
    mean_l = np.mean(lambdas) if lambdas else 0
    std_l = np.std(lambdas) if lambdas else 0
    cv_l = std_l / mean_l if mean_l > 0 else 0
    hl = res.attrs.get('ewma_halflife', '?')
    print(f"  {label:<40} Sharpe={sharpe:.3f}  Δ={sharpe-bh_sharpe:+.3f}  MDD={mdd:.1%}  λ_mean={mean_l:5.1f} λ_std={std_l:5.1f} CV={cv_l:.2f} HL={hl}")

# ==========================================================================
print("\n" + "=" * 80)
print("TEST A: Lambda grid alternatives")
print("=" * 80)

run_wf("Current (11 pts, 0+logspace 1-100)")
run_wf("Paper-like (21 pts, 0+logspace 1-100)", grid=[0.0] + list(np.logspace(0, 2, 20)))
run_wf("Focused [4.6, 10, 21.5, 46.4]", grid=[4.64, 10.0, 21.54, 46.42])
run_wf("Narrow [10, 21.5]", grid=[10.0, 21.54])
run_wf("Fixed λ=21.54", grid=[21.54])
run_wf("Fixed λ=10.0", grid=[10.0])
run_wf("Fixed λ=4.64", grid=[4.64])
run_wf("Low range [0, 1..30, 10pts]", grid=[0.0] + list(np.logspace(0, np.log10(30), 10)))
run_wf("Mid range [5..50, 10pts]", grid=list(np.logspace(np.log10(5), np.log10(50), 10)))

# ==========================================================================
print("\n" + "=" * 80)
print("TEST B: Validation window length")
print("=" * 80)

for val_yrs in [3, 4, 5, 7]:
    run_wf(f"Val={val_yrs}yr", val_yrs=val_yrs)

# ==========================================================================
print("\n" + "=" * 80)
print("TEST C: EWMA halflife with walk-forward")
print("=" * 80)

for hl in [0, 2, 4, 8, 12]:
    run_wf(f"HL={hl}", ewma_hl=hl)

# ==========================================================================
print("\n" + "=" * 80)
print("TEST D: Combined best settings")
print("=" * 80)

# Try combinations of the best from each test
run_wf("Fixed λ=21.54 + HL=8", grid=[21.54], ewma_hl=8)
run_wf("Narrow [10,21.5] + Val=3yr", grid=[10.0, 21.54], val_yrs=3)
run_wf("Narrow [10,21.5] + Val=7yr", grid=[10.0, 21.54], val_yrs=7)
run_wf("Mid [5-50] + Val=3yr", grid=list(np.logspace(np.log10(5), np.log10(50), 10)), val_yrs=3)

print(f"\nReference: B&H Sharpe={bh_sharpe:.3f}, Paper Sharpe=0.79")
