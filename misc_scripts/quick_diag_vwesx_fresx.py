"""Quick diagnostic: VWESX and FRESX with custom lambda grids and EWMA halflives."""
import sys, os, types

try:
    import distutils, distutils.version
except ImportError:
    d = types.ModuleType('distutils')
    dv = types.ModuleType('distutils.version')
    class LooseVersion:
        def __init__(self, vstring=None): self.vstring = vstring
        def __str__(self): return self.vstring
        def __lt__(self, other): return self.vstring < (other.vstring if hasattr(other, 'vstring') else other)
        def __le__(self, other): return self.vstring <= (other.vstring if hasattr(other, 'vstring') else other)
        def __gt__(self, other): return self.vstring > (other.vstring if hasattr(other, 'vstring') else other)
        def __ge__(self, other): return self.vstring >= (other.vstring if hasattr(other, 'vstring') else other)
        def __eq__(self, other): return self.vstring == (other.vstring if hasattr(other, 'vstring') else other)
    dv.LooseVersion = LooseVersion
    d.version = dv
    sys.modules['distutils'] = d
    sys.modules['distutils.version'] = dv

import numpy as np
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'misc_scripts'))

from benchmark_assets import fetch_etf_data, PAPER_EWMA_HL
from diagnose_multi_asset import run_single_asset_backtest

DATA_START = '1975-01-01'

def run_test(label, ticker, df, oos_start, oos_end, lambda_grid, ewma_hl=None):
    print(f"\n  {label}")
    hl_desc = f"hl={ewma_hl}" if ewma_hl is not None else f"hl=paper({PAPER_EWMA_HL.get(ticker, 'default')})"
    print(f"    Ticker={ticker}, {hl_desc}, grid={lambda_grid}")
    sm, bm, lh = run_single_asset_backtest(ticker, df, oos_start, oos_end, lambda_grid=lambda_grid, ewma_hl=ewma_hl)
    if sm and bm:
        delta = sm[2] - bm[2]
        v = 'WIN' if delta > 0 else 'LOSE'
        print(f"    Strategy Sharpe: {sm[2]:.3f}")
        print(f"    B&H Sharpe:      {bm[2]:.3f}")
        print(f"    Delta:           {delta:+.3f} ({v})")
        print(f"    Lambda picks:    {[round(l,2) for l in lh]}")
        return sm[2], bm[2], delta, lh
    else:
        print(f"    FAILED - no results")
        return None

def main():
    print("=" * 80)
    print("  QUICK DIAGNOSTIC: VWESX and FRESX")
    print("=" * 80)

    # Fetch data
    print("\nFetching VWESX data...")
    _, df_vwesx = fetch_etf_data('VWESX', data_start=DATA_START)
    print(f"  VWESX: {len(df_vwesx)} rows ({df_vwesx.index[0].date()} to {df_vwesx.index[-1].date()})")

    print("Fetching FRESX data...")
    _, df_fresx = fetch_etf_data('FRESX', data_start=DATA_START)
    print(f"  FRESX: {len(df_fresx)} rows ({df_fresx.index[0].date()} to {df_fresx.index[-1].date()})")

    # ── VWESX Tests ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  VWESX (Corporate Bond) Tests")
    print("=" * 80)

    wide_grid = [4.64, 10.0, 15.0, 21.54, 30.0, 46.42, 70.0, 100.0]
    tight_grid = [10.0, 15.0, 21.54, 30.0]

    # Test 1: VWESX with hl=0, wide grid
    run_test("Test 1: VWESX hl=0, wide grid", 'VWESX', df_vwesx,
             '2007-01-01', '2025-01-01', lambda_grid=wide_grid, ewma_hl=0)

    # Test 2: VWESX with paper hl=2, tight grid
    run_test("Test 2: VWESX hl=2 (paper), tight grid", 'VWESX', df_vwesx,
             '2007-01-01', '2025-01-01', lambda_grid=tight_grid, ewma_hl=2)

    # ── FRESX Tests ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  FRESX (REIT) Tests")
    print("=" * 80)

    current_grid = [4.64, 10.0, 15.0, 21.54, 30.0, 46.42, 70.0, 100.0]
    high_grid = [30.0, 46.42, 70.0, 100.0]

    # Test 3: FRESX with current grid
    run_test("Test 3: FRESX current wide grid", 'FRESX', df_fresx,
             '2007-01-01', '2025-01-01', lambda_grid=current_grid)

    # Test 4: FRESX with high-only grid
    run_test("Test 4: FRESX high-only grid", 'FRESX', df_fresx,
             '2007-01-01', '2025-01-01', lambda_grid=high_grid)

    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
