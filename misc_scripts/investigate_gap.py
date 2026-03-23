#!/usr/bin/env python3
"""
Focused gap investigation: why can't we match paper's 0.79 Sharpe?

Runs 3 prioritized hypothesis groups in smart order (reuses forecast cache where possible):

  E  Fixed-lambda XGB sweep — isolates: does XGB add value at any specific λ?
     (cheapest: uses paper_config cache from prior runs)
  D  predict_online with last_known_state init — tests period-boundary bias
     (cheap: only needs JM recompute, not XGB)
  A  XGBoost tree_method='exact' — tests if paper used XGB 1.x exact method
     (expensive: new xgb_key = fresh cache; run last)

Usage:
  python misc_scripts/investigate_gap.py        # all
  python misc_scripts/investigate_gap.py E      # just fixed-lambda sweep
  python misc_scripts/investigate_gap.py E D    # two groups
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

os.environ['XGB_END_DATE'] = '2023-12-31'
os.environ['XGB_OOS_START_DATE'] = '2007-01-01'
os.environ['XGB_START_DATE_DATA'] = '1987-01-01'

import main
from config import StrategyConfig

main.END_DATE = '2023-12-31'
main.OOS_START_DATE = '2007-01-01'

PAPER_SHARPE = 0.790
PAPER_MDD    = -0.1769
PAPER_BEAR   = 20.9
PAPER_SHIFTS = 46

PAPER_CONFIG = StrategyConfig(
    name="Paper_BBG",
    ewma_mode="paper",
    tuning_metric="sharpe",
    lambda_selection="best",
    lambda_subwindow_consensus=False,
    allocation_style="binary",
    prob_threshold=0.50,
)
GRID_8PT = [4.64, 10.0, 15.0, 21.54, 30.0, 46.42, 70.0, 100.0]
GRID_19PT = list(np.logspace(0, 2, 19))

# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

def load_bbg():
    import yfinance as yf
    xlsx = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'cache', 'DATA PAUL.xlsx')
    print("Loading Bloomberg data...")
    raw = pd.read_excel(xlsx, header=None, skiprows=6)
    raw.columns = ['Date','SPTR','SPTRMDCP','RU20INTR','NDDUEAFE','NDUEEGF',
                   'LBUSTRUU','IBOXHY','LUACTRUU','DJUSRET','DBLCDBCE','GOLDLNPM','LUTLTRUU']
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw = raw.set_index('Date').sort_index()[['SPTR','LBUSTRUU']].dropna()

    fred = main._fetch_fred_data().ffill().dropna()
    vix = yf.download('^VIX', start='1987-01-01', end='2024-01-01',
                      auto_adjust=False, progress=False)
    irx = yf.download('^IRX', start='1987-01-01', end='2024-01-01',
                      auto_adjust=False, progress=False)

    def _s(d):
        return (d['Adj Close'].iloc[:,0] if isinstance(d.columns, pd.MultiIndex)
                else d.get('Adj Close', d['Close']))

    df = raw.join(fred, how='inner').join(_s(vix).rename('VIX'), how='inner')\
            .join(_s(irx).rename('IRX'), how='inner').ffill().dropna()

    f = pd.DataFrame(index=df.index)
    tr = df['SPTR'].pct_change().fillna(0)
    f['Target_Return'] = tr
    f['Target_Intraday_Ret'] = tr
    f['Target_Overnight_Ret'] = 0.0
    f['RF_Rate'] = (df['IRX'] / 100) / 252
    f['Excess_Return'] = tr - f['RF_Rate']
    dn = np.minimum(f['Excess_Return'], 0)
    for hl in [5, 21]:
        ewm_dd = np.sqrt((dn**2).ewm(halflife=hl).mean()).fillna(0)
        f[f'DD_log_{hl}'] = np.log(ewm_dd + 1e-8)
    for hl in [5, 10, 21]:
        f[f'Avg_Ret_{hl}'] = f['Excess_Return'].ewm(halflife=hl).mean()
    for hl in [5, 10, 21]:
        dd_r = np.maximum(np.sqrt((dn**2).ewm(halflife=hl).mean()).fillna(1e-8), 1e-8)
        f[f'Sortino_{hl}'] = (f[f'Avg_Ret_{hl}'] / dd_r).clip(-10, 10)
    f['Yield_2Y_EWMA_diff'] = df['DGS2'].diff().fillna(0).ewm(halflife=21).mean()
    sl = df['DGS10'] - df['DGS2']
    f['Yield_Slope_EWMA_10'] = sl.ewm(halflife=10).mean()
    f['Yield_Slope_EWMA_diff_21'] = sl.diff().fillna(0).ewm(halflife=21).mean()
    f['VIX_EWMA_log_diff'] = np.log(df['VIX']/df['VIX'].shift(1)).fillna(0).ewm(halflife=63).mean()
    f['Stock_Bond_Corr'] = (tr.rolling(252).corr(df['LBUSTRUU'].pct_change().fillna(0)).fillna(0))
    result = f.dropna()
    print(f"  {result.index.min().date()} → {result.index.max().date()}, {len(result)} rows")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def oos(r):
    return r[(r.index >= '2007-01-01') & (r.index <= '2023-12-31')]

def mets(r):
    o = oos(r)
    _, _, s, _, m = main.calculate_metrics(o['Strat_Return'], o['RF_Rate'])
    return s, m

def reg_stats(r):
    o = oos(r)
    if 'Forecast_State' not in o.columns:
        return None, None
    st = o['Forecast_State']
    return (st==1).mean()*100, max(int((st != st.shift(1)).sum())-1, 0)

def pr(label, r):
    if r is None or (hasattr(r, 'empty') and r.empty):
        print(f"  {label:<55} ERROR/EMPTY")
        return
    s, m = mets(r)
    bp, ns = reg_stats(r)
    lh = np.array(r.attrs.get('lambda_history', [0]))
    print(f"  {label:<55} S={s:.3f}({s-PAPER_SHARPE:+.3f}) "
          f"MDD={m:.1%} Bear={bp:.1f}% Shft={ns} λ̄={lh.mean():.1f}")


# ──────────────────────────────────────────────────────────────────────────────
# E — Fixed-lambda sweep (cheapest: reuses cache if populated from prior run)
# ──────────────────────────────────────────────────────────────────────────────

def test_E(df):
    print("\n" + "="*70)
    print("E: Fixed-lambda XGB sweep — how much does XGB actually add at each λ?")
    print("   Paper: JM-only≈0.59, XGB adds +0.20. We expect oracle λ → near 0.79.")
    print("="*70)
    extended_grid = [4.64, 10.0, 15.0, 21.54, 30.0, 46.42, 70.0, 100.0]

    print(f"\n  {'λ':<8} {'JM S':>7} {'XGB S':>7} {'XGB add':>8} {'Bear%':>7} {'Shft':>6}")
    print(f"  {'─'*8} {'─'*7} {'─'*7} {'─'*8} {'─'*7} {'─'*6}")
    best_xgb = (-np.inf, None, None)
    for lm in extended_grid:
        r_jm  = main.simulate_strategy(df,'2007-01-01','2023-12-31', lm,
                                       PAPER_CONFIG, include_xgboost=False)
        r_xgb = main.simulate_strategy(df,'2007-01-01','2023-12-31', lm,
                                       PAPER_CONFIG, include_xgboost=True, ewma_halflife=8)
        if r_jm.empty or r_xgb.empty:
            continue
        sj, _ = mets(r_jm)
        sx, mx = mets(r_xgb)
        bp, ns = reg_stats(r_xgb)
        print(f"  {lm:<8.2f} {sj:>7.3f} {sx:>7.3f} {sx-sj:>+8.3f} {bp:>7.1f}% {ns:>6}")
        if sx > best_xgb[0]:
            best_xgb = (sx, lm, mx)
    print(f"\n  → Best fixed-λ XGB: λ={best_xgb[1]}, Sharpe={best_xgb[0]:.3f} "
          f"MDD={best_xgb[2]:.1%}  (paper: 0.790 / -17.69%)")

    # Extra: widen lambda range to see if outside 8pt grid does better
    print("\n  [Extended scan λ=30-100 in finer steps]")
    extra = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    best2 = (-np.inf, None, None)
    for lm in extra:
        r_xgb = main.simulate_strategy(df,'2007-01-01','2023-12-31', float(lm),
                                       PAPER_CONFIG, include_xgboost=True, ewma_halflife=8)
        if r_xgb.empty:
            continue
        sx, mx = mets(r_xgb)
        bp, ns = reg_stats(r_xgb)
        print(f"  {lm:<8.2f} {'—':>7} {sx:>7.3f} {'—':>8} {bp:>7.1f}% {ns:>6}")
        if sx > best2[0]:
            best2 = (sx, lm, mx)
    print(f"\n  → Best in extended scan: λ={best2[1]}, Sharpe={best2[0]:.3f} "
          f"MDD={best2[2]:.1%}")


# ──────────────────────────────────────────────────────────────────────────────
# D — predict_online last_known_state initialization
# ──────────────────────────────────────────────────────────────────────────────

def test_D(df):
    print("\n" + "="*70)
    print("D: predict_online WITH last_known_state DP initialization")
    print("   Current: DP starts from loss_mx[0] (no bias). Variant: adds λ penalty")
    print("   for transitioning away from last training state at period start.")
    print("="*70)

    orig = main.StatisticalJumpModel.predict_online

    def predict_with_lks(self, X, last_known_state=None):
        Xa = np.array(X)
        n = Xa.shape[0]
        loss = 0.5 * np.sum((Xa[:,None,:] - self.means[None,:,:])**2, axis=2)
        pen  = self.lambda_penalty * (1 - np.eye(self.n_states))
        vals = np.empty((n, self.n_states))
        vals[0] = loss[0].copy()
        if last_known_state is not None:
            for k in range(self.n_states):
                if k != last_known_state:
                    vals[0, k] += self.lambda_penalty
        for t in range(1, n):
            vals[t] = loss[t] + (vals[t-1][:,np.newaxis] + pen).min(axis=0)
        return vals.argmin(axis=1)

    main.StatisticalJumpModel.predict_online = predict_with_lks
    main._forecast_cache.clear()

    cfg8  = StrategyConfig(name="D_lks_8pt",  ewma_mode="paper")
    cfg19 = StrategyConfig(name="D_lks_19pt", ewma_mode="paper")
    main.LAMBDA_GRID = GRID_8PT
    r8  = main.walk_forward_backtest(df, cfg8)
    main.LAMBDA_GRID = GRID_19PT
    r19 = main.walk_forward_backtest(df, cfg19)

    main.StatisticalJumpModel.predict_online = orig
    main._forecast_cache.clear()
    print("  [predict_online restored]")

    pr("8pt  / WITH last_known_state init", r8)
    pr("19pt / WITH last_known_state init", r19)

    # Also check: how many period boundaries differ vs without init?
    print("\n  [Checking state at first OOS day for period boundaries]")
    print("  (Tests whether initialization changes regime predictions at chunk starts)")
    sample_dates = [pd.Timestamp('2007-01-01'), pd.Timestamp('2010-01-01'),
                    pd.Timestamp('2013-07-01'), pd.Timestamp('2017-01-01')]
    for d in sample_dates:
        td = df[(df.index >= d - pd.DateOffset(days=3)) &
                (df.index <= d + pd.DateOffset(days=3))]
        if len(td) == 0:
            continue
        first_d = td.index[0]
        # Check Forecast_State at that date in r8 vs original
        if r8 is not None and not r8.empty and first_d in r8.index:
            print(f"  {d.date()}: D(with_lks)={int(r8.loc[first_d,'Forecast_State']) if 'Forecast_State' in r8.columns else '?'}")


# ──────────────────────────────────────────────────────────────────────────────
# A — XGBoost tree_method='exact' (paper likely used XGB 1.x)
# ──────────────────────────────────────────────────────────────────────────────

def test_A(df):
    print("\n" + "="*70)
    print("A: XGBoost tree_method='exact'")
    print("   XGBoost 2.0 changed default from 'exact' to 'hist'. Paper likely used 1.x.")
    print(f"   Current XGBoost version: {__import__('xgboost').__version__}")
    print("="*70)

    # Note: each new xgb_params = fresh cache. Run 8pt first (fewer lambdas cached).
    configs = [
        ("default (hist)",       {},                                        GRID_8PT),
        ("tree_method=exact",    {"tree_method": "exact"},                  GRID_8PT),
        ("exact + 19pt",         {"tree_method": "exact"},                  GRID_19PT),
        ("exact + n_est=200",    {"tree_method": "exact","n_estimators":200}, GRID_8PT),
        ("exact + n_est=500",    {"tree_method": "exact","n_estimators":500}, GRID_8PT),
        ("default + n_est=200",  {"n_estimators": 200},                    GRID_8PT),
        ("max_depth=4",          {"max_depth": 4},                          GRID_8PT),
        ("max_depth=4 + exact",  {"tree_method":"exact","max_depth":4},     GRID_8PT),
    ]
    for label, xp, grid in configs:
        safe_name = label.replace(" ", "_").replace("=","").replace("+","_")
        cfg = StrategyConfig(name=f"A_{safe_name}", ewma_mode="paper", xgb_params=xp)
        main.LAMBDA_GRID = grid
        main._forecast_cache.clear()
        r = main.walk_forward_backtest(df, cfg)
        pr(f"{label} / {len(grid)}pt grid", r)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = [a.upper() for a in sys.argv[1:]] if len(sys.argv) > 1 else ['E','D','A']
    df = load_bbg()
    main._forecast_cache.clear()

    print(f"\nPaper targets: S={PAPER_SHARPE}  MDD={PAPER_MDD:.2%}  "
          f"Bear={PAPER_BEAR}%  Shifts={PAPER_SHIFTS}")
    print("\n── Reference ──")
    main.LAMBDA_GRID = GRID_8PT
    cfg_ref8  = StrategyConfig(name="Ref_8pt",  ewma_mode="paper")
    cfg_ref19 = StrategyConfig(name="Ref_19pt", ewma_mode="paper")
    r8  = main.walk_forward_backtest(df, cfg_ref8)
    main.LAMBDA_GRID = GRID_19PT
    r19 = main.walk_forward_backtest(df, cfg_ref19)
    pr("8pt dense  [4.64-100]", r8)
    pr("Log 19pt   [1-100]",    r19)

    if 'E' in args:
        test_E(df)
    if 'D' in args:
        test_D(df)
    if 'A' in args:
        test_A(df)

    print("\n" + "="*70)
    print("DONE")
    print("="*70)
