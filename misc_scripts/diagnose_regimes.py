"""
Diagnostic: Compare our JM regime dates against the paper's Figure 2.

Paper Figure 2 (JM-XGB forecasts, LargeCap 2007-2023):
- Bear%=20.9%, 46 regime shifts
- Bear periods: ~2008-09 to 2009-06 (GFC), short ~2010/2011,
  short ~2015-2016, ~2020-02 to 2020-04 (COVID), multiple short ~2022

This script:
1. Fits JM period-by-period (same as our pipeline) and extracts regime dates
2. Uses both our implementation and paper's jumpmodels library
3. Compares regime timing against paper's Figure 2
4. Tests across lambdas
5. Also generates the full JM-XGB forecast regimes (which is what Figure 2 shows)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['XGB_END_DATE'] = '2024-01-01'

import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel

import main as backend
from config import StrategyConfig

backend.END_DATE = '2024-01-01'
config = StrategyConfig()

df = backend.fetch_and_prepare_data()
oos_start = pd.to_datetime('2007-01-01')
oos_end = pd.to_datetime('2024-01-01')

return_features = [c for c in df.columns if c.startswith(('DD_', 'Avg_Ret_', 'Sortino_'))]


def extract_bear_periods(dates, states):
    """Extract contiguous bear (state=1) periods as date ranges."""
    periods = []
    in_bear = False
    start = None
    for i, (d, s) in enumerate(zip(dates, states)):
        if s == 1 and not in_bear:
            start = d
            in_bear = True
        elif s == 0 and in_bear:
            periods.append((start, dates[i-1]))
            in_bear = False
    if in_bear:
        periods.append((start, dates[-1]))
    return periods


def format_period(start, end):
    days = (end - start).days
    return f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} ({days}d)"


# Paper's known bear periods (from Figure 2 — these are JM-XGB, not JM-only)
paper_bear_periods = [
    ('2008-09-01', '2009-06-30', 'GFC main'),
    ('2010-01-01', '2010-12-31', 'Short episodes ~2010'),
    ('2011-01-01', '2011-12-31', 'Short episodes ~2011'),
    ('2015-06-01', '2016-06-30', 'Short episodes ~2015-2016'),
    ('2020-02-01', '2020-04-30', 'COVID'),
    ('2022-01-01', '2022-12-31', 'Multiple short ~2022'),
]


def get_jm_regimes_library(df, oos_start, oos_end, lmbda, method='online'):
    """Get OOS JM regime assignments using paper's library."""
    all_dates = []
    all_states = []
    current = pd.to_datetime(oos_start)
    end = pd.to_datetime(oos_end)

    while current < end:
        train_start = current - pd.DateOffset(years=11)
        train_df = df[(df.index >= train_start) & (df.index < current)].copy()
        chunk_end = min(current + pd.DateOffset(months=6), end)
        oos_chunk = df[(df.index >= current) & (df.index < chunk_end)].copy()

        if len(train_df) < 252*5 or len(oos_chunk) == 0:
            current = chunk_end
            continue

        X_train = train_df[return_features]
        m, s = X_train.mean(), X_train.std()
        X_train_std = (X_train - m) / s
        X_oos_std = (oos_chunk[return_features] - m) / s

        jm = JumpModel(n_components=2, jump_penalty=lmbda, random_state=42, n_init=10)
        jm.fit(X_train_std, ret_ser=train_df['Excess_Return'], sort_by='cumret')

        if method == 'online':
            labels = jm.predict_online(X_oos_std)
        else:
            labels = jm.predict(X_oos_std)

        if hasattr(labels, 'values'):
            labels = labels.values

        all_dates.extend(oos_chunk.index.tolist())
        all_states.extend(labels.astype(int).tolist())
        current = chunk_end

    return np.array(all_dates), np.array(all_states)


def get_jm_regimes_ours(df, oos_start, oos_end, lmbda):
    """Get OOS JM regime assignments using our implementation."""
    all_dates = []
    all_states = []
    current = pd.to_datetime(oos_start)
    end = pd.to_datetime(oos_end)

    while current < end:
        train_start = current - pd.DateOffset(years=11)
        train_df = df[(df.index >= train_start) & (df.index < current)].copy()
        chunk_end = min(current + pd.DateOffset(months=6), end)
        oos_chunk = df[(df.index >= current) & (df.index < chunk_end)].copy()

        if len(train_df) < 252*5 or len(oos_chunk) == 0:
            current = chunk_end
            continue

        X_train = train_df[return_features]
        m, s = X_train.mean(), X_train.std()
        X_train_std = ((X_train - m) / s).values
        X_oos_std = ((oos_chunk[return_features] - m) / s).values

        jm = backend.StatisticalJumpModel(n_states=2, lambda_penalty=lmbda)
        states = jm.fit_predict(X_train_std)

        # Align
        cum_ret_0 = train_df['Excess_Return'][states == 0].sum()
        cum_ret_1 = train_df['Excess_Return'][states == 1].sum()
        if cum_ret_1 > cum_ret_0:
            states = 1 - states
            jm.means = jm.means[::-1].copy()

        oos_states = jm.predict_online(X_oos_std, last_known_state=states[-1])

        all_dates.extend(oos_chunk.index.tolist())
        all_states.extend(oos_states.tolist())
        current = chunk_end

    return np.array(all_dates), np.array(all_states)


def get_jmxgb_regimes(df, oos_start, oos_end, lmbda, ewma_hl=8):
    """Get JM-XGB forecast regimes (what Figure 2 actually shows)."""
    backend._forecast_cache.clear()
    res = backend.simulate_strategy(df, oos_start, oos_end, lmbda, config,
                                     include_xgboost=True, ewma_halflife=ewma_hl)
    if res.empty:
        return np.array([]), np.array([])
    return res.index.values, res['Forecast_State'].values.astype(int)


def analyze_regimes(dates, states, label, paper_periods=None):
    """Print regime statistics and bear periods."""
    if len(dates) == 0:
        print(f"  {label}: No data")
        return

    bear_pct = states.mean()
    shifts = (np.diff(states) != 0).sum()
    periods = extract_bear_periods(dates, states)

    print(f"\n  {label}")
    print(f"  Bear%={bear_pct:.1%}, Shifts={shifts}, Bear periods={len(periods)}")
    print(f"  {'Bear Period':<45} {'Duration':>10}")
    print(f"  " + "-" * 57)

    for start, end in periods:
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        days = (end_dt - start_dt).days
        if days >= 5:  # Only show periods >= 5 days
            print(f"  {format_period(start_dt, end_dt):<45} {days:>8}d")

    # Compare with paper periods if provided
    if paper_periods:
        # Convert dates to pandas Timestamps for comparison
        dates_ts = pd.DatetimeIndex(dates)
        print(f"\n  Overlap with paper's Figure 2 bear periods:")
        for ps, pe, name in paper_periods:
            ps_dt = pd.Timestamp(ps)
            pe_dt = pd.Timestamp(pe)
            mask = (dates_ts >= ps_dt) & (dates_ts <= pe_dt)
            if mask.sum() > 0:
                our_bear_in_period = states[mask].mean()
                total_days = mask.sum()
                print(f"    {name:<30} Paper bear | Ours: {our_bear_in_period:.0%} bear "
                      f"({int(our_bear_in_period*total_days)}/{total_days}d)")

        # Also check for false bears (our bear outside paper periods)
        in_paper_bear = np.zeros(len(dates), dtype=bool)
        for ps, pe, _ in paper_periods:
            mask = (dates_ts >= pd.Timestamp(ps)) & (dates_ts <= pd.Timestamp(pe))
            in_paper_bear |= mask

        our_bear = states == 1
        false_bear = our_bear & ~in_paper_bear
        missed_bear = ~our_bear & in_paper_bear
        print(f"\n    False bear days (ours=bear, paper=bull): {false_bear.sum()} "
              f"({false_bear.sum()/len(dates):.1%})")
        print(f"    Missed bear days (ours=bull, paper=bear): {missed_bear.sum()} "
              f"({missed_bear.sum()/len(dates):.1%})")


# ==========================================================================
print("=" * 80)
print("REGIME DATE COMPARISON: Our JM vs Paper's Figure 2")
print("Paper Figure 2: LargeCap, JM-XGB forecasts (2007-2023)")
print("Bear%=20.9%, 46 shifts")
print("=" * 80)

# ==========================================================================
print("\n" + "=" * 80)
print("TEST 1: Our JM implementation (greedy predict_online)")
print("=" * 80)

for lmbda in [21.54, 46.42, 80.0]:
    dates, states = get_jm_regimes_ours(df, oos_start, oos_end, lmbda)
    analyze_regimes(dates, states, f"Our JM, λ={lmbda:.1f}", paper_bear_periods)

# ==========================================================================
print("\n" + "=" * 80)
print("TEST 2: Paper's library (predict_online = forward Viterbi)")
print("=" * 80)

for lmbda in [21.54, 46.42, 65.0, 80.0]:
    dates, states = get_jm_regimes_library(df, oos_start, oos_end, lmbda, 'online')
    analyze_regimes(dates, states, f"Library online, λ={lmbda:.1f}", paper_bear_periods)

# ==========================================================================
print("\n" + "=" * 80)
print("TEST 3: JM-XGB forecast regimes (what Figure 2 actually shows)")
print("=" * 80)

# Figure 2 is JM-XGB forecasts. Run with different lambdas.
for lmbda in [4.64, 21.54, 46.42, 80.0]:
    backend._forecast_cache.clear()
    dates, states = get_jmxgb_regimes(df, oos_start, oos_end, lmbda, ewma_hl=8)
    analyze_regimes(dates, states, f"JM-XGB, λ={lmbda:.1f}, HL=8", paper_bear_periods)

# ==========================================================================
print("\n" + "=" * 80)
print("TEST 4: Walk-forward JM-XGB (our actual pipeline)")
print("=" * 80)

backend._forecast_cache.clear()
result_df = backend.walk_forward_backtest(df, config)
if not result_df.empty and 'Forecast_State' in result_df.columns:
    dates = result_df.index.values
    states = result_df['Forecast_State'].values.astype(int)
    lambdas = result_df.attrs.get('lambda_history', [])
    print(f"  Walk-forward lambdas: {[f'{l:.1f}' for l in lambdas[:10]]}...")
    analyze_regimes(dates, states, "Walk-Forward JM-XGB", paper_bear_periods)

# ==========================================================================
print("\n" + "=" * 80)
print("TEST 5: In-sample JM regime quality (training window at 2007-01-01)")
print("=" * 80)

# Show what the JM finds in the 11-year training window ending at 2007
train_start = oos_start - pd.DateOffset(years=11)
train_df = df[(df.index >= train_start) & (df.index < oos_start)].copy()
X_train = train_df[return_features]
m, s = X_train.mean(), X_train.std()
X_train_std = (X_train - m) / s

for lmbda in [21.54, 46.42, 80.0]:
    jm = JumpModel(n_components=2, jump_penalty=lmbda, random_state=42, n_init=10)
    jm.fit(X_train_std, ret_ser=train_df['Excess_Return'], sort_by='cumret')

    labels = jm.labels_
    if hasattr(labels, 'values'):
        labels = labels.values
    labels = np.array(labels, dtype=int)

    bear_pct = (labels == 1).mean()
    shifts = (np.diff(labels) != 0).sum()

    # Average return per regime
    bull_ret = train_df['Excess_Return'][labels == 0].mean() * 252
    bear_ret = train_df['Excess_Return'][labels == 1].mean() * 252

    print(f"\n  In-sample λ={lmbda:.1f}: Bear%={bear_pct:.1%}, Shifts={shifts}")
    print(f"    Bull regime: ann. excess ret={bull_ret:+.1%}")
    print(f"    Bear regime: ann. excess ret={bear_ret:+.1%}")

    # Show known crisis periods in training data
    known_train_crises = [
        ('1998-08-01', '1998-10-31', 'LTCM/Russia'),
        ('2000-03-01', '2002-10-31', 'Dot-com crash'),
    ]
    for ps, pe, name in known_train_crises:
        mask = (train_df.index >= ps) & (train_df.index <= pe)
        if mask.sum() > 0:
            mask_arr = mask.values if hasattr(mask, 'values') else mask
            bear_in_crisis = (labels[mask_arr] == 1).mean()
            print(f"    {name}: {bear_in_crisis:.0%} bear ({int(bear_in_crisis*mask.sum())}/{mask.sum()}d)")

    # Show cluster centers distance
    center_dist = np.sqrt(np.sum((jm.centers_[0] - jm.centers_[1])**2))
    print(f"    Cluster center distance: {center_dist:.3f}")

# ==========================================================================
print("\n" + "=" * 80)
print("TEST 6: Year-by-year bear% comparison")
print("=" * 80)

# Get library online regimes at λ=65 (closest to paper's stats)
dates_lib, states_lib = get_jm_regimes_library(df, oos_start, oos_end, 65.0, 'online')
dates_ours, states_ours = get_jm_regimes_ours(df, oos_start, oos_end, 46.42)

backend._forecast_cache.clear()
dates_xgb, states_xgb = get_jmxgb_regimes(df, oos_start, oos_end, 21.54, ewma_hl=8)

print(f"\n  {'Year':<6} {'Lib JM λ=65':>13} {'Our JM λ=46':>13} {'XGB λ=21':>13} {'S&P500 ret':>12}")
print(f"  " + "-" * 62)

for year in range(2007, 2024):
    yr_start = f'{year}-01-01'
    yr_end = f'{year}-12-31'

    oos_sub = df[(df.index >= yr_start) & (df.index <= yr_end)]
    yr_ret = (1 + oos_sub['Target_Return']).prod() - 1 if len(oos_sub) > 0 else 0

    def year_bear(dates, states, yr_start, yr_end):
        dates_ts = pd.DatetimeIndex(dates)
        mask = (dates_ts >= pd.Timestamp(yr_start)) & (dates_ts <= pd.Timestamp(yr_end))
        return states[mask].mean() if mask.sum() > 0 else float('nan')

    lib_bear = year_bear(dates_lib, states_lib, yr_start, yr_end)
    our_bear = year_bear(dates_ours, states_ours, yr_start, yr_end)
    xgb_bear = year_bear(dates_xgb, states_xgb, yr_start, yr_end)

    print(f"  {year:<6} {lib_bear:13.1%} {our_bear:13.1%} {xgb_bear:13.1%} {yr_ret:+12.1%}")


print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
Paper Figure 2 (JM-XGB): Bear%=20.9%, 46 shifts
Key bear periods: GFC (~2008-09 to 2009-06), COVID (~2020-02 to 2020-04)
                  Short episodes in 2010, 2011, 2015-2016, 2022

Compare our regimes against these known periods to identify:
- False bears: we say bear but paper/market says bull
- Missed bears: we say bull but paper/market says bear
- Timing differences: we detect crisis too late or exit too early
""")
