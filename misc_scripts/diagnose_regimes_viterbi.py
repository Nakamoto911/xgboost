"""
Diagnostic: Use full Viterbi (fit_predict / predict) on data to extract
bear regime dates and compare against paper's Figure 2 timeline.

Paper Figure 2 (LargeCap, 2007-2023):
  Bear%=20.9%, 46 shifts
  ~2008-09 to 2009-06 (GFC)
  Several short episodes ~2010, ~2011
  Short episodes ~2015-2016
  ~2020-02 to 2020-04 (COVID)
  Multiple short episodes ~2022

Approaches tested:
A) fit on 11yr train, predict() (full Viterbi) on 6mo OOS — per-chunk
B) fit on 11yr train, predict() on train+OOS, extract OOS portion — per-chunk
C) Single fit on entire 1996-2023, extract 2007-2023 portion
D) predict_online (forward-only Viterbi) on OOS — per-chunk (for comparison)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['XGB_END_DATE'] = '2024-01-01'

import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel

import main as backend
backend.END_DATE = '2024-01-01'

df = backend.fetch_and_prepare_data()
oos_start = pd.to_datetime('2007-01-01')
oos_end = pd.to_datetime('2024-01-01')

return_features = [c for c in df.columns if c.startswith(('DD_', 'Avg_Ret_', 'Sortino_'))]


def extract_bear_periods(dates, states, min_days=3):
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
    return [(s, e) for s, e in periods if (pd.Timestamp(e) - pd.Timestamp(s)).days >= min_days]


def print_regimes(dates, states, label):
    dates_ts = pd.DatetimeIndex(dates)
    bear_pct = states.mean()
    shifts = (np.diff(states) != 0).sum()
    periods = extract_bear_periods(dates, states)

    print(f"\n  {label}")
    print(f"  Bear%={bear_pct:.1%}, Shifts={shifts}, Bear periods={len(periods)}")
    print(f"  {'#':<4} {'Start':<12} {'End':<12} {'Days':>6}  Period context")
    print(f"  " + "-" * 65)

    for i, (s, e) in enumerate(periods):
        s_dt, e_dt = pd.Timestamp(s), pd.Timestamp(e)
        days = (e_dt - s_dt).days

        # Identify context
        ctx = ""
        if s_dt >= pd.Timestamp('2007-07-01') and e_dt <= pd.Timestamp('2009-07-01'):
            ctx = "GFC"
        elif s_dt >= pd.Timestamp('2009-07-01') and e_dt <= pd.Timestamp('2010-12-31'):
            ctx = "~2010"
        elif s_dt >= pd.Timestamp('2011-01-01') and e_dt <= pd.Timestamp('2012-01-01'):
            ctx = "~2011"
        elif s_dt >= pd.Timestamp('2014-06-01') and e_dt <= pd.Timestamp('2016-07-01'):
            ctx = "~2015-16"
        elif s_dt >= pd.Timestamp('2020-01-01') and e_dt <= pd.Timestamp('2020-06-30'):
            ctx = "COVID"
        elif s_dt >= pd.Timestamp('2021-06-01') and e_dt <= pd.Timestamp('2023-01-01'):
            ctx = "~2022"
        elif s_dt >= pd.Timestamp('2018-01-01') and e_dt <= pd.Timestamp('2019-06-30'):
            ctx = "~2018-19"

        print(f"  {i+1:<4} {s_dt.strftime('%Y-%m-%d'):<12} {e_dt.strftime('%Y-%m-%d'):<12} {days:>6}d  {ctx}")

    return bear_pct, shifts, len(periods)


# ==========================================================================
# Paper reference
# ==========================================================================
print("=" * 80)
print("PAPER FIGURE 2 REFERENCE (LargeCap, JM-XGB, 2007-2023)")
print("=" * 80)
print("""
  Bear%=20.9%, 46 regime shifts
  Bear periods (approximate from figure reading):
    ~2008-09 to 2009-06  (GFC main, ~10 months)
    Several short episodes ~2010, ~2011
    Short episodes ~2015-2016
    ~2020-02 to 2020-04  (COVID, ~2 months)
    Multiple short episodes ~2022
""")


# ==========================================================================
# Approach A: fit on train, predict() on OOS chunk (full Viterbi per chunk)
# ==========================================================================
print("=" * 80)
print("APPROACH A: fit(train 11yr) → predict(OOS 6mo) per chunk")
print("  Full Viterbi on each 6-month OOS window independently")
print("=" * 80)

for lmbda in [30.0, 46.42, 65.0, 80.0]:
    all_dates, all_states = [], []
    current = oos_start
    while current < oos_end:
        train_start = current - pd.DateOffset(years=11)
        train_df = df[(df.index >= train_start) & (df.index < current)].copy()
        chunk_end = min(current + pd.DateOffset(months=6), oos_end)
        oos_chunk = df[(df.index >= current) & (df.index < chunk_end)].copy()
        if len(train_df) >= 252*5 and len(oos_chunk) > 0:
            X_train = train_df[return_features]
            m, s = X_train.mean(), X_train.std()
            jm = JumpModel(n_components=2, jump_penalty=lmbda, random_state=42, n_init=10)
            jm.fit((X_train - m) / s, ret_ser=train_df['Excess_Return'], sort_by='cumret')
            labels = jm.predict((oos_chunk[return_features] - m) / s)
            if hasattr(labels, 'values'):
                labels = labels.values
            all_dates.extend(oos_chunk.index.tolist())
            all_states.extend(labels.astype(int).tolist())
        current = chunk_end
    print_regimes(np.array(all_dates), np.array(all_states), f"A: predict(OOS), λ={lmbda}")


# ==========================================================================
# Approach B: fit on train, predict() on train+OOS, extract OOS states
# ==========================================================================
print("\n" + "=" * 80)
print("APPROACH B: fit(train 11yr) → predict(train+OOS) → extract OOS portion")
print("  Full Viterbi on 11yr+6mo window, take last 6 months")
print("=" * 80)

for lmbda in [30.0, 46.42, 65.0, 80.0]:
    all_dates, all_states = [], []
    current = oos_start
    while current < oos_end:
        train_start = current - pd.DateOffset(years=11)
        train_df = df[(df.index >= train_start) & (df.index < current)].copy()
        chunk_end = min(current + pd.DateOffset(months=6), oos_end)
        oos_chunk = df[(df.index >= current) & (df.index < chunk_end)].copy()
        if len(train_df) >= 252*5 and len(oos_chunk) > 0:
            X_train = train_df[return_features]
            m, s = X_train.mean(), X_train.std()
            # Combine train+OOS for prediction
            combined = pd.concat([train_df, oos_chunk])
            X_combined = (combined[return_features] - m) / s

            jm = JumpModel(n_components=2, jump_penalty=lmbda, random_state=42, n_init=10)
            jm.fit((X_train - m) / s, ret_ser=train_df['Excess_Return'], sort_by='cumret')
            labels_full = jm.predict(X_combined)
            if hasattr(labels_full, 'values'):
                labels_full = labels_full.values

            # Extract OOS portion
            n_train = len(train_df)
            oos_labels = labels_full[n_train:].astype(int)
            all_dates.extend(oos_chunk.index.tolist())
            all_states.extend(oos_labels.tolist())
        current = chunk_end
    print_regimes(np.array(all_dates), np.array(all_states), f"B: predict(train+OOS), λ={lmbda}")


# ==========================================================================
# Approach C: Single fit on ALL data, extract OOS
# ==========================================================================
print("\n" + "=" * 80)
print("APPROACH C: Single fit on full 1996-2023, extract 2007-2023")
print("  This is NOT walk-forward — just checking if JM can identify regimes")
print("=" * 80)

full_df = df[(df.index >= '1996-01-01') & (df.index < oos_end)].copy()
X_full = full_df[return_features]
m, s = X_full.mean(), X_full.std()
X_full_std = (X_full - m) / s

for lmbda in [30.0, 46.42, 65.0, 80.0]:
    jm = JumpModel(n_components=2, jump_penalty=lmbda, random_state=42, n_init=10)
    jm.fit(X_full_std, ret_ser=full_df['Excess_Return'], sort_by='cumret')
    labels = jm.labels_
    if hasattr(labels, 'values'):
        labels = labels.values
    labels = np.array(labels, dtype=int)

    # Extract 2007-2023 portion
    oos_mask = full_df.index >= oos_start
    oos_dates = full_df.index[oos_mask].values
    oos_labels = labels[oos_mask]

    print_regimes(oos_dates, oos_labels, f"C: single fit all data, λ={lmbda}")


# ==========================================================================
# Approach D: predict_online (forward-only Viterbi) for comparison
# ==========================================================================
print("\n" + "=" * 80)
print("APPROACH D: fit(train) → predict_online(OOS) per chunk [for comparison]")
print("  Forward-only Viterbi (no backtracking), causal")
print("=" * 80)

for lmbda in [65.0, 80.0]:
    all_dates, all_states = [], []
    current = oos_start
    while current < oos_end:
        train_start = current - pd.DateOffset(years=11)
        train_df = df[(df.index >= train_start) & (df.index < current)].copy()
        chunk_end = min(current + pd.DateOffset(months=6), oos_end)
        oos_chunk = df[(df.index >= current) & (df.index < chunk_end)].copy()
        if len(train_df) >= 252*5 and len(oos_chunk) > 0:
            X_train = train_df[return_features]
            m, s = X_train.mean(), X_train.std()
            jm = JumpModel(n_components=2, jump_penalty=lmbda, random_state=42, n_init=10)
            jm.fit((X_train - m) / s, ret_ser=train_df['Excess_Return'], sort_by='cumret')
            labels = jm.predict_online((oos_chunk[return_features] - m) / s)
            if hasattr(labels, 'values'):
                labels = labels.values
            all_dates.extend(oos_chunk.index.tolist())
            all_states.extend(labels.astype(int).tolist())
        current = chunk_end
    print_regimes(np.array(all_dates), np.array(all_states), f"D: predict_online(OOS), λ={lmbda}")


# ==========================================================================
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"\n  {'Approach':<50} {'Bear%':>6} {'Shifts':>7} {'Periods':>8}")
print("  " + "-" * 75)
print(f"  {'Paper Figure 2 (reference)':<50} {'20.9%':>6} {'46':>7} {'~5-8':>8}")
