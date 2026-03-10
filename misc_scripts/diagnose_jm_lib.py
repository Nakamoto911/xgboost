"""
Diagnostic: Run JM-only strategy using the paper's official jumpmodels library.
Compare predict_online vs predict, and sweep lambdas.
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
oos_df = df[(df.index >= oos_start) & (df.index < oos_end)]
_, _, bh_sharpe, _, bh_mdd = backend.calculate_metrics(oos_df['Target_Return'], oos_df['RF_Rate'])
print(f"B&H: Sharpe={bh_sharpe:.3f}, MDD={bh_mdd:.1%}")
print(f"Paper JM LargeCap: Sharpe=0.59, MDD=-24.78%, Bear%=20.9%, Shifts=46\n")

return_features = [c for c in df.columns if c.startswith(('DD_', 'Avg_Ret_', 'Sortino_'))]


def run_library_jm(df, oos_start, oos_end, lmbda, method='online', n_init=10):
    """Run JM-only strategy using official jumpmodels library."""
    results = []
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

        # Standardize using training stats
        X_train = train_df[return_features]
        train_mean = X_train.mean()
        train_std = X_train.std()
        X_train_std = (X_train - train_mean) / train_std
        X_oos_std = (oos_chunk[return_features] - train_mean) / train_std

        # Fit with paper's library (n_init=10, max_iter=1000, sort by cumret)
        jm = JumpModel(n_components=2, jump_penalty=lmbda, random_state=42, n_init=n_init)
        jm.fit(X_train_std, ret_ser=train_df['Excess_Return'], sort_by='cumret')
        # After sorting: state 0 = highest cumret (bullish), state 1 = bearish

        # Predict OOS
        if method == 'online':
            oos_labels = jm.predict_online(X_oos_std)
        elif method == 'viterbi':
            oos_labels = jm.predict(X_oos_std)

        if hasattr(oos_labels, 'values'):
            oos_labels = oos_labels.values

        oos_chunk = oos_chunk.copy()
        oos_chunk['Forecast_State'] = oos_labels.astype(int)
        results.append(oos_chunk[['Target_Return', 'RF_Rate', 'Forecast_State']])
        current = chunk_end

    if not results:
        return None, {}

    full = pd.concat(results)
    signals = full['Forecast_State'].shift(1).fillna(0)
    alloc = 1.0 - signals
    strat_ret = (alloc * full['Target_Return']) + ((1.0 - alloc) * full['RF_Rate'])
    trades = alloc.diff().abs().fillna(0)
    full['Strat_Return'] = strat_ret - (trades * backend.TRANSACTION_COST)

    _, _, sharpe, sortino, mdd = backend.calculate_metrics(full['Strat_Return'], full['RF_Rate'])
    bear_pct = full['Forecast_State'].mean()
    switches = (full['Forecast_State'].diff().abs() > 0).sum()

    return full, {'sharpe': sharpe, 'sortino': sortino, 'mdd': mdd,
                  'bear_pct': bear_pct, 'switches': int(switches)}


# ==========================================================================
print("=" * 90)
print("TEST 1: Paper library - predict_online vs predict (Viterbi) across lambdas")
print("=" * 90)

print(f"\n  {'λ':>6} {'Method':<12} {'Sharpe':>8} {'MDD':>8} {'Bear%':>8} {'Shifts':>8}")
print("  " + "-" * 55)

for lmbda in [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]:
    for method in ['online', 'viterbi']:
        _, stats = run_library_jm(df, oos_start, oos_end, float(lmbda), method)
        if stats:
            marker = " *" if method == 'online' and abs(stats['sharpe'] - 0.59) < 0.05 else ""
            print(f"  {lmbda:6.1f} {method:<12} {stats['sharpe']:8.3f} {stats['mdd']:8.1%} "
                  f"{stats['bear_pct']:8.1%} {stats['switches']:8d}{marker}")


# ==========================================================================
print("\n" + "=" * 90)
print("TEST 2: Paper library online - fine lambda sweep around best")
print("=" * 90)

print(f"\n  {'λ':>8} {'Sharpe':>8} {'Sortino':>8} {'MDD':>8} {'Bear%':>8} {'Shifts':>8}")
print("  " + "-" * 55)

best_sharpe = -np.inf
best_lambda = None

for lmbda in np.arange(5, 55, 2.5):
    _, stats = run_library_jm(df, oos_start, oos_end, float(lmbda), 'online')
    if stats:
        marker = ""
        if abs(stats['bear_pct'] - 0.209) < 0.03:
            marker += " ~bear%"
        if abs(stats['switches'] - 46) < 10:
            marker += " ~shifts"
        print(f"  {lmbda:8.1f} {stats['sharpe']:8.3f} {stats['sortino']:8.3f} {stats['mdd']:8.1%} "
              f"{stats['bear_pct']:8.1%} {stats['switches']:8d}{marker}")
        if stats['sharpe'] > best_sharpe:
            best_sharpe = stats['sharpe']
            best_lambda = lmbda

print(f"\n  Best: λ={best_lambda} → Sharpe={best_sharpe:.3f}")

# ==========================================================================
print("\n" + "=" * 90)
print("TEST 3: Our implementation vs paper library (side-by-side)")
print("=" * 90)

print(f"\n  {'λ':>6} {'Ours (greedy)':>14} {'Lib (online)':>14} {'Lib (viterbi)':>14}")
print("  " + "-" * 55)

for lmbda in [10, 20, 30, 40, 50, 80]:
    # Ours
    backend._forecast_cache.clear()
    res_ours = backend.simulate_strategy(df, oos_start, oos_end, float(lmbda),
                                          include_xgboost=False)
    s_ours = 0
    if not res_ours.empty:
        _, _, s_ours, _, _ = backend.calculate_metrics(res_ours['Strat_Return'], res_ours['RF_Rate'])

    # Library
    _, stats_online = run_library_jm(df, oos_start, oos_end, float(lmbda), 'online')
    _, stats_viterbi = run_library_jm(df, oos_start, oos_end, float(lmbda), 'viterbi')

    s_online = stats_online.get('sharpe', 0) if stats_online else 0
    s_viterbi = stats_viterbi.get('sharpe', 0) if stats_viterbi else 0

    print(f"  {lmbda:6.0f} {s_ours:14.3f} {s_online:14.3f} {s_viterbi:14.3f}")


# ==========================================================================
print("\n" + "=" * 90)
print("TEST 4: Check regime quality at best lambda (library online)")
print("=" * 90)

# Run at the lambda closest to paper results
best_l = best_lambda if best_lambda else 25.0
full, stats = run_library_jm(df, oos_start, oos_end, float(best_l), 'online')

if full is not None:
    print(f"\nλ={best_l}: Sharpe={stats['sharpe']:.3f}, MDD={stats['mdd']:.1%}, "
          f"Bear%={stats['bear_pct']:.1%}, Shifts={stats['switches']}")

    # Sub-period analysis
    periods = {
        'GFC (2007-2009)': ('2007-01-01', '2010-01-01'),
        'Recovery (2010-2015)': ('2010-01-01', '2016-01-01'),
        'Late Cycle (2016-2019)': ('2016-01-01', '2020-01-01'),
        'COVID (2020-2021)': ('2020-01-01', '2022-01-01'),
        'Post-COVID (2022-2023)': ('2022-01-01', '2024-01-01'),
    }

    print(f"\n  {'Period':<25} {'Strat':>8} {'B&H':>8} {'Delta':>8} {'Bear%':>8}")
    print("  " + "-" * 55)

    for name, (ps, pe) in periods.items():
        mask = (full.index >= ps) & (full.index < pe)
        sub = full[mask]
        if len(sub) < 20:
            continue
        oos_sub = oos_df[(oos_df.index >= ps) & (oos_df.index < pe)]
        _, _, ss, _, _ = backend.calculate_metrics(sub['Strat_Return'], sub['RF_Rate'])
        _, _, bs, _, _ = backend.calculate_metrics(oos_sub['Target_Return'], oos_sub['RF_Rate'])
        bear_pct = sub['Forecast_State'].mean()
        print(f"  {name:<25} {ss:8.3f} {bs:8.3f} {ss-bs:+8.3f} {bear_pct:8.1%}")


print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"""
B&H Sharpe:  {bh_sharpe:.3f}
Paper JM:    0.59 (Bear%=20.9%, 46 shifts)
Best library online: λ={best_lambda}, Sharpe={best_sharpe:.3f}

Key question: Does the paper use predict_online or predict (full Viterbi on OOS)?
- predict_online: no look-ahead, but forward-only Viterbi (accumulated costs)
- predict (Viterbi): look-ahead within 6-month OOS window, much higher Sharpe
""")
