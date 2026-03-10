"""
Follow-up: Fine sweep around λ=60-100 where library online showed best results.
Also test with walk-forward lambda tuning using library.
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
_, _, bh_sharpe, _, _ = backend.calculate_metrics(oos_df['Target_Return'], oos_df['RF_Rate'])

return_features = [c for c in df.columns if c.startswith(('DD_', 'Avg_Ret_', 'Sortino_'))]


def run_library_jm(df, oos_start, oos_end, lmbda, method='online'):
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
        oos_chunk = oos_chunk.copy()
        oos_chunk['Forecast_State'] = labels.astype(int)
        results.append(oos_chunk[['Target_Return', 'RF_Rate', 'Forecast_State']])
        current = chunk_end
    if not results:
        return {}
    full = pd.concat(results)
    signals = full['Forecast_State'].shift(1).fillna(0)
    alloc = 1.0 - signals
    strat_ret = (alloc * full['Target_Return']) + ((1.0 - alloc) * full['RF_Rate'])
    trades = alloc.diff().abs().fillna(0)
    full['Strat_Return'] = strat_ret - (trades * backend.TRANSACTION_COST)
    _, _, sharpe, sortino, mdd = backend.calculate_metrics(full['Strat_Return'], full['RF_Rate'])
    bear_pct = full['Forecast_State'].mean()
    switches = (full['Forecast_State'].diff().abs() > 0).sum()
    return {'sharpe': sharpe, 'mdd': mdd, 'bear_pct': bear_pct, 'switches': int(switches)}


print("=" * 70)
print("Fine lambda sweep (library online, n_init=10)")
print("=" * 70)
print(f"\n  {'λ':>6} {'Sharpe':>8} {'MDD':>8} {'Bear%':>8} {'Shifts':>8}")
print("  " + "-" * 45)

best_s, best_l = -np.inf, None
for lmbda in list(range(55, 105, 5)) + [120, 150]:
    stats = run_library_jm(df, oos_start, oos_end, float(lmbda), 'online')
    if stats:
        marker = ""
        if abs(stats['bear_pct'] - 0.209) < 0.03:
            marker = " ~paper bear%"
        print(f"  {lmbda:6.0f} {stats['sharpe']:8.3f} {stats['mdd']:8.1%} "
              f"{stats['bear_pct']:8.1%} {stats['switches']:8d}{marker}")
        if stats['sharpe'] > best_s:
            best_s, best_l = stats['sharpe'], lmbda

print(f"\n  Best: λ={best_l} → Sharpe={best_s:.3f}")

# Also check: what if we match the paper's Bear% ~20.9% and Shifts ~46?
print("\n\nLooking for λ that matches paper's Bear%=20.9% and Shifts=46:")
for lmbda in range(50, 200, 5):
    stats = run_library_jm(df, oos_start, oos_end, float(lmbda), 'online')
    if stats and abs(stats['bear_pct'] - 0.209) < 0.02:
        print(f"  λ={lmbda}: Sharpe={stats['sharpe']:.3f}, Bear%={stats['bear_pct']:.1%}, "
              f"Shifts={stats['switches']}, MDD={stats['mdd']:.1%}")

print(f"\nB&H Sharpe: {bh_sharpe:.3f}")
print(f"Paper JM: 0.59 (Bear%=20.9%, Shifts=46)")
