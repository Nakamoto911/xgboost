"""
Diagnostic: Why does our JM-only strategy lose to B&H?
Paper: JM Sharpe 0.59 vs B&H 0.50 for LargeCap (2007-2023)
Ours: JM best ~0.515

DIFFERENCES found between our implementation and paper's reference library (jumpmodels):

1. PREDICT_ONLINE: Paper uses forward-only Viterbi (accumulated costs from t=0),
   we use greedy 1-step (only considers previous state). Paper's approach is globally
   optimal considering full history; ours is myopic.

2. N_INIT: Paper uses 10 random initializations (kmeans++), we use 1 (fixed seed=42).
   More inits → better chance of finding global optimum.

3. MAX_ITER: Paper uses 1000, we use 20. May not converge with 20.

4. PREDICT vs PREDICT_ONLINE: Paper also has predict() which runs full Viterbi
   (with backtracking) on new data. For "JM" row in Table 4, this might be used.

5. STATE SORTING: Paper sorts by "cumret" = ret_ * freq (weighted by time in state).
   We compare sum of excess returns per state, which is equivalent.

6. INITIAL STATE: Paper's predict_online doesn't condition on last training state —
   the DP starts fresh with values[0] = loss[0]. Our greedy starts from the last
   training state.

This script tests each difference.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['XGB_END_DATE'] = '2024-01-01'

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import main as backend
from config import StrategyConfig

backend.END_DATE = '2024-01-01'

config = StrategyConfig()
df = backend.fetch_and_prepare_data()

oos_start = pd.to_datetime('2007-01-01')
oos_end = pd.to_datetime('2024-01-01')
oos_df = df[(df.index >= oos_start) & (df.index < oos_end)]

_, _, bh_sharpe, _, bh_mdd = backend.calculate_metrics(oos_df['Target_Return'], oos_df['RF_Rate'])
print(f"B&H: Sharpe={bh_sharpe:.3f}, MDD={bh_mdd:.1%}")

return_features = [c for c in df.columns if c.startswith(('DD_', 'Avg_Ret_', 'Sortino_'))]

# Paper reference for LargeCap
print(f"Paper JM: Sharpe=0.59, MDD=-24.78%, Bear%=20.9%, Shifts=46\n")


def paper_predict_online(means, penalty, X_oos, n_states=2):
    """Paper's predict_online: forward-only Viterbi (no backtracking).
    values[t] = loss[t] + min_k(values[t-1,k] + penalty_mx[k,:])
    labels[t] = argmin(values[t])
    Does NOT condition on last training state.
    """
    loss_mx = 0.5 * cdist(X_oos, means, 'sqeuclidean')  # (T, K)
    penalty_mx = penalty * (np.ones((n_states, n_states)) - np.eye(n_states))
    T, K = loss_mx.shape
    values = np.empty((T, K))
    values[0] = loss_mx[0]
    for t in range(1, T):
        values[t] = loss_mx[t] + (values[t-1][:, np.newaxis] + penalty_mx).min(axis=0)
    labels = values.argmin(axis=1)
    return labels


def paper_predict_viterbi(means, penalty, X_oos, n_states=2):
    """Paper's predict(): full Viterbi (with backtracking) on new data.
    Globally optimal state sequence for the entire OOS window.
    """
    loss_mx = 0.5 * cdist(X_oos, means, 'sqeuclidean')
    penalty_mx = penalty * (np.ones((n_states, n_states)) - np.eye(n_states))
    T, K = loss_mx.shape
    values = np.empty((T, K))
    backptr = np.empty((T, K), dtype=int)
    values[0] = loss_mx[0]
    for t in range(1, T):
        trans = values[t-1][:, np.newaxis] + penalty_mx
        backptr[t] = trans.argmin(axis=0)
        values[t] = loss_mx[t] + trans.min(axis=0)
    # Backtrace
    labels = np.empty(T, dtype=int)
    labels[-1] = values[-1].argmin()
    for t in range(T-2, -1, -1):
        labels[t] = backptr[t+1, labels[t+1]]
    return labels


def our_predict_online(means, penalty, X_oos, last_state):
    """Our current implementation: greedy, conditioned on last training state."""
    T = X_oos.shape[0]
    states = np.zeros(T, dtype=int)
    prev = last_state
    for t in range(T):
        dist = 0.5 * np.sum((X_oos[t, None, :] - means)**2, axis=1)
        pen = np.full(means.shape[0], penalty)
        pen[prev] = 0.0
        states[t] = np.argmin(dist + pen)
        prev = states[t]
    return states


def run_jm_strategy(df, oos_start, oos_end, lmbda, predict_fn_name, n_init=1, max_iter=20):
    """Run JM-only strategy with specified predict function and fit params."""
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

        # Standardize
        X_train = train_df[return_features]
        train_mean = X_train.mean()
        train_std = X_train.std()
        X_train_std = ((X_train - train_mean) / train_std).values
        X_oos_std = ((oos_chunk[return_features] - train_mean) / train_std).values

        # Fit JM with specified n_init and max_iter
        best_val = np.inf
        best_means = None
        best_states = None

        np.random.seed(42)
        for init_i in range(n_init):
            # K-means++ init
            from sklearn.cluster import kmeans_plusplus
            centers_init, _ = kmeans_plusplus(X_train_std, 2, random_state=42 + init_i)

            jm = backend.StatisticalJumpModel(n_states=2, lambda_penalty=lmbda, max_iter=max_iter)
            jm.means = centers_init.copy()
            states = jm.fit_predict(X_train_std)

            # Compute objective value
            loss_mx = 0.5 * cdist(X_train_std, jm.means, 'sqeuclidean')
            total_loss = sum(loss_mx[t, states[t]] for t in range(len(states)))
            total_penalty = lmbda * (np.diff(states) != 0).sum()
            val = total_loss + total_penalty

            if val < best_val:
                best_val = val
                best_means = jm.means.copy()
                best_states = states.copy()

        states = best_states
        means = best_means

        # Align states
        cum_ret_0 = train_df['Excess_Return'][states == 0].sum()
        cum_ret_1 = train_df['Excess_Return'][states == 1].sum()
        if cum_ret_1 > cum_ret_0:
            states = 1 - states
            means = means[::-1].copy()

        # OOS prediction using specified method
        if predict_fn_name == 'our_greedy':
            oos_states = our_predict_online(means, lmbda, X_oos_std, states[-1])
        elif predict_fn_name == 'paper_online':
            oos_states = paper_predict_online(means, lmbda, X_oos_std)
            # Re-align if needed (paper predict_online doesn't inherit alignment)
            # Check: does state 0 still correspond to bullish?
            # We check using the OOS returns
        elif predict_fn_name == 'paper_viterbi':
            oos_states = paper_predict_viterbi(means, lmbda, X_oos_std)
        elif predict_fn_name == 'paper_online_conditioned':
            # Paper's forward Viterbi but prepend last training state
            # to condition on training history
            X_last_train = X_train_std[-1:, :]
            X_combined = np.vstack([X_last_train, X_oos_std])
            combined_states = paper_predict_online(means, lmbda, X_combined)
            oos_states = combined_states[1:]  # remove the prepended point

        oos_chunk['Forecast_State'] = oos_states
        results.append(oos_chunk[['Target_Return', 'RF_Rate', 'Forecast_State']])

        current = chunk_end

    if not results:
        return None, {}

    full = pd.concat(results)

    # Apply signal with 1-day shift (same as simulate_strategy)
    signals = full['Forecast_State'].shift(1).fillna(0)
    alloc = 1.0 - signals
    strat_ret = (alloc * full['Target_Return']) + ((1.0 - alloc) * full['RF_Rate'])
    trades = alloc.diff().abs().fillna(0)
    full['Strat_Return'] = strat_ret - (trades * backend.TRANSACTION_COST)

    _, _, sharpe, sortino, mdd = backend.calculate_metrics(full['Strat_Return'], full['RF_Rate'])
    bear_pct = full['Forecast_State'].mean()
    switches = (full['Forecast_State'].diff().abs() > 0).sum()

    stats = {'sharpe': sharpe, 'mdd': mdd, 'bear_pct': bear_pct, 'switches': switches}
    return full, stats


# ==========================================================================
print("=" * 90)
print("TEST 1: Prediction method comparison (fixed lambda sweep)")
print("=" * 90)

methods = ['our_greedy', 'paper_online', 'paper_viterbi', 'paper_online_conditioned']

for lmbda in [10.0, 21.54, 46.42, 100.0]:
    print(f"\n--- λ = {lmbda:.1f} ---")
    print(f"  {'Method':<30} {'Sharpe':>8} {'MDD':>8} {'Bear%':>8} {'Shifts':>8}")
    for method in methods:
        _, stats = run_jm_strategy(df, oos_start, oos_end, lmbda, method)
        if stats:
            print(f"  {method:<30} {stats['sharpe']:8.3f} {stats['mdd']:8.1%} "
                  f"{stats['bear_pct']:8.1%} {stats['switches']:8.0f}")

# ==========================================================================
print("\n" + "=" * 90)
print("TEST 2: Effect of n_init (number of random initializations)")
print("=" * 90)

lmbda = 46.42  # Best from previous tests for JM-only
print(f"\nUsing λ={lmbda}, paper_online predict method")
print(f"  {'n_init':<10} {'Sharpe':>8} {'MDD':>8} {'Bear%':>8}")

for n_init in [1, 3, 5, 10]:
    _, stats = run_jm_strategy(df, oos_start, oos_end, lmbda, 'paper_online', n_init=n_init)
    if stats:
        print(f"  {n_init:<10} {stats['sharpe']:8.3f} {stats['mdd']:8.1%} {stats['bear_pct']:8.1%}")

# ==========================================================================
print("\n" + "=" * 90)
print("TEST 3: Effect of max_iter (convergence)")
print("=" * 90)

print(f"\nUsing λ={lmbda}, paper_online, n_init=1")
print(f"  {'max_iter':<10} {'Sharpe':>8} {'MDD':>8}")

for max_iter in [5, 10, 20, 50, 100, 500]:
    _, stats = run_jm_strategy(df, oos_start, oos_end, lmbda, 'paper_online', max_iter=max_iter)
    if stats:
        print(f"  {max_iter:<10} {stats['sharpe']:8.3f} {stats['mdd']:8.1%}")

# ==========================================================================
print("\n" + "=" * 90)
print("TEST 4: Using paper's jumpmodels library directly")
print("=" * 90)

try:
    from jumpmodels import JumpModel

    def run_with_library(df, oos_start, oos_end, lmbda, predict_method='online'):
        """Run using the official jumpmodels library."""
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

            # Standardize
            X_train = train_df[return_features]
            train_mean = X_train.mean()
            train_std = X_train.std()
            X_train_std = (X_train - train_mean) / train_std
            X_oos_std = (oos_chunk[return_features] - train_mean) / train_std

            # Fit using library
            jm = JumpModel(n_components=2, jump_penalty=lmbda, random_state=42)
            jm.fit(X_train_std, ret_ser=train_df['Excess_Return'], sort_by='cumret')

            # State 0 should be bullish (highest cumret) after sort

            # Predict OOS
            if predict_method == 'online':
                oos_labels = jm.predict_online(X_oos_std)
            elif predict_method == 'viterbi':
                oos_labels = jm.predict(X_oos_std)

            # Convert to numpy if Series
            if hasattr(oos_labels, 'values'):
                oos_labels = oos_labels.values

            oos_chunk['Forecast_State'] = oos_labels
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

        _, _, sharpe, _, mdd = backend.calculate_metrics(full['Strat_Return'], full['RF_Rate'])
        bear_pct = full['Forecast_State'].mean()
        switches = (full['Forecast_State'].diff().abs() > 0).sum()
        return full, {'sharpe': sharpe, 'mdd': mdd, 'bear_pct': bear_pct, 'switches': switches}

    print(f"\n  {'λ':>6} {'Method':<15} {'Sharpe':>8} {'MDD':>8} {'Bear%':>8} {'Shifts':>8}")
    print("  " + "-" * 65)

    for lmbda in [10.0, 21.54, 46.42, 100.0]:
        for method in ['online', 'viterbi']:
            _, stats = run_with_library(df, oos_start, oos_end, lmbda, method)
            if stats:
                print(f"  {lmbda:6.1f} {method:<15} {stats['sharpe']:8.3f} {stats['mdd']:8.1%} "
                      f"{stats['bear_pct']:8.1%} {stats['switches']:8.0f}")

except ImportError as e:
    print(f"  jumpmodels library not available: {e}")

# ==========================================================================
print("\n" + "=" * 90)
print("TEST 5: Fine lambda sweep with paper library (online)")
print("=" * 90)

try:
    from jumpmodels import JumpModel

    print(f"\n  {'λ':>8} {'Sharpe':>8} {'MDD':>8} {'Bear%':>8} {'Shifts':>8}")
    print("  " + "-" * 50)

    best_sharpe = -np.inf
    best_lambda = None

    for lmbda in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100]:
        _, stats = run_with_library(df, oos_start, oos_end, float(lmbda), 'online')
        if stats:
            print(f"  {lmbda:8.1f} {stats['sharpe']:8.3f} {stats['mdd']:8.1%} "
                  f"{stats['bear_pct']:8.1%} {stats['switches']:8.0f}")
            if stats['sharpe'] > best_sharpe:
                best_sharpe = stats['sharpe']
                best_lambda = lmbda

    print(f"\n  Best: λ={best_lambda} → Sharpe={best_sharpe:.3f}")

except ImportError as e:
    print(f"  jumpmodels library not available: {e}")

# ==========================================================================
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"""
Paper reference:  JM Sharpe=0.59, MDD=-24.78%, Bear%=20.9%, Shifts=46
B&H:              Sharpe={bh_sharpe:.3f}, MDD={bh_mdd:.1%}

Key differences found between our JM and paper's jumpmodels library:
1. predict_online: Paper uses forward-only Viterbi (accumulated costs), we use greedy
2. n_init: Paper uses 10, we use 1
3. max_iter: Paper uses 1000, we use 20
4. State sorting: Paper uses cumret-weighted sorting with ret_ser parameter
""")
