"""
Test: Does passing training+OOS data to predict_online (warm-start Viterbi)
improve performance vs passing only OOS data (cold-start)?

The paper's jumpmodels library might condition the forward Viterbi DP on
accumulated costs from training data, giving more stable state assignments
for early OOS days. Our current implementation starts fresh each 6-month chunk.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from main import (fetch_and_prepare_data, StatisticalJumpModel, calculate_metrics,
                  LAMBDA_GRID, PAPER_EWMA_HL, TRANSACTION_COST)
from config import StrategyConfig

# Test on ^SP500TR first
TARGET = '^SP500TR'
OOS_START = '2007-01-01'
OOS_END = '2024-01-01'  # Paper period
EWMA_HL = PAPER_EWMA_HL.get(TARGET, 8)

print(f"Testing warm-start vs cold-start predict_online")
print(f"Asset: {TARGET}, Period: {OOS_START} to {OOS_END}, EWMA HL: {EWMA_HL}")
print(f"Lambda grid: {LAMBDA_GRID}")
print("=" * 80)

df = fetch_and_prepare_data()
return_features = [c for c in df.columns if c.startswith(('DD_', 'Avg_Ret_', 'Sortino_'))]
config = StrategyConfig()

def run_backtest(df, warm_start=False, fixed_lambda=None):
    """Run walk-forward backtest with warm or cold predict_online."""
    current = pd.to_datetime(OOS_START)
    end = pd.to_datetime(OOS_END)

    all_chunks = []
    lambda_history = []

    while current < end:
        chunk_end = min(current + pd.DateOffset(months=6), end)
        val_start = current - pd.DateOffset(years=5)

        # Lambda tuning on validation window
        if fixed_lambda is not None:
            best_lambda = fixed_lambda
        else:
            best_metric = -np.inf
            best_lambda = LAMBDA_GRID[len(LAMBDA_GRID)//2]
            for lmbda in LAMBDA_GRID:
                val_res = run_chunk(df, val_start, current, lmbda, warm_start=warm_start)
                if val_res is not None and not val_res.empty:
                    _, _, sharpe, _, _ = calculate_metrics(val_res['Strat_Return'], val_res['RF_Rate'])
                    if not np.isnan(sharpe) and sharpe > best_metric:
                        best_metric = sharpe
                        best_lambda = lmbda

        lambda_history.append(best_lambda)

        # OOS chunk with best lambda
        oos = run_chunk(df, current, chunk_end, best_lambda, warm_start=warm_start, is_oos=True)
        if oos is not None:
            all_chunks.append(oos)
        current = chunk_end

    if not all_chunks:
        return None, lambda_history

    full = pd.concat(all_chunks)

    # Apply EWMA smoothing
    if EWMA_HL == 0:
        full['State_Prob'] = full['Raw_Prob']
    else:
        full['State_Prob'] = full['Raw_Prob'].ewm(halflife=EWMA_HL).mean()

    full['Forecast_State'] = (full['State_Prob'] > 0.5).astype(int)
    signals = full['Forecast_State'].shift(1).fillna(0)
    alloc = 1.0 - signals
    full['Strat_Return'] = (alloc * full['Target_Return']) + ((1 - alloc) * full['RF_Rate'])
    trades = alloc.diff().abs().fillna(0)
    full['Strat_Return'] -= trades * TRANSACTION_COST

    return full, lambda_history


def run_chunk(df, start, end, lmbda, warm_start=False, is_oos=False):
    """Run a single forecast chunk. If warm_start, pass train+oos to predict_online."""
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    train_start = start_dt - pd.DateOffset(years=11)
    train_df = df[(df.index >= train_start) & (df.index < start_dt)].copy()
    oos_df = df[(df.index >= start_dt) & (df.index < end_dt)].copy()

    if len(train_df) < 252 * 5 or len(oos_df) == 0:
        return None

    # Standardize for JM
    X_train = train_df[return_features]
    train_mean = X_train.mean()
    train_std = X_train.std()
    train_std[train_std == 0] = 1.0
    X_train_std = (X_train - train_mean) / train_std

    # Fit JM
    jm = StatisticalJumpModel(n_states=2, lambda_penalty=lmbda)
    identified_states = jm.fit_predict(X_train_std.values)

    # Align states
    cum_ret_0 = train_df['Excess_Return'][identified_states == 0].sum()
    cum_ret_1 = train_df['Excess_Return'][identified_states == 1].sum()
    if cum_ret_1 > cum_ret_0:
        identified_states = 1 - identified_states
        jm.means = jm.means[::-1].copy()

    # Prepare XGB targets
    train_df_xgb = train_df.iloc[:-1].copy()
    train_df_xgb['Target_State'] = np.roll(identified_states, -1)[:-1]

    # Get all features for XGB
    all_features = [c for c in train_df.columns if c.startswith(('DD_', 'Avg_Ret_', 'Sortino_', 'Yield_', 'VIX_', 'Stock_Bond'))]

    from xgboost import XGBClassifier
    xgb = XGBClassifier(eval_metric='logloss', random_state=42, **config.xgb_params)
    xgb.fit(train_df_xgb[all_features], train_df_xgb['Target_State'])

    # XGB prediction on OOS
    oos_probs = xgb.predict_proba(oos_df[all_features])[:, 1]

    # JM predict_online: warm vs cold
    X_oos_std = (oos_df[return_features] - train_mean) / train_std

    if warm_start:
        # Pass training + OOS data, extract only OOS labels
        X_all_std = pd.concat([X_train_std, X_oos_std])
        all_states = jm.predict_online(X_all_std.values)
        oos_states = all_states[len(X_train_std):]
    else:
        # Cold start: pass only OOS data (current implementation)
        oos_states = jm.predict_online(X_oos_std.values)

    result = oos_df[['Target_Return', 'RF_Rate']].copy()
    result['Raw_Prob'] = oos_probs
    result['JM_State'] = oos_states

    return result


# Test 1: Fixed lambda comparison (eliminates WF noise)
print("\n--- TEST 1: Fixed lambda, cold vs warm predict_online ---")
print(f"{'Lambda':>8}  {'Cold Sharpe':>12}  {'Warm Sharpe':>12}  {'Delta':>8}  {'Cold Shifts':>12}  {'Warm Shifts':>12}")
print("-" * 72)

for lmbda in [10.0, 21.54, 46.42]:
    cold, _ = run_backtest(df, warm_start=False, fixed_lambda=lmbda)
    warm, _ = run_backtest(df, warm_start=True, fixed_lambda=lmbda)

    if cold is not None and warm is not None:
        _, _, cold_sharpe, _, _ = calculate_metrics(cold['Strat_Return'], cold['RF_Rate'])
        _, _, warm_sharpe, _, _ = calculate_metrics(warm['Strat_Return'], warm['RF_Rate'])
        cold_shifts = (cold['Forecast_State'].diff().abs() > 0).sum()
        warm_shifts = (warm['Forecast_State'].diff().abs() > 0).sum()
        delta = warm_sharpe - cold_sharpe
        print(f"{lmbda:>8.2f}  {cold_sharpe:>12.3f}  {warm_sharpe:>12.3f}  {delta:>+8.3f}  {cold_shifts:>12}  {warm_shifts:>12}")


# Test 2: Full walk-forward comparison
print("\n\n--- TEST 2: Full walk-forward, cold vs warm predict_online ---")
cold_full, cold_lambdas = run_backtest(df, warm_start=False)
warm_full, warm_lambdas = run_backtest(df, warm_start=True)

if cold_full is not None and warm_full is not None:
    _, _, cold_s, cold_sort, cold_mdd = calculate_metrics(cold_full['Strat_Return'], cold_full['RF_Rate'])
    _, _, warm_s, warm_sort, warm_mdd = calculate_metrics(warm_full['Strat_Return'], warm_full['RF_Rate'])
    _, _, bh_s, _, _ = calculate_metrics(cold_full['Target_Return'], cold_full['RF_Rate'])

    print(f"\n{'Method':<20} {'Sharpe':>8} {'Sortino':>8} {'MDD':>8} {'Shifts':>8}")
    print("-" * 55)
    print(f"{'Cold (current)':<20} {cold_s:>8.3f} {cold_sort:>8.3f} {cold_mdd:>7.1%} {(cold_full['Forecast_State'].diff().abs() > 0).sum():>8}")
    print(f"{'Warm (train+OOS)':<20} {warm_s:>8.3f} {warm_sort:>8.3f} {warm_mdd:>7.1%} {(warm_full['Forecast_State'].diff().abs() > 0).sum():>8}")
    print(f"{'B&H':<20} {bh_s:>8.3f}")

    print(f"\nLambda selections - Cold: {[f'{l:.1f}' for l in cold_lambdas]}")
    print(f"Lambda selections - Warm: {[f'{l:.1f}' for l in warm_lambdas]}")

    # Compare JM state differences
    cold_jm = cold_full['JM_State'] if 'JM_State' in cold_full.columns else None
    warm_jm = warm_full['JM_State'] if 'JM_State' in warm_full.columns else None
    if cold_jm is not None and warm_jm is not None:
        matching = (cold_jm.values == warm_jm.values).mean()
        print(f"\nJM state agreement: {matching:.1%}")

print("\nDone.")
