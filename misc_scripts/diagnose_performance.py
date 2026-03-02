"""
Comprehensive diagnostic test suite to understand why B&H beats JM-XGB.

Investigates:
1. Signal quality (accuracy, precision, recall of regime forecasts)
2. Time-in-market analysis (how much time is spent in risk-free vs invested)
3. Sub-period performance decomposition
4. EWMA smoothing sensitivity (paper says tune from {0, 2, 4, 8})
5. Lambda tuning behavior (what lambdas are chosen and why)
6. XGBoost prediction calibration
7. False signal cost analysis
8. Comparison of raw vs smoothed signals
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Patch main.py to extend end date to 2026
import main as backend

# Override end date for our analysis
backend.END_DATE = '2026-01-01'

def load_data():
    """Load data from existing cache."""
    cache_file = 'data_cache.pkl'
    if os.path.exists(cache_file):
        print("Loading data from existing cache...")
        return pd.read_pickle(cache_file)
    return backend.fetch_and_prepare_data()


def test_signal_quality(df):
    """Test 1: Analyze signal quality - how accurate are regime forecasts?"""
    print("\n" + "="*80)
    print("TEST 1: SIGNAL QUALITY ANALYSIS")
    print("="*80)

    backend._forecast_cache.clear()

    # Run full strategy for extended period
    res = backend.simulate_strategy(df, '2007-01-01', '2026-01-01', lambda_penalty=10.0, include_xgboost=True)

    if res.empty:
        print("ERROR: No results returned")
        return

    # Analyze forecast states
    trading_signal = res['Forecast_State'].shift(1).fillna(0)

    total_days = len(res)
    bull_days = (trading_signal == 0).sum()
    bear_days = (trading_signal == 1).sum()

    print(f"\nTotal trading days: {total_days}")
    print(f"Days invested (bull): {bull_days} ({100*bull_days/total_days:.1f}%)")
    print(f"Days in risk-free (bear): {bear_days} ({100*bear_days/total_days:.1f}%)")

    # Calculate returns during each regime
    bull_mask = trading_signal == 0
    bear_mask = trading_signal == 1

    avg_ret_when_invested = res.loc[bull_mask, 'Target_Return'].mean() * 252
    avg_ret_when_out = res.loc[bear_mask, 'Target_Return'].mean() * 252

    print(f"\nAvg annualized return of market WHEN model says BULL: {avg_ret_when_invested*100:.2f}%")
    print(f"Avg annualized return of market WHEN model says BEAR: {avg_ret_when_out*100:.2f}%")
    print(f"Signal value (bull - bear returns): {(avg_ret_when_invested - avg_ret_when_out)*100:.2f}%")

    if avg_ret_when_out > 0:
        print("WARNING: Market goes UP on average when model says BEAR - signals have negative value!")

    # Count regime switches
    switches = trading_signal.diff().abs().sum()
    print(f"\nTotal regime switches: {int(switches)}")
    print(f"Avg switches per year: {switches / (total_days/252):.1f}")

    return res


def test_subperiod_performance(df):
    """Test 2: Performance decomposition by market regime periods."""
    print("\n" + "="*80)
    print("TEST 2: SUB-PERIOD PERFORMANCE DECOMPOSITION")
    print("="*80)

    backend._forecast_cache.clear()

    # Run strategy
    res = backend.simulate_strategy(df, '2007-01-01', '2026-01-01', lambda_penalty=10.0, include_xgboost=True)

    if res.empty:
        return

    periods = {
        'GFC (2007-2009)': ('2007-01-01', '2009-03-09'),
        'Recovery (2009-2015)': ('2009-03-09', '2015-01-01'),
        'Bull (2015-2020)': ('2015-01-01', '2020-02-19'),
        'COVID Crash (2020-Feb to 2020-Mar)': ('2020-02-19', '2020-03-23'),
        'COVID Recovery (2020-Mar to 2021)': ('2020-03-23', '2021-01-01'),
        'Post-COVID Bull (2021)': ('2021-01-01', '2022-01-01'),
        '2022 Bear Market': ('2022-01-01', '2022-10-12'),
        '2023 Recovery': ('2023-01-01', '2024-01-01'),
        '2024 Bull': ('2024-01-01', '2025-01-01'),
        '2025+': ('2025-01-01', '2026-01-01'),
    }

    print(f"\n{'Period':<40} {'B&H Return':>12} {'JM-XGB Return':>14} {'% Time Bull':>12} {'Winner':>10}")
    print("-" * 90)

    for name, (start, end) in periods.items():
        mask = (res.index >= start) & (res.index < end)
        if mask.sum() == 0:
            continue

        period_data = res[mask]
        bh_ret = (1 + period_data['Target_Return']).prod() - 1
        strat_ret = (1 + period_data['Strat_Return']).prod() - 1

        signal = period_data['Forecast_State'].shift(1).fillna(0)
        pct_bull = (signal == 0).mean() * 100

        winner = "JM-XGB" if strat_ret > bh_ret else "B&H"
        print(f"{name:<40} {bh_ret*100:>11.2f}% {strat_ret*100:>13.2f}% {pct_bull:>11.1f}% {winner:>10}")


def test_ewma_sensitivity(df):
    """Test 3: EWMA smoothing sensitivity - paper says tune from {0, 2, 4, 8}."""
    print("\n" + "="*80)
    print("TEST 3: EWMA SMOOTHING SENSITIVITY")
    print("="*80)
    print("Paper specifies halflife candidates: {0, 2, 4, 8}")
    print("Current implementation hardcodes halflife=8")

    halflife_candidates = [0, 2, 4, 8, 16]

    for hl in halflife_candidates:
        backend._forecast_cache.clear()

        # Get raw forecasts
        results = []
        current_date = pd.to_datetime('2007-01-01')
        end_date = pd.to_datetime('2026-01-01')

        while current_date < end_date:
            res = backend.run_period_forecast(df, current_date, lambda_penalty=10.0, include_xgboost=True)
            if res is not None:
                results.append(res)
            current_date += pd.DateOffset(months=6)

        if not results:
            continue

        full_res = pd.concat(results)

        # Apply EWMA with different halflife
        if hl == 0:
            full_res['State_Prob'] = full_res['Raw_Prob']
        else:
            full_res['State_Prob'] = full_res['Raw_Prob'].ewm(halflife=hl).mean()

        full_res['Forecast_State'] = (full_res['State_Prob'] > 0.5).astype(int)

        # Calculate strategy returns
        trading_signal = full_res['Forecast_State'].shift(1).fillna(0)
        strat_returns = np.where(trading_signal == 0, full_res['Target_Return'], full_res['RF_Rate'])
        trades = trading_signal.diff().abs().fillna(0)
        strat_returns = strat_returns - (trades.values * backend.TRANSACTION_COST)

        full_res['Strat_Return'] = strat_returns

        # Metrics
        ret, vol, sharpe, sortino, mdd = backend.calculate_metrics(
            pd.Series(strat_returns, index=full_res.index), full_res['RF_Rate'])

        bh_ret, bh_vol, bh_sharpe, bh_sortino, bh_mdd = backend.calculate_metrics(
            full_res['Target_Return'], full_res['RF_Rate'])

        n_trades = int(trades.sum())
        pct_invested = (trading_signal == 0).mean() * 100

        print(f"\nHalflife={hl:>2}: Sharpe={sharpe:.3f} Sortino={sortino:.3f} "
              f"Ret={ret*100:.1f}% Vol={vol*100:.1f}% MDD={mdd*100:.1f}% "
              f"Trades={n_trades} Invested={pct_invested:.0f}%")

    # Print B&H baseline
    print(f"\nB&H Baseline: Sharpe={bh_sharpe:.3f} Sortino={bh_sortino:.3f} "
          f"Ret={bh_ret*100:.1f}% Vol={bh_vol*100:.1f}% MDD={bh_mdd*100:.1f}%")


def test_lambda_sensitivity(df):
    """Test 4: Lambda sensitivity - does the optimal lambda vary significantly?"""
    print("\n" + "="*80)
    print("TEST 4: LAMBDA SENSITIVITY (Fixed Lambda OOS Performance)")
    print("="*80)

    test_lambdas = [0.0, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    for lmbda in test_lambdas:
        backend._forecast_cache.clear()

        res = backend.simulate_strategy(df, '2007-01-01', '2026-01-01',
                                        lambda_penalty=lmbda, include_xgboost=True)
        if res.empty:
            continue

        ret, vol, sharpe, sortino, mdd = backend.calculate_metrics(
            res['Strat_Return'], res['RF_Rate'])

        signal = res['Forecast_State'].shift(1).fillna(0)
        pct_invested = (signal == 0).mean() * 100
        n_trades = int(signal.diff().abs().fillna(0).sum())

        print(f"Lambda={lmbda:>6.1f}: Sharpe={sharpe:.3f} Sortino={sortino:.3f} "
              f"Ret={ret*100:.1f}% MDD={mdd*100:.1f}% Trades={n_trades} Invested={pct_invested:.0f}%")

    # B&H baseline
    backend._forecast_cache.clear()
    res = backend.simulate_strategy(df, '2007-01-01', '2026-01-01', lambda_penalty=10.0, include_xgboost=True)
    bh_ret, bh_vol, bh_sharpe, bh_sortino, bh_mdd = backend.calculate_metrics(
        res['Target_Return'], res['RF_Rate'])
    print(f"\nB&H Baseline:  Sharpe={bh_sharpe:.3f} Sortino={bh_sortino:.3f} "
          f"Ret={bh_ret*100:.1f}% MDD={bh_mdd*100:.1f}%")


def test_xgb_prediction_calibration(df):
    """Test 5: Are XGBoost probabilities well-calibrated?"""
    print("\n" + "="*80)
    print("TEST 5: XGBOOST PREDICTION CALIBRATION")
    print("="*80)

    backend._forecast_cache.clear()

    results = []
    current_date = pd.to_datetime('2007-01-01')
    end_date = pd.to_datetime('2026-01-01')

    while current_date < end_date:
        res = backend.run_period_forecast(df, current_date, lambda_penalty=10.0, include_xgboost=True)
        if res is not None:
            results.append(res)
        current_date += pd.DateOffset(months=6)

    if not results:
        return

    full_res = pd.concat(results)

    probs = full_res['Raw_Prob']

    print(f"\nRaw probability statistics:")
    print(f"  Mean: {probs.mean():.3f}")
    print(f"  Std:  {probs.std():.3f}")
    print(f"  Min:  {probs.min():.3f}")
    print(f"  Max:  {probs.max():.3f}")
    print(f"  % > 0.5 (bearish): {(probs > 0.5).mean()*100:.1f}%")
    print(f"  % > 0.6 (strong bearish): {(probs > 0.6).mean()*100:.1f}%")
    print(f"  % < 0.4 (strong bullish): {(probs < 0.4).mean()*100:.1f}%")

    # Check if probabilities are clustered near 0.5 (low confidence)
    near_half = ((probs > 0.45) & (probs < 0.55)).mean()
    print(f"  % near 0.5 (±0.05, uncertain): {near_half*100:.1f}%")

    # Probability by decile vs actual forward returns
    print(f"\nProbability decile analysis (bearish prob vs next-day return):")
    full_res['Prob_Decile'] = pd.qcut(full_res['Raw_Prob'], 5, labels=False, duplicates='drop')
    for decile in sorted(full_res['Prob_Decile'].unique()):
        mask = full_res['Prob_Decile'] == decile
        avg_prob = full_res.loc[mask, 'Raw_Prob'].mean()
        avg_ret = full_res.loc[mask, 'Target_Return'].mean() * 252
        n = mask.sum()
        print(f"  Decile {decile}: avg_prob={avg_prob:.3f} avg_ann_ret={avg_ret*100:.1f}% n={n}")


def test_jm_only_vs_jm_xgb(df):
    """Test 6: Compare JM-only baseline vs JM-XGB."""
    print("\n" + "="*80)
    print("TEST 6: JM-ONLY vs JM-XGB COMPARISON")
    print("="*80)

    for strategy_name, use_xgb in [("JM-Only", False), ("JM-XGB", True)]:
        backend._forecast_cache.clear()

        res = backend.simulate_strategy(df, '2007-01-01', '2026-01-01',
                                        lambda_penalty=10.0, include_xgboost=use_xgb)
        if res.empty:
            continue

        ret, vol, sharpe, sortino, mdd = backend.calculate_metrics(
            res['Strat_Return'], res['RF_Rate'])

        signal = res['Forecast_State'].shift(1).fillna(0)
        pct_invested = (signal == 0).mean() * 100
        n_trades = int(signal.diff().abs().fillna(0).sum())

        print(f"{strategy_name:<10}: Sharpe={sharpe:.3f} Sortino={sortino:.3f} "
              f"Ret={ret*100:.1f}% Vol={vol*100:.1f}% MDD={mdd*100:.1f}% "
              f"Trades={n_trades} Invested={pct_invested:.0f}%")

    # B&H
    bh_ret, bh_vol, bh_sharpe, bh_sortino, bh_mdd = backend.calculate_metrics(
        res['Target_Return'], res['RF_Rate'])
    print(f"B&H       : Sharpe={bh_sharpe:.3f} Sortino={bh_sortino:.3f} "
          f"Ret={bh_ret*100:.1f}% Vol={bh_vol*100:.1f}% MDD={bh_mdd*100:.1f}%")


def test_threshold_sensitivity(df):
    """Test 7: Threshold sensitivity - is 0.5 optimal?"""
    print("\n" + "="*80)
    print("TEST 7: THRESHOLD SENSITIVITY")
    print("="*80)

    backend._forecast_cache.clear()

    # Get raw forecasts
    results = []
    current_date = pd.to_datetime('2007-01-01')
    end_date = pd.to_datetime('2026-01-01')

    while current_date < end_date:
        res = backend.run_period_forecast(df, current_date, lambda_penalty=10.0, include_xgboost=True)
        if res is not None:
            results.append(res)
        current_date += pd.DateOffset(months=6)

    if not results:
        return

    full_res = pd.concat(results)
    full_res['Smoothed_Prob'] = full_res['Raw_Prob'].ewm(halflife=8).mean()

    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

    for thresh in thresholds:
        full_res['Forecast_State'] = (full_res['Smoothed_Prob'] > thresh).astype(int)

        trading_signal = full_res['Forecast_State'].shift(1).fillna(0)
        strat_returns = np.where(trading_signal == 0, full_res['Target_Return'], full_res['RF_Rate'])
        trades = trading_signal.diff().abs().fillna(0)
        strat_returns = strat_returns - (trades.values * backend.TRANSACTION_COST)

        ret, vol, sharpe, sortino, mdd = backend.calculate_metrics(
            pd.Series(strat_returns, index=full_res.index), full_res['RF_Rate'])

        pct_invested = (trading_signal == 0).mean() * 100
        n_trades = int(trades.sum())

        print(f"Threshold={thresh:.2f}: Sharpe={sharpe:.3f} Sortino={sortino:.3f} "
              f"Ret={ret*100:.1f}% MDD={mdd*100:.1f}% Trades={n_trades} Invested={pct_invested:.0f}%")


def test_missed_opportunity_cost(df):
    """Test 8: Quantify the cost of false bear signals."""
    print("\n" + "="*80)
    print("TEST 8: MISSED OPPORTUNITY COST ANALYSIS")
    print("="*80)

    backend._forecast_cache.clear()

    res = backend.simulate_strategy(df, '2007-01-01', '2026-01-01',
                                    lambda_penalty=10.0, include_xgboost=True)
    if res.empty:
        return

    trading_signal = res['Forecast_State'].shift(1).fillna(0)

    # Days model was in risk-free but market went up
    missed_bull = (trading_signal == 1) & (res['Target_Return'] > 0)
    # Days model was invested but market went down
    caught_bear = (trading_signal == 1) & (res['Target_Return'] < 0)

    # Value of avoided losses
    avoided_loss = res.loc[caught_bear, 'Target_Return'].sum()
    # Cost of missed gains
    missed_gain = res.loc[missed_bull, 'Target_Return'].sum()

    print(f"\nDays in risk-free when market UP (missed gains): {missed_bull.sum()} days")
    print(f"Days in risk-free when market DOWN (avoided losses): {caught_bear.sum()} days")
    print(f"\nTotal missed gains: {missed_gain*100:.2f}% (cumulative)")
    print(f"Total avoided losses: {avoided_loss*100:.2f}% (cumulative)")
    print(f"Net impact: {(avoided_loss - missed_gain)*100:.2f}%")

    if missed_gain > abs(avoided_loss):
        print("\nVERDICT: False bear signals cost MORE than the value of correctly avoided losses.")
        print("This explains why B&H outperforms - the model enters risk-free too often during bull markets.")


def test_walk_forward_lambda_trace(df):
    """Test 9: Trace walk-forward lambda selections."""
    print("\n" + "="*80)
    print("TEST 9: WALK-FORWARD LAMBDA TRACE")
    print("="*80)

    current_date = pd.to_datetime('2007-01-01')
    final_end_date = pd.to_datetime('2026-01-01')

    print(f"\n{'Period Start':<14} {'Best Lambda':>12} {'Val Sharpe':>12} {'OOS 6m Ret':>12}")
    print("-" * 55)

    while current_date < final_end_date:
        backend._forecast_cache.clear()

        val_start = current_date - pd.DateOffset(years=5)

        best_sharpe = -np.inf
        best_lambda = 0.0

        for lmbda in backend.LAMBDA_GRID:
            val_res = backend.simulate_strategy(df, val_start, current_date, lmbda, include_xgboost=True)
            if not val_res.empty:
                _, _, sharpe, _, _ = backend.calculate_metrics(val_res['Strat_Return'], val_res['RF_Rate'])
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_lambda = lmbda

        # OOS performance with chosen lambda
        chunk_end = min(current_date + pd.DateOffset(months=6), final_end_date)
        oos_res = backend.simulate_strategy(df, current_date, chunk_end, best_lambda, include_xgboost=True)

        if not oos_res.empty:
            oos_ret = (1 + oos_res['Strat_Return']).prod() - 1
            bh_ret = (1 + oos_res['Target_Return']).prod() - 1
            winner = "***" if oos_ret > bh_ret else ""
            print(f"{current_date.date()} {best_lambda:>12.2f} {best_sharpe:>12.3f} "
                  f"{oos_ret*100:>11.2f}% (B&H: {bh_ret*100:.2f}%) {winner}")

        current_date += pd.DateOffset(months=6)


def test_xgb_hyperparameter_sensitivity(df):
    """Test 10: Does tuning XGBoost hyperparameters help?"""
    print("\n" + "="*80)
    print("TEST 10: XGBOOST HYPERPARAMETER SENSITIVITY")
    print("="*80)
    print("Paper uses default XGBoost hyperparameters.")
    print("Testing if conservative/aggressive settings change results.\n")

    from xgboost import XGBClassifier

    configs = {
        'Default': {},
        'Shallow (depth=3)': {'max_depth': 3, 'n_estimators': 100},
        'Deep (depth=10)': {'max_depth': 10, 'n_estimators': 200},
        'Conservative': {'max_depth': 3, 'n_estimators': 50, 'learning_rate': 0.05, 'subsample': 0.8},
        'Regularized': {'max_depth': 4, 'n_estimators': 100, 'learning_rate': 0.1,
                        'reg_alpha': 1.0, 'reg_lambda': 5.0, 'subsample': 0.8, 'colsample_bytree': 0.8},
    }

    for config_name, params in configs.items():
        backend._forecast_cache.clear()

        # We need to monkey-patch XGBClassifier to test different params
        # Run period forecasts manually
        results = []
        current_date = pd.to_datetime('2007-01-01')
        end_date = pd.to_datetime('2026-01-01')

        while current_date < end_date:
            train_start = current_date - pd.DateOffset(years=11)
            train_df = df[(df.index >= train_start) & (df.index < current_date)].copy()

            if len(train_df) < 252 * 5:
                current_date += pd.DateOffset(months=6)
                continue

            return_features = [c for c in train_df.columns if c.startswith(('DD_', 'Avg_Ret_', 'Sortino_'))]
            X_train_jm = train_df[return_features]
            X_train_jm = (X_train_jm - X_train_jm.mean()) / X_train_jm.std()

            jm = backend.StatisticalJumpModel(n_states=2, lambda_penalty=10.0)
            identified_states = jm.fit_predict(X_train_jm.values)

            cum_ret_0 = train_df['Excess_Return'][identified_states == 0].sum()
            cum_ret_1 = train_df['Excess_Return'][identified_states == 1].sum()
            if cum_ret_1 > cum_ret_0:
                identified_states = 1 - identified_states
                jm.means = jm.means[::-1].copy()

            train_df['Target_State'] = np.roll(identified_states, -1)
            train_df = train_df.iloc[:-1]

            oos_end = current_date + pd.DateOffset(months=6)
            oos_df = df[(df.index >= current_date) & (df.index < oos_end)].copy()

            if len(oos_df) == 0:
                current_date += pd.DateOffset(months=6)
                continue

            macro_features = ['Yield_2Y_EWMA_diff', 'Yield_Slope_EWMA_10',
                            'Yield_Slope_EWMA_diff_21', 'VIX_EWMA_log_diff', 'Stock_Bond_Corr']
            all_features = return_features + macro_features

            X_train_xgb = train_df[all_features]
            y_train_xgb = train_df['Target_State']
            X_oos_xgb = oos_df[all_features]

            xgb = XGBClassifier(eval_metric='logloss', random_state=42, **params)
            xgb.fit(X_train_xgb, y_train_xgb)

            oos_probs = xgb.predict_proba(X_oos_xgb)[:, 1]
            oos_df['Raw_Prob'] = oos_probs
            oos_df_out = oos_df[['Target_Return', 'RF_Rate', 'Raw_Prob']]
            results.append(oos_df_out)

            current_date += pd.DateOffset(months=6)

        if not results:
            continue

        full_res = pd.concat(results)
        full_res['State_Prob'] = full_res['Raw_Prob'].ewm(halflife=8).mean()
        full_res['Forecast_State'] = (full_res['State_Prob'] > 0.5).astype(int)

        trading_signal = full_res['Forecast_State'].shift(1).fillna(0)
        strat_returns = np.where(trading_signal == 0, full_res['Target_Return'], full_res['RF_Rate'])
        trades = trading_signal.diff().abs().fillna(0)
        strat_returns = strat_returns - (trades.values * backend.TRANSACTION_COST)

        ret, vol, sharpe, sortino, mdd = backend.calculate_metrics(
            pd.Series(strat_returns, index=full_res.index), full_res['RF_Rate'])

        pct_invested = (trading_signal == 0).mean() * 100
        n_trades = int(trades.sum())

        print(f"{config_name:<25}: Sharpe={sharpe:.3f} Sortino={sortino:.3f} "
              f"Ret={ret*100:.1f}% MDD={mdd*100:.1f}% Trades={n_trades} Invested={pct_invested:.0f}%")


if __name__ == '__main__':
    print("="*80)
    print("JM-XGB PERFORMANCE DIAGNOSTIC SUITE")
    print(f"OOS Period: 2007-01-01 to 2026-01-01 | Asset: {backend.TARGET_TICKER}")
    print("="*80)

    df = load_data()
    print(f"Data loaded: {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")

    # Run all tests
    test_signal_quality(df)
    test_subperiod_performance(df)
    test_ewma_sensitivity(df)
    test_lambda_sensitivity(df)
    test_xgb_prediction_calibration(df)
    test_jm_only_vs_jm_xgb(df)
    test_threshold_sensitivity(df)
    test_missed_opportunity_cost(df)
    # These are slower - run them last
    test_walk_forward_lambda_trace(df)
    test_xgb_hyperparameter_sensitivity(df)

    print("\n" + "="*80)
    print("DIAGNOSTIC SUITE COMPLETE")
    print("="*80)
