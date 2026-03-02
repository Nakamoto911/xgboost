"""
Fast diagnostic suite - bypasses SHAP computation which is the bottleneck.
Runs a custom forecast loop without SHAP to diagnose signal quality issues.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier

# Import only what we need from main
from main import StatisticalJumpModel, calculate_metrics, TRANSACTION_COST

TARGET_TICKER = '^SP500TR'
OOS_START = '2007-01-01'
OOS_END = '2026-01-01'
LAMBDA_GRID = [0.0] + list(np.logspace(0, 2, 20))


def load_data():
    df = pd.read_pickle('data_cache.pkl')
    print(f"Data: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
    return df


def forecast_no_shap(df, current_date, lambda_penalty, include_xgboost=True):
    """Like run_period_forecast but without SHAP (10x faster)."""
    train_start = current_date - pd.DateOffset(years=11)
    train_df = df[(df.index >= train_start) & (df.index < current_date)].copy()
    if len(train_df) < 252 * 5:
        return None

    return_features = [c for c in train_df.columns if c.startswith(('DD_', 'Avg_Ret_', 'Sortino_'))]
    X_train_jm = train_df[return_features]
    X_train_jm_z = (X_train_jm - X_train_jm.mean()) / X_train_jm.std()

    jm = StatisticalJumpModel(n_states=2, lambda_penalty=lambda_penalty)
    identified_states = jm.fit_predict(X_train_jm_z.values)

    # Align states
    cum_ret_0 = train_df['Excess_Return'][identified_states == 0].sum()
    cum_ret_1 = train_df['Excess_Return'][identified_states == 1].sum()
    if cum_ret_1 > cum_ret_0:
        identified_states = 1 - identified_states
        jm.means = jm.means[::-1].copy()

    # Shift labels (predict t+1 from t)
    train_df['Target_State'] = np.roll(identified_states, -1)
    train_df = train_df.iloc[:-1]

    oos_end = current_date + pd.DateOffset(months=6)
    oos_df = df[(df.index >= current_date) & (df.index < oos_end)].copy()
    if len(oos_df) == 0:
        return None

    if not include_xgboost:
        X_oos_jm = oos_df[return_features]
        X_oos_jm = (X_oos_jm - train_df[return_features].mean()) / train_df[return_features].std()
        oos_states = jm.predict_online(X_oos_jm.values, last_known_state=identified_states[-1])
        oos_df['Forecast_State'] = oos_states
        return oos_df[['Target_Return', 'RF_Rate', 'Forecast_State']].copy()

    macro_features = ['Yield_2Y_EWMA_diff', 'Yield_Slope_EWMA_10',
                      'Yield_Slope_EWMA_diff_21', 'VIX_EWMA_log_diff', 'Stock_Bond_Corr']
    all_features = return_features + macro_features

    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb.fit(train_df[all_features], train_df['Target_State'])

    oos_probs = xgb.predict_proba(oos_df[all_features])[:, 1]
    oos_df['Raw_Prob'] = oos_probs

    # Also get JM online for comparison
    X_oos_jm = oos_df[return_features]
    X_oos_jm = (X_oos_jm - train_df[return_features].mean()) / train_df[return_features].std()
    oos_states = jm.predict_online(X_oos_jm.values, last_known_state=identified_states[-1])
    oos_df['JM_State'] = oos_states

    return oos_df[['Target_Return', 'RF_Rate', 'Raw_Prob', 'JM_State']].copy()


def run_strategy(df, start, end, lmbda, include_xgboost=True, ewma_hl=8, threshold=0.5):
    """Run the full strategy with configurable EWMA halflife and threshold."""
    results = []
    current = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    while current < end_dt:
        res = forecast_no_shap(df, current, lmbda, include_xgboost)
        if res is not None:
            results.append(res)
        current += pd.DateOffset(months=6)

    if not results:
        return pd.DataFrame()

    full = pd.concat(results)

    if include_xgboost:
        if ewma_hl == 0:
            full['State_Prob'] = full['Raw_Prob']
        else:
            full['State_Prob'] = full['Raw_Prob'].ewm(halflife=ewma_hl).mean()
        full['Forecast_State'] = (full['State_Prob'] > threshold).astype(int)

    signal = full['Forecast_State'].shift(1).fillna(0)
    strat_ret = np.where(signal == 0, full['Target_Return'], full['RF_Rate'])
    trades = signal.diff().abs().fillna(0)
    full['Strat_Return'] = strat_ret - (trades.values * TRANSACTION_COST)
    full['Signal'] = signal

    return full


# ==================== TESTS ====================

def test1_signal_quality(df):
    print("\n" + "="*80)
    print("TEST 1: SIGNAL QUALITY")
    print("="*80)

    res = run_strategy(df, OOS_START, OOS_END, lmbda=10.0)
    if res.empty:
        return

    signal = res['Signal']
    total = len(res)
    bull = (signal == 0).sum()
    bear = (signal == 1).sum()

    print(f"Total days: {total}")
    print(f"Bull (invested): {bull} ({100*bull/total:.1f}%)")
    print(f"Bear (risk-free): {bear} ({100*bear/total:.1f}%)")

    # Returns during each signal
    bull_mask = signal == 0
    bear_mask = signal == 1
    ret_when_bull = res.loc[bull_mask, 'Target_Return'].mean() * 252
    ret_when_bear = res.loc[bear_mask, 'Target_Return'].mean() * 252

    print(f"\nMarket return when model says BULL: {ret_when_bull*100:.2f}% ann.")
    print(f"Market return when model says BEAR: {ret_when_bear*100:.2f}% ann.")

    if ret_when_bear > 0:
        print(">>> WARNING: Market goes UP when model says BEAR! Signals destroy value.")

    switches = signal.diff().abs().sum()
    print(f"\nSwitches: {int(switches)} ({switches/(total/252):.1f}/year)")

    return res


def test2_subperiods(df):
    print("\n" + "="*80)
    print("TEST 2: SUB-PERIOD DECOMPOSITION")
    print("="*80)

    res = run_strategy(df, OOS_START, OOS_END, lmbda=10.0)
    if res.empty:
        return

    periods = [
        ('GFC (2007-2009)',           '2007-01-01', '2009-03-09'),
        ('Recovery (2009-2015)',       '2009-03-09', '2015-01-01'),
        ('Bull (2015-2020)',          '2015-01-01', '2020-02-19'),
        ('COVID Crash',              '2020-02-19', '2020-03-23'),
        ('COVID Recovery',           '2020-03-23', '2021-01-01'),
        ('2021 Bull',                '2021-01-01', '2022-01-01'),
        ('2022 Bear',                '2022-01-01', '2022-10-12'),
        ('2023 Recovery',            '2023-01-01', '2024-01-01'),
        ('2024 Bull',                '2024-01-01', '2025-01-01'),
        ('2025',                     '2025-01-01', '2026-01-01'),
    ]

    print(f"\n{'Period':<28} {'B&H':>10} {'JM-XGB':>10} {'%Invested':>10} {'Winner':>8}")
    print("-" * 70)

    for name, s, e in periods:
        mask = (res.index >= s) & (res.index < e)
        if mask.sum() == 0:
            continue
        p = res[mask]
        bh = (1 + p['Target_Return']).prod() - 1
        st = (1 + p['Strat_Return']).prod() - 1
        inv = (p['Signal'] == 0).mean() * 100
        w = "JM-XGB" if st > bh else "B&H"
        print(f"{name:<28} {bh*100:>9.1f}% {st*100:>9.1f}% {inv:>9.0f}% {w:>8}")


def test3_ewma_sensitivity(df):
    print("\n" + "="*80)
    print("TEST 3: EWMA HALFLIFE SENSITIVITY")
    print("="*80)
    print("Paper says tune from {0, 2, 4, 8}; current code hardcodes 8")

    for hl in [0, 1, 2, 4, 8, 16]:
        res = run_strategy(df, OOS_START, OOS_END, lmbda=10.0, ewma_hl=hl)
        if res.empty:
            continue
        ret, vol, sharpe, sortino, mdd = calculate_metrics(res['Strat_Return'], res['RF_Rate'])
        inv = (res['Signal'] == 0).mean() * 100
        trades = int(res['Signal'].diff().abs().fillna(0).sum())
        print(f"hl={hl:>2}: Sharpe={sharpe:.3f} Sort={sortino:.3f} Ret={ret*100:.1f}% "
              f"MDD={mdd*100:.1f}% Trades={trades} Inv={inv:.0f}%")

    # B&H baseline
    res = run_strategy(df, OOS_START, OOS_END, lmbda=10.0)
    bh_ret, bh_vol, bh_sharpe, bh_sortino, bh_mdd = calculate_metrics(
        res['Target_Return'], res['RF_Rate'])
    print(f"B&H:  Sharpe={bh_sharpe:.3f} Sort={bh_sortino:.3f} Ret={bh_ret*100:.1f}% "
          f"MDD={bh_mdd*100:.1f}%")


def test4_lambda_sensitivity(df):
    print("\n" + "="*80)
    print("TEST 4: LAMBDA SENSITIVITY")
    print("="*80)

    for lmbda in [0.0, 1.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        res = run_strategy(df, OOS_START, OOS_END, lmbda=lmbda)
        if res.empty:
            continue
        ret, vol, sharpe, sortino, mdd = calculate_metrics(res['Strat_Return'], res['RF_Rate'])
        inv = (res['Signal'] == 0).mean() * 100
        trades = int(res['Signal'].diff().abs().fillna(0).sum())
        print(f"λ={lmbda:>6.1f}: Sharpe={sharpe:.3f} Sort={sortino:.3f} Ret={ret*100:.1f}% "
              f"MDD={mdd*100:.1f}% Trades={trades} Inv={inv:.0f}%")

    # B&H
    bh_ret, bh_vol, bh_sharpe, bh_sortino, bh_mdd = calculate_metrics(
        res['Target_Return'], res['RF_Rate'])
    print(f"B&H:     Sharpe={bh_sharpe:.3f} Sort={bh_sortino:.3f} Ret={bh_ret*100:.1f}% "
          f"MDD={bh_mdd*100:.1f}%")


def test5_xgb_calibration(df):
    print("\n" + "="*80)
    print("TEST 5: XGB PROBABILITY CALIBRATION")
    print("="*80)

    res = run_strategy(df, OOS_START, OOS_END, lmbda=10.0)
    if res.empty:
        return

    probs = res['State_Prob']
    print(f"Smoothed prob stats: mean={probs.mean():.3f} std={probs.std():.3f} "
          f"min={probs.min():.3f} max={probs.max():.3f}")
    print(f"% > 0.5 (bearish): {(probs > 0.5).mean()*100:.1f}%")
    print(f"% in [0.45, 0.55] (uncertain): {((probs > 0.45) & (probs < 0.55)).mean()*100:.1f}%")

    # Check raw prob
    raw = res['Raw_Prob']
    print(f"\nRaw prob stats: mean={raw.mean():.3f} std={raw.std():.3f} "
          f"min={raw.min():.3f} max={raw.max():.3f}")
    print(f"Raw % > 0.5: {(raw > 0.5).mean()*100:.1f}%")

    # Quintile analysis
    print(f"\nQuintile analysis (raw prob vs forward return):")
    res['Q'] = pd.qcut(res['Raw_Prob'], 5, labels=False, duplicates='drop')
    for q in sorted(res['Q'].unique()):
        m = res['Q'] == q
        p_avg = res.loc[m, 'Raw_Prob'].mean()
        r_avg = res.loc[m, 'Target_Return'].mean() * 252
        n = m.sum()
        print(f"  Q{q}: prob={p_avg:.3f} ann_ret={r_avg*100:.1f}% n={n}")


def test6_jm_vs_jmxgb(df):
    print("\n" + "="*80)
    print("TEST 6: JM-ONLY vs JM-XGB")
    print("="*80)

    for name, xgb in [("JM-Only", False), ("JM-XGB", True)]:
        res = run_strategy(df, OOS_START, OOS_END, lmbda=10.0, include_xgboost=xgb)
        if res.empty:
            continue
        ret, vol, sharpe, sortino, mdd = calculate_metrics(res['Strat_Return'], res['RF_Rate'])
        inv = (res['Signal'] == 0).mean() * 100
        trades = int(res['Signal'].diff().abs().fillna(0).sum())
        print(f"{name:<10}: Sharpe={sharpe:.3f} Sort={sortino:.3f} Ret={ret*100:.1f}% "
              f"Vol={vol*100:.1f}% MDD={mdd*100:.1f}% Trades={trades} Inv={inv:.0f}%")

    bh_ret, _, bh_sharpe, bh_sortino, bh_mdd = calculate_metrics(
        res['Target_Return'], res['RF_Rate'])
    print(f"B&H       : Sharpe={bh_sharpe:.3f} Sort={bh_sortino:.3f} Ret={bh_ret*100:.1f}% MDD={bh_mdd*100:.1f}%")


def test7_threshold(df):
    print("\n" + "="*80)
    print("TEST 7: THRESHOLD SENSITIVITY")
    print("="*80)

    for thresh in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        res = run_strategy(df, OOS_START, OOS_END, lmbda=10.0, threshold=thresh)
        if res.empty:
            continue
        ret, vol, sharpe, sortino, mdd = calculate_metrics(res['Strat_Return'], res['RF_Rate'])
        inv = (res['Signal'] == 0).mean() * 100
        print(f"thresh={thresh:.2f}: Sharpe={sharpe:.3f} Sort={sortino:.3f} Ret={ret*100:.1f}% "
              f"MDD={mdd*100:.1f}% Inv={inv:.0f}%")


def test8_missed_opportunity(df):
    print("\n" + "="*80)
    print("TEST 8: MISSED OPPORTUNITY COST")
    print("="*80)

    res = run_strategy(df, OOS_START, OOS_END, lmbda=10.0)
    if res.empty:
        return

    signal = res['Signal']
    missed_bull = (signal == 1) & (res['Target_Return'] > 0)
    caught_bear = (signal == 1) & (res['Target_Return'] < 0)

    missed_gain = res.loc[missed_bull, 'Target_Return'].sum()
    avoided_loss = res.loc[caught_bear, 'Target_Return'].sum()

    print(f"Days risk-free when mkt UP: {missed_bull.sum()}")
    print(f"Days risk-free when mkt DOWN: {caught_bear.sum()}")
    print(f"Missed gains (cum): {missed_gain*100:.2f}%")
    print(f"Avoided losses (cum): {avoided_loss*100:.2f}%")
    print(f"Net: {(avoided_loss - missed_gain)*100:.2f}%")

    if missed_gain > abs(avoided_loss):
        print(">>> VERDICT: False bear signals cost MORE than avoided losses!")


def test9_xgb_hyperparams(df):
    print("\n" + "="*80)
    print("TEST 9: XGB HYPERPARAMETER CONFIGS")
    print("="*80)

    configs = {
        'Default': {},
        'Shallow d=3': {'max_depth': 3, 'n_estimators': 100},
        'Deep d=10': {'max_depth': 10, 'n_estimators': 200},
        'Conservative': {'max_depth': 3, 'n_estimators': 50, 'learning_rate': 0.05, 'subsample': 0.8},
        'Regularized': {'max_depth': 4, 'n_estimators': 100, 'learning_rate': 0.1,
                        'reg_alpha': 1.0, 'reg_lambda': 5.0, 'subsample': 0.8, 'colsample_bytree': 0.8},
    }

    for cname, params in configs.items():
        results = []
        current = pd.to_datetime(OOS_START)
        end_dt = pd.to_datetime(OOS_END)

        while current < end_dt:
            train_start = current - pd.DateOffset(years=11)
            train_df = df[(df.index >= train_start) & (df.index < current)].copy()
            if len(train_df) < 252 * 5:
                current += pd.DateOffset(months=6)
                continue

            return_features = [c for c in train_df.columns if c.startswith(('DD_', 'Avg_Ret_', 'Sortino_'))]
            X_jm = (train_df[return_features] - train_df[return_features].mean()) / train_df[return_features].std()

            jm = StatisticalJumpModel(n_states=2, lambda_penalty=10.0)
            states = jm.fit_predict(X_jm.values)

            if train_df['Excess_Return'][states == 1].sum() > train_df['Excess_Return'][states == 0].sum():
                states = 1 - states
                jm.means = jm.means[::-1].copy()

            train_df['Target_State'] = np.roll(states, -1)
            train_df = train_df.iloc[:-1]

            oos_end = current + pd.DateOffset(months=6)
            oos_df = df[(df.index >= current) & (df.index < oos_end)].copy()
            if len(oos_df) == 0:
                current += pd.DateOffset(months=6)
                continue

            macro_features = ['Yield_2Y_EWMA_diff', 'Yield_Slope_EWMA_10',
                            'Yield_Slope_EWMA_diff_21', 'VIX_EWMA_log_diff', 'Stock_Bond_Corr']
            all_features = return_features + macro_features

            model = XGBClassifier(eval_metric='logloss', random_state=42, **params)
            model.fit(train_df[all_features], train_df['Target_State'])
            oos_df['Raw_Prob'] = model.predict_proba(oos_df[all_features])[:, 1]
            results.append(oos_df[['Target_Return', 'RF_Rate', 'Raw_Prob']].copy())
            current += pd.DateOffset(months=6)

        if not results:
            continue

        full = pd.concat(results)
        full['State_Prob'] = full['Raw_Prob'].ewm(halflife=8).mean()
        full['Forecast_State'] = (full['State_Prob'] > 0.5).astype(int)
        signal = full['Forecast_State'].shift(1).fillna(0)
        sr = np.where(signal == 0, full['Target_Return'], full['RF_Rate'])
        trades = signal.diff().abs().fillna(0)
        sr = sr - (trades.values * TRANSACTION_COST)

        ret, vol, sharpe, sortino, mdd = calculate_metrics(
            pd.Series(sr, index=full.index), full['RF_Rate'])
        inv = (signal == 0).mean() * 100
        nt = int(trades.sum())
        print(f"{cname:<15}: Sharpe={sharpe:.3f} Sort={sortino:.3f} Ret={ret*100:.1f}% "
              f"MDD={mdd*100:.1f}% Trades={nt} Inv={inv:.0f}%")


def test10_walkforward_trace(df):
    print("\n" + "="*80)
    print("TEST 10: WALK-FORWARD LAMBDA TRACE")
    print("="*80)

    current = pd.to_datetime(OOS_START)
    end_dt = pd.to_datetime(OOS_END)

    print(f"\n{'Start':<12} {'λ':>6} {'ValSharpe':>10} {'OOS Ret':>10} {'B&H Ret':>10} {'Win':>5}")
    print("-" * 60)

    while current < end_dt:
        val_start = current - pd.DateOffset(years=5)
        best_sharpe = -np.inf
        best_lmbda = 0.0

        for lmbda in LAMBDA_GRID:
            vr = run_strategy(df, val_start, current, lmbda)
            if not vr.empty:
                _, _, sh, _, _ = calculate_metrics(vr['Strat_Return'], vr['RF_Rate'])
                if sh > best_sharpe:
                    best_sharpe = sh
                    best_lmbda = lmbda

        chunk_end = min(current + pd.DateOffset(months=6), end_dt)
        oos = run_strategy(df, current, chunk_end, best_lmbda)

        if not oos.empty:
            oos_ret = (1 + oos['Strat_Return']).prod() - 1
            bh_ret = (1 + oos['Target_Return']).prod() - 1
            win = "*" if oos_ret > bh_ret else ""
            print(f"{current.date()} {best_lmbda:>6.1f} {best_sharpe:>10.3f} "
                  f"{oos_ret*100:>9.2f}% {bh_ret*100:>9.2f}% {win:>5}")

        current += pd.DateOffset(months=6)


if __name__ == '__main__':
    import time
    print("="*80)
    print("JM-XGB FAST DIAGNOSTIC SUITE")
    print(f"Period: {OOS_START} to {OOS_END}")
    print("="*80)

    df = load_data()

    t0 = time.time()

    # Fast tests first (single runs)
    test1_signal_quality(df)
    print(f"  [{time.time()-t0:.0f}s elapsed]")

    test2_subperiods(df)
    print(f"  [{time.time()-t0:.0f}s elapsed]")

    test5_xgb_calibration(df)
    print(f"  [{time.time()-t0:.0f}s elapsed]")

    test6_jm_vs_jmxgb(df)
    print(f"  [{time.time()-t0:.0f}s elapsed]")

    test8_missed_opportunity(df)
    print(f"  [{time.time()-t0:.0f}s elapsed]")

    # Medium tests (multiple single runs)
    test3_ewma_sensitivity(df)
    print(f"  [{time.time()-t0:.0f}s elapsed]")

    test4_lambda_sensitivity(df)
    print(f"  [{time.time()-t0:.0f}s elapsed]")

    test7_threshold(df)
    print(f"  [{time.time()-t0:.0f}s elapsed]")

    test9_xgb_hyperparams(df)
    print(f"  [{time.time()-t0:.0f}s elapsed]")

    # Slow test (walk-forward with full lambda grid)
    test10_walkforward_trace(df)
    print(f"  [{time.time()-t0:.0f}s elapsed]")

    print(f"\n{'='*80}")
    print(f"DONE in {time.time()-t0:.0f}s")
    print("="*80)
