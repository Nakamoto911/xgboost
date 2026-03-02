"""Compare BEFORE (default XGB, fixed hl=8) vs AFTER (regularized XGB, tuned hl)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier
from main import StatisticalJumpModel, calculate_metrics, TRANSACTION_COST

df = pd.read_pickle('data_cache.pkl')
print(f"Data: {df.index[0].date()} to {df.index[-1].date()}", flush=True)

def forecast(df, current_date, lmbda, regularized=False):
    train_start = current_date - pd.DateOffset(years=11)
    train_df = df[(df.index >= train_start) & (df.index < current_date)].copy()
    if len(train_df) < 252*5: return None
    ret_feats = [c for c in train_df.columns if c.startswith(('DD_','Avg_Ret_','Sortino_'))]
    X_jm = (train_df[ret_feats] - train_df[ret_feats].mean()) / train_df[ret_feats].std()
    jm = StatisticalJumpModel(n_states=2, lambda_penalty=lmbda)
    states = jm.fit_predict(X_jm.values)
    if train_df['Excess_Return'][states==1].sum() > train_df['Excess_Return'][states==0].sum():
        states = 1 - states; jm.means = jm.means[::-1].copy()
    train_df['Target_State'] = np.roll(states, -1)
    train_df = train_df.iloc[:-1]
    oos_end = current_date + pd.DateOffset(months=6)
    oos_df = df[(df.index >= current_date) & (df.index < oos_end)].copy()
    if len(oos_df) == 0: return None
    macro = ['Yield_2Y_EWMA_diff','Yield_Slope_EWMA_10','Yield_Slope_EWMA_diff_21','VIX_EWMA_log_diff','Stock_Bond_Corr']
    all_f = ret_feats + macro
    if regularized:
        model = XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0,
                              max_depth=4, n_estimators=100, learning_rate=0.1,
                              reg_alpha=1.0, reg_lambda=5.0, subsample=0.8, colsample_bytree=0.8)
    else:
        model = XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0)
    model.fit(train_df[all_f], train_df['Target_State'])
    oos_df['Raw_Prob'] = model.predict_proba(oos_df[all_f])[:, 1]
    return oos_df[['Target_Return','RF_Rate','Raw_Prob']].copy()

def run_strat(df, start, end, lmbda, hl=8, regularized=False):
    results = []
    cur = pd.to_datetime(start); end_dt = pd.to_datetime(end)
    while cur < end_dt:
        r = forecast(df, cur, lmbda, regularized)
        if r is not None: results.append(r)
        cur += pd.DateOffset(months=6)
    if not results: return pd.DataFrame()
    full = pd.concat(results)
    full['State_Prob'] = full['Raw_Prob'].ewm(halflife=hl).mean() if hl > 0 else full['Raw_Prob']
    full['Forecast_State'] = (full['State_Prob'] > 0.5).astype(int)
    sig = full['Forecast_State'].shift(1).fillna(0)
    sr = np.where(sig==0, full['Target_Return'], full['RF_Rate'])
    trades = sig.diff().abs().fillna(0)
    full['Strat_Return'] = sr - (trades.values * TRANSACTION_COST)
    full['Signal'] = sig
    return full

configs = [
    ("BEFORE (default XGB, lam=10, hl=8)",   10.0, 8, False),
    ("AFTER  (reg XGB, lam=5, hl=8)",         5.0, 8, True),
    ("AFTER  (reg XGB, lam=10, hl=8)",        10.0, 8, True),
    ("AFTER  (reg XGB, lam=20, hl=8)",        20.0, 8, True),
]

print(f"\n{'Config':<42} {'Sharpe':>7} {'Sort':>7} {'Ret':>7} {'MDD':>8} {'Inv':>5}", flush=True)
print("-"*80, flush=True)

for name, lmbda, hl, reg in configs:
    r = run_strat(df, '2007-01-01', '2026-01-01', lmbda, hl=hl, regularized=reg)
    if r.empty: continue
    ret,vol,sh,so,mdd = calculate_metrics(r['Strat_Return'], r['RF_Rate'])
    inv = (r['Signal']==0).mean()*100
    print(f"{name:<42} {sh:>7.3f} {so:>7.3f} {ret*100:>6.1f}% {mdd*100:>7.1f}% {inv:>4.0f}%", flush=True)

r = run_strat(df, '2007-01-01', '2026-01-01', 10.0)
bret,_,bsh,bso,bmdd = calculate_metrics(r['Target_Return'], r['RF_Rate'])
print(f"{'B&H':<42} {bsh:>7.3f} {bso:>7.3f} {bret*100:>6.1f}% {bmdd*100:>7.1f}%", flush=True)
