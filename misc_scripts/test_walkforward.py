"""Walk-forward lambda trace test."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier
from main import StatisticalJumpModel, calculate_metrics, TRANSACTION_COST

df = pd.read_pickle('data_cache.pkl')
LAMBDA_GRID = [0.0] + list(np.logspace(0, 2, 10))  # Reduced grid for speed

def forecast(df, current_date, lmbda):
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
    model = XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0)
    model.fit(train_df[all_f], train_df['Target_State'])
    oos_df['Raw_Prob'] = model.predict_proba(oos_df[all_f])[:, 1]
    return oos_df[['Target_Return','RF_Rate','Raw_Prob']].copy()

def run_strat(df, start, end, lmbda, hl=8):
    results = []
    cur = pd.to_datetime(start); end_dt = pd.to_datetime(end)
    while cur < end_dt:
        r = forecast(df, cur, lmbda)
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

print('WALK-FORWARD LAMBDA TRACE (2007-2026)', flush=True)
cur = pd.to_datetime('2007-01-01'); end_dt = pd.to_datetime('2026-01-01')
print(f"{'Start':<12} {'lam':>6} {'ValSh':>7} {'OOS':>8} {'B&H':>8} {'Win':>4}", flush=True)
print('-'*50, flush=True)
wins = 0; total = 0
while cur < end_dt:
    val_start = cur - pd.DateOffset(years=5)
    best_sh = -np.inf; best_lam = 0.0
    for lam in LAMBDA_GRID:
        vr = run_strat(df, val_start, cur, lam)
        if not vr.empty:
            _,_,sh,_,_ = calculate_metrics(vr['Strat_Return'], vr['RF_Rate'])
            if sh > best_sh: best_sh = sh; best_lam = lam
    chunk_end = min(cur + pd.DateOffset(months=6), end_dt)
    oos = run_strat(df, cur, chunk_end, best_lam)
    if not oos.empty:
        oos_ret = (1+oos['Strat_Return']).prod()-1
        bh_ret = (1+oos['Target_Return']).prod()-1
        total += 1
        w = '*' if oos_ret > bh_ret else ''
        if w: wins += 1
        print(f'{cur.date()} {best_lam:>6.1f} {best_sh:>7.3f} {oos_ret*100:>7.2f}% {bh_ret*100:>7.2f}% {w:>4}', flush=True)
    cur += pd.DateOffset(months=6)
print(f'Periods won: {wins}/{total} ({100*wins/total:.0f}%)', flush=True)
