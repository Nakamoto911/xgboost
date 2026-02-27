import pandas as pd
try:
    df = pd.read_pickle('backtest_cache.pkl')
    if isinstance(df, dict):
        jm_xgb_df = df.get('jm_xgb_df')
    else:
        jm_xgb_df = df
    
    if jm_xgb_df is not None:
        shap_cols = [c for c in jm_xgb_df.columns if c.startswith('SHAP_') and c != 'SHAP_Base_Value']
        missing_shap = jm_xgb_df[shap_cols].isna().any(axis=1)
        print("Total rows:", len(jm_xgb_df))
        print("Rows with missing SHAP:", missing_shap.sum())
        if missing_shap.sum() > 0:
            print("Missing SHAP from:", jm_xgb_df[missing_shap].index.min(), "to", jm_xgb_df[missing_shap].index.max())
            print("Present SHAP from:", jm_xgb_df[~missing_shap].index.min(), "to", jm_xgb_df[~missing_shap].index.max())
except Exception as e:
    print("Error:", e)
