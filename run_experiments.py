import pandas as pd
from main import fetch_and_prepare_data, walk_forward_backtest, calculate_metrics
from config import StrategyConfig

def run_all_experiments():
    df = fetch_and_prepare_data()
    
    configs = [
        StrategyConfig(name="1. Paper Baseline"),
        StrategyConfig(name="2. Sortino Tuned", tuning_metric="sortino"),
        StrategyConfig(name="3. Conservative Threshold (0.6)", prob_threshold=0.60),
        StrategyConfig(name="4. Continuous Allocation", allocation_style="continuous"),
        StrategyConfig(name="5. Lambda Smoothing", lambda_smoothing=True),
        StrategyConfig(name="6. Expanding Window", validation_window_type="expanding"),
        StrategyConfig(name="7. Lambda Ensemble (Top 3)", lambda_ensemble_k=3),
        StrategyConfig(
            name="8. The Ultimate Combo", 
            tuning_metric="sortino",
            prob_threshold=0.55,
            allocation_style="continuous",
            lambda_smoothing=True
        )
    ]
    
    results = {}
    for config in configs:
        print(f"Running Experiment: {config.name}...")
        res_df = walk_forward_backtest(df, config) 
        ret, vol, sharpe, sortino, mdd = calculate_metrics(res_df['Strat_Return'], res_df['RF_Rate'])
        results[config.name] = {
            "Ann Ret (%)": round(ret * 100, 2),
            "Ann Vol (%)": round(vol * 100, 2),
            "Sharpe": round(sharpe, 3),
            "Sortino": round(sortino, 3),
            "Max DD (%)": round(mdd * 100, 2)
        }
    
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS")
    print("="*80)
    results_df = pd.DataFrame(results).T
    print(results_df.to_string())

if __name__ == "__main__":
    run_all_experiments()
