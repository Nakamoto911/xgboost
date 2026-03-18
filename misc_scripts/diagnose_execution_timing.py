import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf

# Add parent directory to path to import main and config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import fetch_and_prepare_data, walk_forward_backtest, calculate_metrics, TARGET_TICKER, TRANSACTION_COST
from config import StrategyConfig

def fetch_open_prices(start_date, end_date):
    """Fetches just the Open prices to evaluate execution slippage."""
    print(f"Fetching Open prices for {TARGET_TICKER}...")
    ticker_data = yf.download(TARGET_TICKER, start=start_date, end=end_date, auto_adjust=False)
    
    if isinstance(ticker_data.columns, pd.MultiIndex):
        open_prices = ticker_data['Open'].iloc[:, 0].rename('Open')
        close_prices = ticker_data['Close'].iloc[:, 0].rename('Close')
    else:
        open_prices = ticker_data['Open'].rename('Open')
        close_prices = ticker_data['Close'].rename('Close')
        
    df = pd.concat([open_prices, close_prices], axis=1)
    # Intraday: Buy at Open, hold to Close
    df['Target_Intraday_Ret'] = (df['Close'] - df['Open']) / df['Open']
    # Overnight: Hold from Prev Close, sell at Open
    df['Target_Overnight_Ret'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    return df.fillna(0)

def evaluate_execution(df, execution_data, name):
    """Calculates performance based on theoretical vs realistic execution."""
    
    # 1. Theoretical Execution (Current main.py logic)
    # Signal is shifted by 1. Assumes we trade exactly at the close.
    tradable_signal = df['Forecast_State'].shift(1).fillna(0)
    alloc_target = 1.0 - tradable_signal
    trades = alloc_target.diff().abs().fillna(0)
    
    theo_returns = (alloc_target * df['Target_Return']) + ((1.0 - alloc_target) * df['RF_Rate'])
    theo_strat_ret = theo_returns - (trades * TRANSACTION_COST)
    
    # 2. Next-Open Execution (Realistic logic)
    # Merge the open/intraday returns we fetched
    df_eval = df.join(execution_data[['Target_Intraday_Ret', 'Target_Overnight_Ret']], how='left').fillna(0)
    trade_direction = alloc_target.diff().fillna(0)
    
    market_returns = df_eval['Target_Return']
    
    # If we just bought the market (alloc changes 0 -> 1, diff is 1)
    # We execute at the Open. We only capture the Intraday return.
    market_returns = np.where(trade_direction == 1, 
                              df_eval['Target_Intraday_Ret'], 
                              market_returns)
                              
    # If we just sold the market (alloc changes 1 -> 0, diff is -1)
    # We execute at the Open. We only capture the Overnight return.
    market_returns = np.where(trade_direction == -1, 
                              df_eval['Target_Overnight_Ret'], 
                              market_returns)
                              
    real_returns = (alloc_target * market_returns) + ((1.0 - alloc_target) * df['RF_Rate'])
    real_strat_ret = real_returns - (trades * TRANSACTION_COST)
    
    # Calculate Metrics
    t_ret, t_vol, t_sharpe, _, t_mdd = calculate_metrics(theo_strat_ret, df['RF_Rate'])
    r_ret, r_vol, r_sharpe, _, r_mdd = calculate_metrics(real_strat_ret, df['RF_Rate'])
    
    print(f"\n{'='*50}")
    print(f" PRESET: {name}")
    print(f"{'='*50}")
    print(f"{'Metric':<20} | {'Theoretical (Close)':<20} | {'Realistic (Next-Open)':<20} | {'Delta'}")
    print(f"-" * 75)
    print(f"{'Ann. Return':<20} | {t_ret*100:>19.2f}% | {r_ret*100:>20.2f}% | {(r_ret-t_ret)*100:>8.2f}%")
    print(f"{'Sharpe Ratio':<20} | {t_sharpe:>19.2f}  | {r_sharpe:>20.2f}  | {r_sharpe-t_sharpe:>8.2f}")
    print(f"{'Max Drawdown':<20} | {t_mdd*100:>19.2f}% | {r_mdd*100:>20.2f}% | {(r_mdd-t_mdd)*100:>8.2f}%")
    print(f"{'Total Trades':<20} | {int(trades.sum()):>19}  | {int(trades.sum()):>20}  | {'-':>8}")

if __name__ == "__main__":
    print("Loading data...")
    main_df = fetch_and_prepare_data()
    start_date = main_df.index.min().strftime('%Y-%m-%d')
    end_date = main_df.index.max().strftime('%Y-%m-%d')
    exec_data = fetch_open_prices(start_date, end_date)
    
    # Preset 1: Paper Baseline
    cfg_baseline = StrategyConfig(
        name="Paper Baseline",
        tuning_metric="sharpe",
        lambda_selection="best",
        lambda_subwindow_consensus=False,
        # Uses the default dense 8-pt lambda grid
    )
    
    print("\nRunning Backtest for Paper Baseline...")
    res_baseline = walk_forward_backtest(main_df, cfg_baseline)
    
    # Preset 2: Optimized
    cfg_optimized = StrategyConfig(
        name="Optimized",
        tuning_metric="sharpe",
        lambda_selection="best",
        lambda_subwindow_consensus=True,
    )
    # Mocking the 4-point grid mutation done by the dashboard
    import main
    original_grid = main.LAMBDA_GRID
    main.LAMBDA_GRID = [4.64, 10.0, 21.54, 46.42]
    
    print("\nRunning Backtest for Optimized Preset...")
    res_optimized = walk_forward_backtest(main_df, cfg_optimized)
    
    # Restore original grid just in case
    main.LAMBDA_GRID = original_grid
    
    evaluate_execution(res_baseline, exec_data, "1. Paper Baseline")
    evaluate_execution(res_optimized, exec_data, "2. Optimized (4-pt Grid + SubWindow)")