import re

with open('main.py', 'r') as f:
    text = f.read()

text = text.replace('def main():', 'def main(run_simple_jm=False):')

text = text.replace(
'''    jm_xgb_results = []
    simple_jm_results = []''',
'''    jm_xgb_results = []
    simple_jm_results = []
    lambda_history = []
    lambda_dates = []''')

text = text.replace(
'''        print(f"Optimal Lambda selected for JM-XGB: {best_lambda} (Val Sharpe: {best_sharpe:.2f})")''',
'''        print(f"Optimal Lambda selected for JM-XGB: {best_lambda} (Val Sharpe: {best_sharpe:.2f})")
        lambda_history.append(best_lambda)
        lambda_dates.append(current_date)''')

text = text.replace(
'''        # Simple JM (Using a default lambda for the simple baseline to save compute, 
        # normally you'd tune this separately)
        oos_chunk_simple_jm = run_period_forecast(df, current_date, 10.0, include_xgboost=False)
        if oos_chunk_simple_jm is not None:
            simple_jm_results.append(oos_chunk_simple_jm)''',
'''        # Simple JM (Using a default lambda for the simple baseline to save compute, 
        # normally you'd tune this separately)
        if run_simple_jm:
            oos_chunk_simple_jm = run_period_forecast(df, current_date, 10.0, include_xgboost=False)
            if oos_chunk_simple_jm is not None:
                simple_jm_results.append(oos_chunk_simple_jm)''')

text = text.replace(
'''    # Combine Results
    jm_xgb_df = pd.concat(jm_xgb_results)
    simple_jm_df = pd.concat(simple_jm_results)
    
    # Calculate Strategy Returns applying transaction costs
    for strat_df in [jm_xgb_df, simple_jm_df]:
        strat_returns = np.where(strat_df['Forecast_State'] == 0, 
                                 strat_df['Target_Return'], 
                                 strat_df['RF_Rate'])
        trades = strat_df['Forecast_State'].diff().abs().fillna(0)
        strat_df['Strat_Return'] = strat_returns - (trades * TRANSACTION_COST)
        strat_df['Trades'] = trades''',
'''    # Combine Results
    jm_xgb_df = pd.concat(jm_xgb_results)
    strat_dfs = [jm_xgb_df]
    if run_simple_jm:
        simple_jm_df = pd.concat(simple_jm_results)
        strat_dfs.append(simple_jm_df)
    
    # Calculate Strategy Returns applying transaction costs
    for strat_df in strat_dfs:
        strat_returns = np.where(strat_df['Forecast_State'] == 0, 
                                 strat_df['Target_Return'], 
                                 strat_df['RF_Rate'])
        trades = strat_df['Forecast_State'].diff().abs().fillna(0)
        strat_df['Strat_Return'] = strat_returns - (trades * TRANSACTION_COST)
        strat_df['Trades'] = trades''')


text = text.replace(
'''    strategies = {
        'Buy & Hold': bh_returns,
        'Simple JM Baseline': simple_jm_df['Strat_Return'],
        'JM-XGB Strategy': jm_xgb_df['Strat_Return']
    }
    
    # Prepare Table Data
    columns = ['Strategy', 'Ann. Ret', 'Ann. Vol', 'Sharpe', 'Max DD', 'Total Trades']
    cell_text = []
    
    for name, returns in strategies.items():
        ret, vol, sharpe, mdd = calculate_metrics(returns, rf_returns)
        trades = "N/A" if name == 'Buy & Hold' else int(jm_xgb_df['Trades'].sum() if "XGB" in name else simple_jm_df['Trades'].sum())
        cell_text.append([
            name,
            f"{ret*100:.2f}%", 
            f"{vol*100:.2f}%", 
            f"{sharpe:.2f}", 
            f"{mdd*100:.2f}%", 
            str(trades)
        ])''',
'''    strategies = {
        'Buy & Hold': bh_returns,
        'JM-XGB Strategy': jm_xgb_df['Strat_Return']
    }
    if run_simple_jm:
        strategies['Simple JM Baseline'] = simple_jm_df['Strat_Return']
    
    # Prepare Table Data
    columns = ['Strategy', 'Ann. Ret', 'Ann. Vol', 'Sharpe', 'Max DD', 'Total Trades']
    cell_text = []
    
    for name, returns in strategies.items():
        ret, vol, sharpe, mdd = calculate_metrics(returns, rf_returns)
        if name == 'Buy & Hold':
            trades = "N/A"
        elif "XGB" in name:
            trades = int(jm_xgb_df['Trades'].sum())
        else:
            trades = int(simple_jm_df['Trades'].sum())
            
        cell_text.append([
            name,
            f"{ret*100:.2f}%", 
            f"{vol*100:.2f}%", 
            f"{sharpe:.2f}", 
            f"{mdd*100:.2f}%", 
            str(trades)
        ])''')


text = text.replace(
'''    # Create Figure with Table and Plot
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4])''',
'''    # Create Figure with Table and Plot
    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 4, 1.5])''')


text = text.replace(
'''    # Axis for Plotting Wealth Curves
    ax_plot = fig.add_subplot(gs[1])
    for name, returns in strategies.items():
        wealth = (1 + returns).cumprod()
        ax_plot.plot(wealth, label=name)
        
    ax_plot.set_yscale('log')
    ax_plot.set_title(f'Log Wealth Curves: {OOS_START_DATE} to {END_DATE}', fontsize=12)
    ax_plot.set_ylabel('Cumulative Wealth (Log Scale)')
    ax_plot.set_xlabel('Date')
    ax_plot.legend()
    ax_plot.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()''',
'''    # Axis for Plotting Wealth Curves
    ax_plot = fig.add_subplot(gs[1])
    for name, returns in strategies.items():
        wealth = (1 + returns).cumprod()
        # Add ending wealth to label
        end_wealth = wealth.iloc[-1]
        ax_plot.plot(wealth, label=f"{name} (Final: {end_wealth:.2f}x)")
        
    ax_plot.set_title(f'Wealth Curves: {OOS_START_DATE} to {END_DATE}', fontsize=12)
    ax_plot.set_ylabel('Cumulative Wealth (Multiplier)')
    ax_plot.legend()
    ax_plot.grid(True, which="both", ls="--", alpha=0.5)

    # Axis for Plotting Lambda
    lambda_dates_full = lambda_dates + [pd.to_datetime(END_DATE)]
    lambda_history_full = lambda_history + [lambda_history[-1]]
    ax_lambda = fig.add_subplot(gs[2], sharex=ax_plot)
    ax_lambda.step(lambda_dates_full, lambda_history_full, where='post', color='purple', label='Selected Lambda Penalty', linewidth=2)
    ax_lambda.set_ylabel('Lambda Penalty')
    ax_lambda.set_xlabel('Date')
    ax_lambda.grid(True, which="both", ls="--", alpha=0.5)
    ax_lambda.legend(loc='upper left')

    plt.tight_layout()''')

with open('main.py', 'w') as f:
    f.write(text)

