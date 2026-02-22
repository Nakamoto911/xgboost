import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io

import main as backend

st.set_page_config(page_title="XGBoost Strategy Portal", layout="wide")

st.title("XGBoost Strategy Backtest Portal")

st.sidebar.header("Configuration")
target_ticker = st.sidebar.text_input("Target Ticker", backend.TARGET_TICKER)
bond_ticker = st.sidebar.text_input("Bond Ticker", backend.BOND_TICKER)
risk_free_ticker = st.sidebar.text_input("Risk-Free Ticker", backend.RISK_FREE_TICKER)
vix_ticker = st.sidebar.text_input("VIX Ticker", backend.VIX_TICKER)

start_date_data = st.sidebar.text_input("Data Start Date", backend.START_DATE_DATA)
oos_start_date = st.sidebar.text_input("OOS Start Date", backend.OOS_START_DATE)
end_date = st.sidebar.text_input("End Date", backend.END_DATE)

transaction_cost = st.sidebar.number_input("Transaction Cost", value=backend.TRANSACTION_COST, format="%.4f")
lambda_grid_str = st.sidebar.text_input("Lambda Grid (comma separated)", ",".join(map(str, backend.LAMBDA_GRID)))

run_simple_jm = st.sidebar.checkbox("Run Simple JM Baseline", value=False)

if st.sidebar.button("Run Backtest"):
    # Clear cache if parameters changed to ensure new data is fetched
    # We will just delete the cache file if it exists, to be safe
    if os.path.exists('data_cache.pkl'):
        os.remove('data_cache.pkl')
    
    # Update backend globals
    backend.TARGET_TICKER = target_ticker
    backend.BOND_TICKER = bond_ticker
    backend.RISK_FREE_TICKER = risk_free_ticker
    backend.VIX_TICKER = vix_ticker
    backend.START_DATE_DATA = start_date_data
    backend.OOS_START_DATE = oos_start_date
    backend.END_DATE = end_date
    backend.TRANSACTION_COST = transaction_cost
    
    try:
        backend.LAMBDA_GRID = [float(x.strip()) for x in lambda_grid_str.split(',')]
    except ValueError:
        st.error("Invalid Lambda Grid format. Using default.")

    st.write("Fetching and preparing data...")
    try:
        df = backend.fetch_and_prepare_data()
        st.success("Data fetched and features prepared successfully!")
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()
        
    st.write(f"Starting Out-of-Sample Backtest ({backend.OOS_START_DATE} to {backend.END_DATE})")
    
    current_date = pd.to_datetime(backend.OOS_START_DATE)
    final_end_date = pd.to_datetime(backend.END_DATE)
    
    jm_xgb_results = []
    simple_jm_results = []
    lambda_history = []
    lambda_dates = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Calculate total steps for progress bar
    total_months = (final_end_date.year - current_date.year) * 12 + final_end_date.month - current_date.month
    total_steps = max(1, total_months // 6)
    step = 0
    
    while current_date < final_end_date:
        chunk_end = min(current_date + pd.DateOffset(months=6), final_end_date)
        val_start = current_date - pd.DateOffset(years=5)
        
        status_text.text(f"Evaluating period: {current_date.date()} to {chunk_end.date()}")
        
        # 1. Hyperparameter Tuning
        best_sharpe = -np.inf
        best_lambda = backend.LAMBDA_GRID[0]
        
        for lmbda in backend.LAMBDA_GRID:
            val_res = backend.simulate_strategy(df, val_start, current_date, lmbda, include_xgboost=True)
            if not val_res.empty:
                _, _, sharpe, _ = backend.calculate_metrics(val_res['Strat_Return'], val_res['RF_Rate'])
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_lambda = lmbda
                    
        lambda_history.append(best_lambda)
        lambda_dates.append(current_date)
        
        # 2. Out-of-Sample Execution
        oos_chunk_jm_xgb = backend.run_period_forecast(df, current_date, best_lambda, include_xgboost=True)
        if oos_chunk_jm_xgb is not None:
            jm_xgb_results.append(oos_chunk_jm_xgb)
            
        if run_simple_jm:
            oos_chunk_simple_jm = backend.run_period_forecast(df, current_date, 10.0, include_xgboost=False)
            if oos_chunk_simple_jm is not None:
                simple_jm_results.append(oos_chunk_simple_jm)
            
        current_date = chunk_end
        step += 1
        progress_bar.progress(min(1.0, step / total_steps))
        
    status_text.text("Backtest complete! Generating results...")
    progress_bar.empty()
    
    if not jm_xgb_results:
        st.error("No results generated. Check your date ranges and data availability.")
        st.stop()
        
    jm_xgb_df = pd.concat(jm_xgb_results)
    
    strat_dfs = [jm_xgb_df]
    if run_simple_jm:
        simple_jm_df = pd.concat(simple_jm_results)
        strat_dfs.append(simple_jm_df)
        
    for strat_df in strat_dfs:
        strat_returns = np.where(strat_df['Forecast_State'] == 0, 
                                 strat_df['Target_Return'], 
                                 strat_df['RF_Rate'])
        trades = strat_df['Forecast_State'].diff().abs().fillna(0)
        strat_df['Strat_Return'] = strat_returns - (trades * backend.TRANSACTION_COST)
        strat_df['Trades'] = trades

    bh_returns = jm_xgb_df['Target_Return']
    rf_returns = jm_xgb_df['RF_Rate']
    
    strategies = {
        'Buy & Hold': bh_returns,
        'JM-XGB Strategy': jm_xgb_df['Strat_Return']
    }
    if run_simple_jm:
        strategies['Simple JM Baseline'] = simple_jm_df['Strat_Return']
        
    columns = ['Strategy', 'Ann. Ret', 'Ann. Vol', 'Sharpe', 'Max DD', 'Total Trades']
    cell_text = []
    
    metrics_data = [] # For Streamlit dataframe
    
    for name, returns in strategies.items():
        ret, vol, sharpe, mdd = backend.calculate_metrics(returns, rf_returns)
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
        ])
        
        metrics_data.append({
            'Strategy': name,
            'Ann. Ret': f"{ret*100:.2f}%",
            'Ann. Vol': f"{vol*100:.2f}%",
            'Sharpe': f"{sharpe:.2f}",
            'Max DD': f"{mdd*100:.2f}%",
            'Total Trades': str(trades)
        })

    st.subheader("Performance Metrics")
    st.dataframe(pd.DataFrame(metrics_data))
    
    st.subheader("Results Chart")
    
    # Generate Matplotlib Figure
    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 4, 1.5])
    
    ax_table = fig.add_subplot(gs[0])
    ax_table.axis('off')
    table = ax_table.table(cellText=cell_text, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)
    ax_table.set_title("Out-of-Sample Performance Metrics", fontweight='bold', fontsize=14)
    
    ax_plot = fig.add_subplot(gs[1])
    for name, returns in strategies.items():
        wealth = (1 + returns).cumprod()
        end_wealth = wealth.iloc[-1]
        ax_plot.plot(wealth, label=f"{name} (Final: {end_wealth:.2f}x)")
        
    ax_plot.set_title(f"Wealth Curves: {backend.OOS_START_DATE} to {backend.END_DATE}", fontsize=12)
    ax_plot.set_ylabel('Cumulative Wealth (Multiplier)')
    
    bear_regimes = jm_xgb_df['Forecast_State'] == 1
    ax_plot.fill_between(jm_xgb_df.index, 0, 1, where=bear_regimes, color='red', alpha=0.15, 
                         transform=ax_plot.get_xaxis_transform(), label='Bear Regime (JM-XGB)')
                         
    ax_plot.legend()
    ax_plot.grid(True, which="both", ls="--", alpha=0.5)

    lambda_dates_full = lambda_dates + [pd.to_datetime(backend.END_DATE)]
    lambda_history_full = lambda_history + [lambda_history[-1]]
    
    ax_lambda = fig.add_subplot(gs[2], sharex=ax_plot)
    ax_lambda.step(lambda_dates_full, lambda_history_full, where='post', color='purple', label='Selected Lambda Penalty', linewidth=2)
    ax_lambda.set_ylabel('Lambda Penalty')
    ax_lambda.set_xlabel('Date')
    ax_lambda.grid(True, which="both", ls="--", alpha=0.5)
    ax_lambda.legend(loc='upper left')

    plt.tight_layout()
    
    # Generate Plotly Figure for interactive UI
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig_plotly = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.1,
                               subplot_titles=("Wealth Curves", "Lambda Penalty"),
                               row_heights=[0.7, 0.3])
                               
    for name, returns in strategies.items():
        wealth = (1 + returns).cumprod()
        fig_plotly.add_trace(go.Scatter(x=wealth.index, y=wealth, mode='lines', 
                                        name=f"{name} (Final: {wealth.iloc[-1]:.2f}x)"), 
                             row=1, col=1)
                             
    bear_series = jm_xgb_df['Forecast_State'] == 1
    starts = bear_series.index[(bear_series) & (~bear_series.shift(1).fillna(False))]
    ends = bear_series.index[(bear_series) & (~bear_series.shift(-1).fillna(False))]
    
    for s, e in zip(starts, ends):
        fig_plotly.add_vrect(x0=s, x1=e, fillcolor="red", opacity=0.15, line_width=0, row=1, col=1)
        
    fig_plotly.add_trace(go.Scatter(x=lambda_dates_full, y=lambda_history_full, 
                                    line_shape='hv', name='Selected Lambda Penalty',
                                    line=dict(color='purple', width=2)),
                         row=2, col=1)
                         
    fig_plotly.update_layout(height=600, hovermode="x unified",
                             margin=dict(l=20, r=20, t=40, b=20))
    fig_plotly.update_yaxes(title_text="Cumulative Wealth", row=1, col=1)
    fig_plotly.update_yaxes(title_text="Lambda Penalty", row=2, col=1)
    
    st.plotly_chart(fig_plotly, use_container_width=True)
    
    # Save to buffer for download
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches='tight')
    buf.seek(0)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f'performance_report_{timestamp}.pdf'
    
    st.download_button(
        label="Download PDF Report",
        data=buf,
        file_name=pdf_filename,
        mime="application/pdf"
    )
    
    plt.close(fig)
