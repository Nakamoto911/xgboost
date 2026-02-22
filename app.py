import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import pickle

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
                _, _, sharpe, _, _ = backend.calculate_metrics(val_res['Strat_Return'], val_res['RF_Rate'])
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

    cache_data = {
        'jm_xgb_results': jm_xgb_results,
        'simple_jm_results': simple_jm_results,
        'lambda_history': lambda_history,
        'lambda_dates': lambda_dates,
        'run_simple_jm': run_simple_jm,
        'oos_start_date': backend.OOS_START_DATE,
        'end_date': backend.END_DATE,
    }
    with open('backtest_cache.pkl', 'wb') as f:
        pickle.dump(cache_data, f)

cache_loaded = False
if os.path.exists('backtest_cache.pkl'):
    try:
        with open('backtest_cache.pkl', 'rb') as f:
            cache_data = pickle.load(f)
            
        jm_xgb_results = cache_data.get('jm_xgb_results', [])
        simple_jm_results = cache_data.get('simple_jm_results', [])
        lambda_history = cache_data.get('lambda_history', [])
        lambda_dates = cache_data.get('lambda_dates', [])
        run_simple_jm_cached = cache_data.get('run_simple_jm', False)
        cached_oos_start = cache_data.get('oos_start_date', backend.OOS_START_DATE)
        cached_end_date = cache_data.get('end_date', backend.END_DATE)
        cache_loaded = True
    except Exception as e:
        st.error(f"Could not load cache: {e}")

if not cache_loaded:
    st.info("No cached results found. Please configure parameters and click 'Run Backtest'.")
    st.stop()
    
if not jm_xgb_results:
    st.error("No results generated. Check your date ranges and data availability.")
    st.stop()
    
jm_xgb_df = pd.concat(jm_xgb_results)
if run_simple_jm_cached:
    simple_jm_df = pd.concat(simple_jm_results)

st.subheader("Analysis Period")

min_date = jm_xgb_df.index.min().date()
max_date = jm_xgb_df.index.max().date()

if 'period_slider' not in st.session_state:
    st.session_state.period_slider = (min_date, max_date)

def set_period(months):
    if months is None:
        st.session_state.period_slider = (min_date, max_date)
    else:
        new_start = (pd.to_datetime(max_date) - pd.DateOffset(months=months)).date()
        new_start = max(min_date, new_start)
        st.session_state.period_slider = (new_start, max_date)

# Quick selection buttons
col1, col2, col3, col4, col5, _ = st.columns([1, 1, 1, 1, 1, 5])
with col1:
    st.button("1 Year", on_click=set_period, args=(12,))
with col2:
    st.button("3 Years", on_click=set_period, args=(36,))
with col3:
    st.button("5 Years", on_click=set_period, args=(60,))
with col4:
    st.button("10 Years", on_click=set_period, args=(120,))
with col5:
    st.button("Max", on_click=set_period, args=(None,))

selected_dates = st.slider(
    "Select Date Range",
    min_value=min_date,
    max_value=max_date,
    key='period_slider',
    format="YYYY-MM-DD"
)

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    filter_start, filter_end = selected_dates
else:
    st.warning("Please select both a start and end date from the slider.")
    st.stop()

if filter_start >= filter_end:
    st.warning("Start date must be before end date.")
    st.stop()

jm_xgb_df = jm_xgb_df.loc[str(filter_start):str(filter_end)]
if jm_xgb_df.empty:
    st.warning("No data in the selected period. Please expand your date range.")
    st.stop()
    
if run_simple_jm_cached:
    simple_jm_df = simple_jm_df.loc[str(filter_start):str(filter_end)]

strat_dfs = [jm_xgb_df]
if run_simple_jm_cached:
    strat_dfs.append(simple_jm_df)
    
for strat_df in strat_dfs:
    strat_returns = np.where(strat_df['Forecast_State'] == 0, 
                             strat_df['Target_Return'], 
                             strat_df['RF_Rate'])
    trades = strat_df['Forecast_State'].diff().abs().fillna(0)
    strat_df['Strat_Return'] = strat_returns - (trades * transaction_cost)
    strat_df['Trades'] = trades

bh_returns = jm_xgb_df['Target_Return']
rf_returns = jm_xgb_df['RF_Rate']

strategies = {
    'Buy & Hold': bh_returns,
    'JM-XGB Strategy': jm_xgb_df['Strat_Return']
}
if run_simple_jm_cached:
    strategies['Simple JM Baseline'] = simple_jm_df['Strat_Return']

strategy_colors = {
    'Buy & Hold': 'dodgerblue',
    'JM-XGB Strategy': 'darkorange',
    'Simple JM Baseline': 'forestgreen'
}

columns = ['Strategy', 'Ann. Ret', 'Ann. Vol', 'Sharpe', 'Sortino', 'Max DD', 'Total Trades']
cell_text = []

metrics_data = [] # For Streamlit dataframe

for name, returns in strategies.items():
    ret, vol, sharpe, sortino, mdd = backend.calculate_metrics(returns, rf_returns)
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
        f"{sortino:.2f}",
        f"{mdd*100:.2f}%", 
        str(trades)
    ])
    
    metrics_data.append({
        'Strategy': name,
        'Ann. Ret': f"{ret*100:.2f}%",
        'Ann. Vol': f"{vol*100:.2f}%",
        'Sharpe': f"{sharpe:.2f}",
        'Sortino': f"{sortino:.2f}",
        'Max DD': f"{mdd*100:.2f}%",
        'Total Trades': str(trades)
    })

st.subheader("Performance Metrics")
st.dataframe(pd.DataFrame(metrics_data))

st.subheader("Results Chart")

# Generate Matplotlib Figure
fig = plt.figure(figsize=(12, 25))
gs = fig.add_gridspec(6, 1, height_ratios=[1, 4, 1.5, 1.5, 1.5, 1.5])

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
    ax_plot.plot(wealth, label=f"{name} (Final: {end_wealth:.2f}x)", color=strategy_colors.get(name))
    
ax_plot.set_title(f"Wealth Curves: {filter_start} to {filter_end}", fontsize=12)
ax_plot.set_ylabel('Cumulative Wealth (Multiplier)')

bear_regimes = jm_xgb_df['Forecast_State'] == 1
ax_plot.fill_between(jm_xgb_df.index, 0, 1, where=bear_regimes, color='red', alpha=0.15, 
                     transform=ax_plot.get_xaxis_transform(), label='Bear Regime (JM-XGB)')
                     
ax_plot.legend()
ax_plot.grid(True, which="both", ls="--", alpha=0.5)

# Filter lambda dates for plotting
if lambda_dates:
    lambda_ts = pd.Series(lambda_history, index=pd.to_datetime(lambda_dates))
    lambda_ts_daily = lambda_ts.reindex(jm_xgb_df.index.union(lambda_ts.index)).ffill().bfill().loc[jm_xgb_df.index]
    lambda_dates_full = lambda_ts_daily.index
    lambda_history_full = lambda_ts_daily.values
else:
    lambda_dates_full = [pd.to_datetime(filter_start), pd.to_datetime(filter_end)]
    lambda_history_full = [0, 0]

ax_lambda = fig.add_subplot(gs[2], sharex=ax_plot)
ax_lambda.step(lambda_dates_full, lambda_history_full, where='post', color='purple', label='Selected Lambda Penalty', linewidth=2)
ax_lambda.set_ylabel('Lambda Penalty')
ax_lambda.grid(True, which="both", ls="--", alpha=0.5)
ax_lambda.legend(loc='upper left')

ax_trades = fig.add_subplot(gs[3], sharex=ax_plot)
# Aggregate trades by the 6-month evaluation periods
trades_by_period = jm_xgb_df.groupby(pd.Grouper(freq='6M'))['Trades'].sum()
ax_trades.step(trades_by_period.index, trades_by_period, where='post', color=strategy_colors.get('JM-XGB Strategy', 'orange'), label='Transactions (JM-XGB)', linewidth=2)
if run_simple_jm_cached:
    simple_jm_trades_by_period = simple_jm_df.groupby(pd.Grouper(freq='6M'))['Trades'].sum()
    ax_trades.step(simple_jm_trades_by_period.index, simple_jm_trades_by_period, where='post', color=strategy_colors.get('Simple JM Baseline', 'green'), label='Transactions (Simple JM)', linewidth=2)
ax_trades.set_ylabel('Transactions')
ax_trades.grid(True, which="both", ls="--", alpha=0.5)
ax_trades.legend(loc='upper left')

ax_returns = fig.add_subplot(gs[4], sharex=ax_plot)
returns_by_period_dict = {}
for name, returns in strategies.items():
    ret_per = returns.groupby(pd.Grouper(freq='6M')).apply(lambda x: (1 + x).prod() - 1)
    returns_by_period_dict[name] = ret_per
    ax_returns.step(ret_per.index, ret_per, where='post', label=f"{name} Returns", color=strategy_colors.get(name), linewidth=2)
ax_returns.set_ylabel('Periodic Returns')
ax_returns.set_xlabel('Date')
import matplotlib.ticker as mtick
ax_returns.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax_returns.grid(True, which="both", ls="--", alpha=0.5)
ax_returns.legend(loc='upper left')

ax_vol = fig.add_subplot(gs[5], sharex=ax_plot)
vol_by_period_dict = {}
for name, returns in strategies.items():
    vol_per = returns.groupby(pd.Grouper(freq='6M')).apply(lambda x: x.std() * np.sqrt(252))
    vol_by_period_dict[name] = vol_per
    ax_vol.step(vol_per.index, vol_per, where='post', label=f"{name} Volatility", color=strategy_colors.get(name), linewidth=2)
ax_vol.set_ylabel('Periodic Volatility')
ax_vol.set_xlabel('Date')
ax_vol.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax_vol.grid(True, which="both", ls="--", alpha=0.5)
ax_vol.legend(loc='upper left')

plt.tight_layout()

# Generate Plotly Figure for interactive UI
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig_plotly = make_subplots(rows=5, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.06,
                           subplot_titles=("Wealth Curves", "Lambda Penalty", "Transactions", "Periodic Returns", "Periodic Volatility"),
                           row_heights=[0.3, 0.175, 0.175, 0.175, 0.175])
                           
for name, returns in strategies.items():
    wealth = (1 + returns).cumprod()
    fig_plotly.add_trace(go.Scatter(x=wealth.index, y=wealth, mode='lines', 
                                    name=f"{name} (Final: {wealth.iloc[-1]:.2f}x)",
                                    line=dict(color=strategy_colors.get(name))), 
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

fig_plotly.add_trace(go.Scatter(x=trades_by_period.index, y=trades_by_period, 
                                line_shape='hv', name='Transactions (JM-XGB)',
                                line=dict(color=strategy_colors.get('JM-XGB Strategy', 'orange'), width=2)),
                     row=3, col=1)
if run_simple_jm_cached:
    fig_plotly.add_trace(go.Scatter(x=simple_jm_trades_by_period.index, y=simple_jm_trades_by_period, 
                                    line_shape='hv', name='Transactions (Simple JM)',
                                    line=dict(color=strategy_colors.get('Simple JM Baseline', 'green'), width=2)),
                         row=3, col=1)

for name, ret_per in returns_by_period_dict.items():
    fig_plotly.add_trace(go.Scatter(x=ret_per.index, y=ret_per, 
                                    line_shape='hv', name=f"{name} Periodic Returns",
                                    line=dict(color=strategy_colors.get(name), width=2)),
                         row=4, col=1)

for name, vol_per in vol_by_period_dict.items():
    fig_plotly.add_trace(go.Scatter(x=vol_per.index, y=vol_per, 
                                    line_shape='hv', name=f"{name} Periodic Volatility",
                                    line=dict(color=strategy_colors.get(name), width=2)),
                         row=5, col=1)
                     
fig_plotly.update_layout(height=1200, hovermode="x unified",
                         margin=dict(l=20, r=20, t=40, b=20))
fig_plotly.update_yaxes(title_text="Cumulative Wealth", row=1, col=1)
fig_plotly.update_yaxes(title_text="Lambda Penalty", row=2, col=1)
fig_plotly.update_yaxes(title_text="Transactions", row=3, col=1)
fig_plotly.update_yaxes(title_text="Periodic Returns", tickformat=".2%", row=4, col=1)
fig_plotly.update_yaxes(title_text="Periodic Volatility", tickformat=".2%", row=5, col=1)

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
