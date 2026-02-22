import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import pickle
import json
import tempfile
from datetime import datetime
import main as backend

st.set_page_config(page_title="XGBoost Strategy Portal", layout="wide")

st.title("XGBoost Strategy Backtest Portal")
export_container = st.container()

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
if hasattr(st, 'segmented_control'):
    def update_from_segment():
        sel = st.session_state.period_selector
        if sel == "1 Year": set_period(12)
        elif sel == "3 Years": set_period(36)
        elif sel == "5 Years": set_period(60)
        elif sel == "10 Years": set_period(120)
        elif sel == "Max": set_period(None)
        
    st.segmented_control(
        "Period Selection", 
        ["1 Year", "3 Years", "5 Years", "10 Years", "Max"],
        key="period_selector", 
        on_change=update_from_segment,
        selection_mode="single",
        label_visibility="collapsed"
    )
elif hasattr(st, 'pills'):
    def update_from_pills():
        sel = st.session_state.period_selector
        if sel == "1 Year": set_period(12)
        elif sel == "3 Years": set_period(36)
        elif sel == "5 Years": set_period(60)
        elif sel == "10 Years": set_period(120)
        elif sel == "Max": set_period(None)
        
    st.pills(
        "Period Selection", 
        ["1 Year", "3 Years", "5 Years", "10 Years", "Max"],
        key="period_selector", 
        on_change=update_from_pills,
        selection_mode="single",
        label_visibility="collapsed"
    )
else:
    col1, col2, col3, col4, col5, _ = st.columns([1, 1, 1, 1, 1, 5])
    with col1:
        st.button("1 Year", on_click=set_period, args=(12,), width='stretch')
    with col2:
        st.button("3 Years", on_click=set_period, args=(36,), width='stretch')
    with col3:
        st.button("5 Years", on_click=set_period, args=(60,), width='stretch')
    with col4:
        st.button("10 Years", on_click=set_period, args=(120,), width='stretch')
    with col5:
        st.button("Max", on_click=set_period, args=(None,), width='stretch')

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
        
    # Calculate Drawdown for the current period
    cum_wealth = (1 + returns).cumprod()
    peak = cum_wealth.cummax()
    drawdowns = (cum_wealth - peak) / peak
    
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

# Export UI logic has been moved to the end of the file to capture generated charts.

# We no longer use Matplotlib for the main display, we will just rely entirely on Plotly 
# for a much better interactive experience with these dense charts.

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# We need the benchmark wealth for the Probability chart
bh_wealth = (1 + bh_returns).cumprod()
jm_xgb_wealth = (1 + jm_xgb_df['Strat_Return']).cumprod()

# Overlay Actual Bear Regimes as background color (from thresholded state)
bear_series = jm_xgb_df['Forecast_State'] == 1
starts = bear_series.index[(bear_series) & (~bear_series.shift(1).fillna(False))]
ends = bear_series.index[(bear_series) & (~bear_series.shift(-1).fillna(False))]

def create_base_fig(title, y_title, height=500):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        yaxis_title=y_title,
        height=height,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="top", y=-0.15, x=0.5, xanchor="center", bgcolor="rgba(0,0,0,0)")
    )
    return fig

def apply_bear_shading(fig):
    for s, e in zip(starts, ends):
        fig.add_vrect(x0=s, x1=e, fillcolor="red", opacity=0.1, line_width=0)

# --------------------------------------------------------------------------
# Chart 1: Cumulative Wealth
# --------------------------------------------------------------------------
st.subheader("1. Cumulative Wealth")
fig_wealth = create_base_fig("Cumulative Wealth Multiplier", "Wealth Multiplier")

for name, returns in strategies.items():
    if name == 'Simple JM Baseline' and not run_simple_jm_cached:
        continue
    wealth = (1 + returns).cumprod()
    final_val = wealth.iloc[-1]
    
    fig_wealth.add_trace(go.Scatter(
        x=wealth.index, y=wealth, mode='lines', 
        name=f"{name} Wealth",
        line=dict(color=strategy_colors.get(name), width=2 if 'JM' in name else 1.5)
    ))
    
    # Add ending wealth annotation
    fig_wealth.add_annotation(
        x=wealth.index[-1], y=final_val,
        text=f" {final_val:.2f}x",
        showarrow=False,
        xanchor="left",
        xshift=5,
        font=dict(color=strategy_colors.get(name), size=12),
        bgcolor="rgba(0,0,0,0.3)"
    )

apply_bear_shading(fig_wealth)
st.plotly_chart(fig_wealth, width='stretch')

# --------------------------------------------------------------------------
# Chart 2: Drawdown Profile
# --------------------------------------------------------------------------
st.subheader("2. Drawdown Profile")
fig_dd = create_base_fig("Portfolio Drawdown Percentage", "Drawdown")
fig_dd.update_yaxes(tickformat=".1%")

# Sort strategies to ensure JM-XGB is plotted last (rendered on top)
for name, returns in sorted(strategies.items(), key=lambda x: 1 if x[0] == 'JM-XGB Strategy' else 0):
    if name == 'Simple JM Baseline' and not run_simple_jm_cached:
        continue
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak
    
    is_xgb = name == 'JM-XGB Strategy'
    fig_dd.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown, mode='lines', 
        name=f"{name} DD",
        line=dict(color=strategy_colors.get(name), width=2.5 if is_xgb else 1.5),
        fill='tozeroy' if not is_xgb else None,
        fillcolor=strategy_colors.get(name) if not is_xgb else None,
        opacity=1.0 if is_xgb else 0.3
    ))

apply_bear_shading(fig_dd)
st.plotly_chart(fig_dd, width='stretch')

# --------------------------------------------------------------------------
# Chart 3: Regime Probability vs Benchmark Price
# --------------------------------------------------------------------------
st.subheader("3. Regime Probability vs. Market Price")
fig_prob = make_subplots(specs=[[{"secondary_y": True}]])
fig_prob.update_layout(
    height=500,
    template="plotly_dark",
    hovermode="x unified",
    margin=dict(l=20, r=20, t=60, b=40),
    legend=dict(orientation="h", yanchor="top", y=-0.15, x=0.5, xanchor="center", bgcolor="rgba(0,0,0,0)")
)

# Left Y-Axis: Benchmark Wealth Curve
fig_prob.add_trace(
    go.Scatter(x=bh_wealth.index, y=bh_wealth, mode='lines', 
               name="S&P 500 (Wealth)",
               line=dict(color='gray', width=1.5, dash='dot')), 
    secondary_y=False
)

# Right Y-Axis: Bear Market Probability
if 'State_Prob' in jm_xgb_df.columns:
    fig_prob.add_trace(
        go.Scatter(x=jm_xgb_df.index, y=jm_xgb_df['State_Prob'], mode='lines', 
                   name="Bear Regime Probability",
                   line=dict(color='red', width=1),
                   fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.15)'), 
        secondary_y=True
    )
    fig_prob.add_hline(y=0.5, line_dash="dash", line_color="red", secondary_y=True)

fig_prob.update_yaxes(title_text="Benchmark Price", secondary_y=False)
fig_prob.update_yaxes(title_text="Bear Probability", tickformat=".0%", range=[0, 1], secondary_y=True)

apply_bear_shading(fig_prob)
st.plotly_chart(fig_prob, width='stretch')

# --------------------------------------------------------------------------
# Chart 4: Rolling 1-Year Sharpe Ratio
# --------------------------------------------------------------------------
st.subheader("4. Rolling 1-Year Sharpe Ratio")
fig_sharpe = create_base_fig("Rolling 12-Month Sharpe Ratio", "Sharpe Ratio")
ROLLING_WINDOW = 252 # 1 Trading Year

for name, returns in strategies.items():
    if name == 'Simple JM Baseline' and not run_simple_jm_cached:
        continue
    excess_ret = returns - rf_returns
    rolling_ann_ret = excess_ret.rolling(window=ROLLING_WINDOW).sum()
    rolling_ann_vol = returns.rolling(window=ROLLING_WINDOW).std() * np.sqrt(252)
    rolling_sharpe = rolling_ann_ret / rolling_ann_vol.replace(0, np.nan)
    
    fig_sharpe.add_trace(go.Scatter(
        x=rolling_sharpe.index, y=rolling_sharpe, mode='lines', 
        name=f"{name} (1Y Sharpe)",
        line=dict(color=strategy_colors.get(name), width=1.5)
    ))

fig_sharpe.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
apply_bear_shading(fig_sharpe)
st.plotly_chart(fig_sharpe, width='stretch')

# --------------------------------------------------------------------------
# Chart 5: Time-Series Feature Contribution (Local SHAP)
# --------------------------------------------------------------------------
st.subheader("5. Time-Series Feature Contribution (Local SHAP)")
shap_cols = [c for c in jm_xgb_df.columns if c.startswith('SHAP_') and c != 'SHAP_Base_Value']

if shap_cols and not jm_xgb_df[shap_cols].empty:
    fig_shap_ts = create_base_fig("Local Feature Impact (XGBoost)", "SHAP Value (Log-Odds Impact)", height=600)
    shap_df = jm_xgb_df[shap_cols].resample('2W').mean().dropna(how='all')
    
    import plotly.colors
    colors_cycle = plotly.colors.qualitative.Plotly
    
    for i, col in enumerate(shap_cols):
        feature_name = col.replace('SHAP_', '')
        if shap_df[col].abs().sum() > 0.001:
            fig_shap_ts.add_trace(go.Bar(
                x=shap_df.index, y=shap_df[col], 
                name=feature_name,
                marker_color=colors_cycle[i % len(colors_cycle)],
                showlegend=False,
                legendgroup="SHAP"
            ))
            
    fig_shap_ts.update_layout(barmode='relative')
    st.plotly_chart(fig_shap_ts, width='stretch')

# --------------------------------------------------------------------------
# Chart 6: Transactions & Lambda
# --------------------------------------------------------------------------
st.subheader("6. Transactions & Lambda Penalty")
# Filter lambda dates for plotting
if lambda_dates:
    lambda_ts = pd.Series(lambda_history, index=pd.to_datetime(lambda_dates))
    lambda_ts_daily = lambda_ts.reindex(jm_xgb_df.index.union(lambda_ts.index)).ffill().bfill().loc[jm_xgb_df.index]
    lambda_dates_full = lambda_ts_daily.index
    lambda_history_full = lambda_ts_daily.values
else:
    lambda_dates_full = [pd.to_datetime(filter_start), pd.to_datetime(filter_end)]
    lambda_history_full = [0, 0]

# Aggregate trades by the 6-month evaluation periods
trades_by_period = jm_xgb_df.groupby(pd.Grouper(freq='6M'))['Trades'].sum()

fig_trades = make_subplots(specs=[[{"secondary_y": True}]])
fig_trades.update_layout(
    height=400,
    template="plotly_dark",
    hovermode="x unified",
    margin=dict(l=20, r=20, t=60, b=40),
    legend=dict(orientation="h", yanchor="top", y=-0.15, x=0.5, xanchor="center", bgcolor="rgba(0,0,0,0)")
)

fig_trades.add_trace(
    go.Scatter(x=trades_by_period.index, y=trades_by_period, 
               mode='lines+markers', name='Transactions',
               line=dict(color=strategy_colors.get('JM-XGB Strategy', 'orange'), width=2)),
    secondary_y=False
)

if run_simple_jm_cached:
    simple_jm_trades_by_period = simple_jm_df.groupby(pd.Grouper(freq='6M'))['Trades'].sum()
    fig_trades.add_trace(go.Scatter(x=simple_jm_trades_by_period.index, y=simple_jm_trades_by_period, 
                                    mode='lines+markers', name='Transactions (Simple JM)',
                                    line=dict(color=strategy_colors.get('Simple JM Baseline', 'green'), width=2)),
                         secondary_y=False)

fig_trades.add_trace(
    go.Scatter(x=lambda_dates_full, y=lambda_history_full, 
               line_shape='hv', name='Selected Lambda Penalty',
               line=dict(color='purple', width=2, dash='dot')),
    secondary_y=True
)

fig_trades.update_yaxes(title_text="Transactions", secondary_y=False)
fig_trades.update_yaxes(title_text="Lambda Penalty", secondary_y=True)

apply_bear_shading(fig_trades)
st.plotly_chart(fig_trades, width='stretch')

# --------------------------------------------------------------------------
# Chart 7: Periodic Returns
# --------------------------------------------------------------------------
st.subheader("7. Periodic Returns")
fig_ret = create_base_fig("Periodic Returns (6-Month Intervals)", "Returns", height=400)
fig_ret.update_yaxes(tickformat=".2%")

returns_by_period_dict = {}
for name, returns in strategies.items():
    if name == 'Simple JM Baseline' and not run_simple_jm_cached:
        continue
    ret_per = returns.groupby(pd.Grouper(freq='6M')).apply(lambda x: (1 + x).prod() - 1)
    returns_by_period_dict[name] = ret_per
    fig_ret.add_trace(go.Scatter(
        x=ret_per.index, y=ret_per, 
        mode='lines+markers', name=f"{name} Periodic Returns",
        line=dict(color=strategy_colors.get(name), width=2)
    ))

apply_bear_shading(fig_ret)
st.plotly_chart(fig_ret, width='stretch')

# --------------------------------------------------------------------------
# Chart 8: Periodic Volatility
# --------------------------------------------------------------------------
st.subheader("8. Periodic Volatility")
fig_vol = create_base_fig("Periodic Volatility (6-Month Intervals)", "Volatility", height=400)
fig_vol.update_yaxes(tickformat=".2%")

vol_by_period_dict = {}
for name, returns in strategies.items():
    if name == 'Simple JM Baseline' and not run_simple_jm_cached:
        continue
    vol_per = returns.groupby(pd.Grouper(freq='6M')).apply(lambda x: x.std() * np.sqrt(252))
    vol_by_period_dict[name] = vol_per
    fig_vol.add_trace(go.Scatter(
        x=vol_per.index, y=vol_per, 
        mode='lines+markers', name=f"{name} Periodic Volatility",
        line=dict(color=strategy_colors.get(name), width=2)
    ))

apply_bear_shading(fig_vol)
st.plotly_chart(fig_vol, width='stretch')


# --------------------------------------------------------------------------
# Global SHAP Summary Plot 
# --------------------------------------------------------------------------
if shap_cols and not jm_xgb_df[shap_cols].empty:
    st.subheader("Global Feature Importance (SHAP Summary)")
    st.write("Average absolute impact on model output across the entire selected time period.")
    
    # Calculate mean absolute SHAP value for each feature
    mean_abs_shap = jm_xgb_df[shap_cols].abs().mean().sort_values(ascending=True)
    mean_abs_shap.index = [x.replace('SHAP_', '') for x in mean_abs_shap.index]
    
    fig_shap_summary = go.Figure(go.Bar(
        x=mean_abs_shap.values,
        y=mean_abs_shap.index,
        orientation='h',
        marker_color='royalblue'
    ))
    
    fig_shap_summary.update_layout(
        height=400 + (len(mean_abs_shap) * 20),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Mean Absolute SHAP Value (Impact on prediction)"
    )
    st.plotly_chart(fig_shap_summary, width='stretch')

# --------------------------------------------------------------------------
# Export Actions (JSON & PDF) inside the Top Container
# --------------------------------------------------------------------------
with export_container:
    col_exp1, col_exp2 = st.columns(2)
    
    dialog_func = getattr(st, "dialog", getattr(st, "experimental_dialog", None))
    
    def generate_export_dict():
        df_no_shap = jm_xgb_df.drop(columns=[c for c in jm_xgb_df.columns if c.startswith('SHAP_')])
        features = [c for c in df_no_shap.columns if c not in ["Target_Return", "RF_Rate", "Forecast_State", "Strat_Return", "Trades", "State_Prob"]]
        
        last_record = df_no_shap.tail(1).copy()
        last_record.index = last_record.index.astype(str)
        last_record_dict = last_record.reset_index().to_dict('records')
        
        return {
            "params": {
                "tgt": backend.TARGET_TICKER, 
                "bnd": backend.BOND_TICKER, 
                "rf": backend.RISK_FREE_TICKER,
                "vix": backend.VIX_TICKER,
                "cost": backend.TRANSACTION_COST, 
                "oos": f"{filter_start} to {filter_end}",
                "lambda_sel": lambda_history[-1] if lambda_history else "N/A"
            },
            "data": {
                "rows": jm_xgb_df.shape[0], 
                "feats": features,
                "has_shap": any(c.startswith('SHAP_') for c in jm_xgb_df.columns)
            },
            "latest_row": last_record_dict[0] if len(last_record_dict) > 0 else {},
            "outputs": [
                {
                    "strat": m["Strategy"][:15], "ret": m["Ann. Ret"], "vol": m["Ann. Vol"], 
                    "sr": m["Sharpe"], "dd": m["Max DD"], "trd": m["Total Trades"]
                } for m in metrics_data
            ]
        }

    def generate_pdf_report():
        try:
            from fpdf import FPDF
            import kaleido
        except ImportError:
            return None

        export_dict = generate_export_dict()
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Title & Subtitle
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="XGBoost Strategy Backtest Report", ln=True, align='C')
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(200, 10, txt=f"Generated on: {current_time_str}", ln=True, align='C')
        pdf.ln(5)
        
        # Configuration / Parameters
        param_labels = {
            "tgt": "Target Ticker",
            "bnd": "Bond Ticker",
            "rf": "Risk-Free Ticker",
            "vix": "VIX Ticker",
            "cost": "Transaction Cost",
            "oos": "Out-of-Sample Period",
            "lambda_sel": "Final Lambda Penalty"
        }
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Configuration Parameters:", ln=True)
        pdf.set_font("Arial", size=10)
        for k, v in export_dict["params"].items():
            label = param_labels.get(k, k)
            pdf.cell(200, 6, txt=f"- {label}: {v}", ln=True)
        pdf.ln(5)

        # Data Properties
        data_labels = {
            "rows": "Total Observations (Rows)",
            "has_shap": "SHAP Values Available"
        }
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Data Properties:", ln=True)
        pdf.set_font("Arial", size=10)
        for k, v in export_dict["data"].items():
            if k != "feats":
                label = data_labels.get(k, k)
                pdf.cell(200, 6, txt=f"- {label}: {v}", ln=True)
        pdf.ln(5)
        
        # Performance Metrics
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Performance Metrics:", ln=True)
        pdf.set_font("Arial", size=10)
        
        headers = ['Strategy', 'Ann. Ret', 'Ann. Vol', 'Sharpe', 'Max DD', 'Total Trades']
        col_widths = [45, 25, 25, 25, 25, 25]
        
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 8, header, border=1, align='C')
        pdf.ln()
        
        for m in metrics_data:
            pdf.cell(col_widths[0], 8, m['Strategy'][:20], border=1)
            pdf.cell(col_widths[1], 8, m['Ann. Ret'], border=1, align='C')
            pdf.cell(col_widths[2], 8, m['Ann. Vol'], border=1, align='C')
            pdf.cell(col_widths[3], 8, m['Sharpe'], border=1, align='C')
            pdf.cell(col_widths[4], 8, m['Max DD'], border=1, align='C')
            pdf.cell(col_widths[5], 8, m['Total Trades'], border=1, align='C')
            pdf.ln()
        pdf.ln(5)
        
        # Charts
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Charts:", ln=True)
        
        figs = {
            "Cumulative Wealth": fig_wealth,
            "Drawdown Profile": fig_dd,
            "Regime Probability vs Market": fig_prob,
            "Rolling 1-Year Sharpe": fig_sharpe,
            "Transactions & Lambda": fig_trades,
            "Periodic Returns": fig_ret,
            "Periodic Volatility": fig_vol,
        }
        
        if 'fig_shap_ts' in locals() or 'fig_shap_ts' in globals():
            f = globals().get('fig_shap_ts') or locals().get('fig_shap_ts')
            if f: figs["SHAP Time-Series"] = f
            
        if 'fig_shap_summary' in locals() or 'fig_shap_summary' in globals():
            f = globals().get('fig_shap_summary') or locals().get('fig_shap_summary')
            if f: figs["SHAP Summary"] = f
            
        with tempfile.TemporaryDirectory() as tmpdir:
            for name, fig in figs.items():
                if fig is not None:
                    img_path = os.path.join(tmpdir, f"{name.replace(' ', '_')}.png")
                    fig_copy = go.Figure(fig)
                    fig_copy.update_layout(template="plotly_white", paper_bgcolor='rgba(255,255,255,1)', plot_bgcolor='rgba(255,255,255,1)', font=dict(color='black'))
                    try:
                        fig_copy.write_image(img_path, width=800, height=450, scale=2)
                        
                        pdf.add_page()
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(200, 10, txt=name, ln=True)
                        pdf.image(img_path, x=10, w=190)
                    except Exception as e:
                        pdf.set_font("Arial", size=10)
                        pdf.cell(200, 10, txt=f"Could not render {name}: {str(e)}", ln=True)
                        continue
                    
        return bytes(pdf.output())

    with col_exp1:
        if st.button("📄 Export to PDF"):
            with st.spinner("Generating PDF Report... This may take a few moments."):
                pdf_bytes = generate_pdf_report()
            
            if pdf_bytes is None:
                st.error("Missing dependencies. Please run `pip install fpdf2 kaleido` in your terminal and restart Streamlit.")
            else:
                current_time_file = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="✅ Click to Save PDF",
                    data=pdf_bytes,
                    file_name=f"xgb_{current_time_file}.pdf",
                    mime="application/pdf"
                )

    with col_exp2:
        if dialog_func:
            @dialog_func("LLM Audit Export")
            def show_export_dialog():
                st.markdown("Copy the JSON below and share it with your LLM for auditing context.")
                export_dict = generate_export_dict()
                st.code(json.dumps(export_dict, separators=(',', ':')), language="json")
                
            if st.button("🤖 Export to LLM"):
                show_export_dialog()
        else:
            with st.expander("🤖 Export JSON"):
                st.markdown("Copy the JSON below and share it with your LLM for auditing context.")
                export_dict = generate_export_dict()
                st.code(json.dumps(export_dict, separators=(',', ':')), language="json")
