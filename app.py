import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import pickle
import json
import tempfile
import time
from datetime import datetime
import main as backend

st.set_page_config(page_title="XGBoost Strategy Portal", layout="wide")

st.title("XGBoost Strategy Backtest Portal")
export_container = st.container()

if 'start_date_input' not in st.session_state:
    st.session_state.start_date_input = backend.START_DATE_DATA
if 'oos_start_input' not in st.session_state:
    st.session_state.oos_start_input = backend.OOS_START_DATE
if 'target_ticker_input' not in st.session_state:
    st.session_state.target_ticker_input = backend.TARGET_TICKER
if 'val_window_input' not in st.session_state:
    st.session_state.val_window_input = backend.VALIDATION_WINDOW_YRS

def on_ticker_change():
    new_ticker = st.session_state.target_ticker_input
    import yfinance as yf
    try:
        t = yf.Ticker(new_ticker)
        hist = t.history(period="max")
        if not hist.empty:
            earliest = hist.index.min()
            st.session_state.start_date_input = earliest.strftime('%Y-%m-%d')
            val_window = st.session_state.val_window_input
            oos_date = earliest + pd.DateOffset(years=int(11 + val_window))
            st.session_state.oos_start_input = oos_date.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error fetching data for ticker {new_ticker}: {e}")

st.sidebar.header("Configuration")
target_ticker = st.sidebar.text_input("Target Ticker", key='target_ticker_input', on_change=on_ticker_change)
bond_ticker = st.sidebar.text_input("Bond Ticker", backend.BOND_TICKER)
risk_free_ticker = st.sidebar.text_input("Risk-Free Ticker", backend.RISK_FREE_TICKER)
vix_ticker = st.sidebar.text_input("VIX Ticker", backend.VIX_TICKER)

start_date_data = st.sidebar.text_input("Data Start Date", key='start_date_input')
oos_start_date = st.sidebar.text_input("OOS Start Date", key='oos_start_input')
end_date = st.sidebar.text_input("End Date", backend.END_DATE)

transaction_cost = st.sidebar.number_input("Transaction Cost", value=float(backend.TRANSACTION_COST), format="%.4f")
validation_window = st.sidebar.number_input("Validation Window (Years)", min_value=1, max_value=20, key='val_window_input', on_change=on_ticker_change)
lambda_grid_str = st.sidebar.text_input("Lambda Grid (comma separated)", ",".join(map(str, backend.LAMBDA_GRID)))

run_simple_jm = st.sidebar.checkbox("Run Simple JM Baseline", value=False)

run_button = st.sidebar.button("Run Backtest")
duration_placeholder = st.sidebar.empty()

@st.cache_data
def get_cached_data(target, bond, rf, vix, start, end):
    backend.TARGET_TICKER = target
    backend.BOND_TICKER = bond
    backend.RISK_FREE_TICKER = rf
    backend.VIX_TICKER = vix
    backend.START_DATE_DATA = start
    backend.END_DATE = end
    if os.path.exists('data_cache.pkl'):
        try:
            os.remove('data_cache.pkl')
        except:
            pass
    return backend.fetch_and_prepare_data()

if run_button:
    backtest_start_time = time.time()
    
    # Update backend globals
    backend.TARGET_TICKER = target_ticker
    backend.BOND_TICKER = bond_ticker
    backend.RISK_FREE_TICKER = risk_free_ticker
    backend.VIX_TICKER = vix_ticker
    backend.START_DATE_DATA = start_date_data
    backend.OOS_START_DATE = oos_start_date
    backend.END_DATE = end_date
    backend.TRANSACTION_COST = transaction_cost
    backend.VALIDATION_WINDOW_YRS = validation_window
    
    try:
        backend.LAMBDA_GRID = [float(x.strip()) for x in lambda_grid_str.split(',')]
    except ValueError:
        st.error("Invalid Lambda Grid format. Using default.")

    st.write("Fetching and preparing data...")
    try:
        df = get_cached_data(target_ticker, bond_ticker, risk_free_ticker, vix_ticker, start_date_data, end_date)
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
        val_start = current_date - pd.DateOffset(years=backend.VALIDATION_WINDOW_YRS)
        
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
    
    backtest_duration = time.time() - backtest_start_time

    cache_data = {
        'jm_xgb_results': jm_xgb_results,
        'simple_jm_results': simple_jm_results,
        'lambda_history': lambda_history,
        'lambda_dates': lambda_dates,
        'run_simple_jm': run_simple_jm,
        'oos_start_date': backend.OOS_START_DATE,
        'end_date': backend.END_DATE,
        'backtest_duration': backtest_duration,
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
        backtest_duration = cache_data.get('backtest_duration', None)
        cache_loaded = True
        
        if backtest_duration is not None:
            if backtest_duration >= 3600:
                h = int(backtest_duration // 3600)
                m = int((backtest_duration % 3600) // 60)
                s = int(backtest_duration % 60)
                duration_str = f"{h}h {m}m {s}s"
            elif backtest_duration >= 60:
                m = int(backtest_duration // 60)
                s = int(backtest_duration % 60)
                duration_str = f"{m}m {s}s"
            else:
                duration_str = f"{backtest_duration:.2f} seconds"
            duration_placeholder.info(f"Last backtest duration: {duration_str}")
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
        fig.add_vrect(x0=s, x1=e, fillcolor="red", opacity=0.1, line_width=0, layer="below")

# --------------------------------------------------------------------------
# Main Dashboard Layout
# --------------------------------------------------------------------------
tab_perf, tab_features, tab_feat_charts, tab_jm_audit, tab_xgb_eval = st.tabs(["Performance & Tracking", "Feature Impact Analysis", "Feature Charts", "JM Audit", "XGBoost Eval"])

with tab_perf:
    st.subheader(
        "1. Cumulative Wealth",
        help="**Calculated:** Compounding growth of daily returns `(1 + returns).cumprod()`.\n\n**Data:** Strategy returns (net of transaction costs) vs. benchmarks.\n\n**Interpret:** Shows the total value of investment over time. A curve that stays above the benchmark indicates outperformance."
    )
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
    st.subheader(
        "2. Drawdown Profile",
        help="**Calculated:** Percentage drop from the highest historical peak `(wealth - peak) / peak`.\n\n**Data:** Cumulative wealth trajectories of the strategy vs. benchmarks.\n\n**Interpret:** Visualizes downside risk. The area closer to 0% is better. Shallower drawdowns during red regions (bear markets) show the strategy successfully preserved capital."
    )
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
    # Chart 3: Rolling 1-Year Sortino Ratio
    # --------------------------------------------------------------------------
    st.subheader(
        "3. Rolling 1-Year Sortino Ratio",
        help="**Calculated:** Rolling 252-day annualized excess return over the risk-free rate, divided by downside deviation (standard deviation of only negative returns).\n\n**Data:** Daily strategy returns, risk-free rate, and benchmark returns.\n\n**Interpret:** Evaluates risk-adjusted performance while only penalizing downside volatility. Essential for regime-switching models designed to cut losses but let winners run. Higher values are better."
    )
    fig_sortino = create_base_fig("Rolling 12-Month Sortino Ratio", "Sortino Ratio")
    ROLLING_WINDOW = 252 # 1 Trading Year
    
    for name, returns in strategies.items():
        if name == 'Simple JM Baseline' and not run_simple_jm_cached:
            continue
        excess_ret = returns - rf_returns
        rolling_ann_ret = excess_ret.rolling(window=ROLLING_WINDOW).sum()
        
        # Calculate trailing downside deviation
        # Create a boolean mask for negative excess returns, calculate rolling std on that
        downside_returns = excess_ret.where(excess_ret < 0, 0)
        rolling_downside_vol = downside_returns.rolling(window=ROLLING_WINDOW).std() * np.sqrt(252)
        
        rolling_sortino = rolling_ann_ret / rolling_downside_vol.replace(0, np.nan)
        
        fig_sortino.add_trace(go.Scatter(
            x=rolling_sortino.index, y=rolling_sortino, mode='lines', 
            name=f"{name} (1Y Sortino)",
            line=dict(color=strategy_colors.get(name), width=1.5)
        ))
    
    fig_sortino.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
    apply_bear_shading(fig_sortino)
    st.plotly_chart(fig_sortino, width='stretch')

    # --------------------------------------------------------------------------
    # Chart 4: Monthly Returns Heatmap
    # --------------------------------------------------------------------------
    st.subheader(
        "4. Monthly Returns Heatmap (JM-XGB Strategy)",
        help="**Calculated:** Compounded daily returns for each calendar month.\n\n**Data:** Daily strategy returns.\n\n**Interpret:** Shows seasonality and consistency. Green months are profitable. Look for a balance of consistency (lots of green) and magnitude (deep green vs shallow red)."
    )
    
    # Calculate monthly returns for the main strategy
    strat_returns_daily = strategies['JM-XGB Strategy']
    # Resample to monthly and calculate geometric return
    monthly_returns = strat_returns_daily.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    
    # Create a DataFrame for the heatmap
    heatmap_df = pd.DataFrame({'Return': monthly_returns})
    heatmap_df['Year'] = heatmap_df.index.year
    heatmap_df['Month'] = heatmap_df.index.strftime('%b')
    
    # Pivot the data
    heatmap_pivot = heatmap_df.pivot(index='Year', columns='Month', values='Return')
    
    # Reorder columns to standard calendar order
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # Only include months that exist in the data
    valid_months = [m for m in month_order if m in heatmap_pivot.columns]
    heatmap_pivot = heatmap_pivot[valid_months]
    
    # Calculate Annual Returns for row totals
    heatmap_pivot['YTD'] = strat_returns_daily.resample('YE').apply(lambda x: (1 + x).prod() - 1).values
    
    # Format as percentages
    heatmap_text = heatmap_pivot.map(lambda x: f"{x:.1%}" if pd.notnull(x) else "")
    
    # Create the heatmap figure
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        text=heatmap_text.values,
        texttemplate="%{text}",
        textfont={"size": 10},
        colorscale=["red", "black", "green"],
        zmid=0,  # Center the color scale at 0%
        showscale=False,
        xgap=2,
        ygap=2
    ))
    
    fig_heatmap.update_layout(
        height=min(400, 50 + 30 * len(heatmap_pivot)), # Dynamic height based on years
        template="plotly_dark",
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis=dict(autorange="reversed", type='category') # Reverse year order so most recent is top
    )
    st.plotly_chart(fig_heatmap, width='stretch')

    col_dist, col_scatter = st.columns(2)
    
    with col_dist:
        # --------------------------------------------------------------------------
        # Chart 5: Return Distribution Histogram
        # --------------------------------------------------------------------------
        st.subheader(
            "5. Return Distribution",
            help="**Calculated:** Histogram of daily return frequencies.\n\n**Data:** Daily returns of the strategy vs. Buy & Hold benchmark.\n\n**Interpret:** A good strategy chops off the 'left tail' (fewer large negative days) while preserving the 'right tail' (large positive days)."
        )
        
        fig_dist = go.Figure()
        
        all_rets = pd.concat([strategies['Buy & Hold'], strategies['JM-XGB Strategy']]).dropna()
        if not all_rets.empty:
            p_low = all_rets.quantile(0.005)
            p_high = all_rets.quantile(0.995)
            max_abs = max(abs(p_low), abs(p_high), 0.02)
            x_range = [-max_abs, max_abs]
            b_size = max_abs / 40.0
            x_bins = dict(start=-max_abs, end=max_abs, size=b_size)
        else:
            x_range = [-0.05, 0.05]
            x_bins = None
        
        # Plot Benchmark first (in background)
        if x_bins is not None:
            import numpy as np
            bins_array = np.arange(x_bins['start'], x_bins['end'] + x_bins['size'], x_bins['size'])
        else:
            bins_array = 100

        def get_hist_lines(data, bins):
            counts, bin_edges = np.histogram(data.dropna(), bins=bins)
            probs = counts / counts.sum()
            centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            return centers, probs

        centers_bh, probs_bh = get_hist_lines(strategies['Buy & Hold'], bins_array)
        fig_dist.add_trace(go.Scatter(
            x=centers_bh, y=probs_bh,
            mode='lines',
            name='Buy & Hold',
            line=dict(color='gray', width=2, shape='spline'),
            fill='tozeroy',
            fillcolor='rgba(128, 128, 128, 0.4)'
        ))

        centers_st, probs_st = get_hist_lines(strategies['JM-XGB Strategy'], bins_array)
        fig_dist.add_trace(go.Scatter(
            x=centers_st, y=probs_st,
            mode='lines',
            name='JM-XGB Strategy',
            line=dict(color='darkorange', width=2, shape='spline'),
            fill='tozeroy',
            fillcolor='rgba(255, 140, 0, 0.5)'
        ))

        fig_dist.update_layout(
            barmode='overlay',
            template="plotly_dark",
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Daily Return",
            yaxis_title="Probability",
            xaxis=dict(tickformat=".1%", range=x_range),
            legend=dict(orientation="h", yanchor="top", y=-0.15, x=0.5, xanchor="center", bgcolor="rgba(0,0,0,0)")
        )
        # Add a vertical line at 0
        fig_dist.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
        st.plotly_chart(fig_dist, width='stretch')
        
    with col_scatter:
        # --------------------------------------------------------------------------
        # Chart 6: Risk / Return Scatter
        # --------------------------------------------------------------------------
        st.subheader(
            "6. Risk / Return Profile",
            help="**Calculated:** Annualized Return vs. Annualized Volatility over the entire backtest period.\n\n**Data:** Strategy metrics table.\n\n**Interpret:** The 'Holy Grail' is the top-left quadrant (High Return, Low Risk). Strategies further up and to the left are mathematically superior."
        )
        
        fig_scatter = go.Figure()
        
        for m in metrics_data:
            name = m['Strategy']
            # Parse percentage strings back to floats
            ret = float(m['Ann. Ret'].strip('%')) / 100
            vol = float(m['Ann. Vol'].strip('%')) / 100
            
            fig_scatter.add_trace(go.Scatter(
                x=[vol], y=[ret],
                mode='markers+text',
                name=name,
                marker=dict(
                    size=15, 
                    color=strategy_colors.get(name, 'gray'),
                    line=dict(width=2, color='white')
                ),
                text=[name.replace(' Strategy', '').replace(' Baseline', '')],
                textposition="top center"
            ))
            
        fig_scatter.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Annualized Volatility (Risk)",
            yaxis_title="Annualized Return",
            xaxis=dict(tickformat=".1%", rangemode="tozero"),
            yaxis=dict(tickformat=".1%"),
            showlegend=False
        )
        # Add risk-free rate assumption line (horizontal)
        rf_mean = rf_returns.mean() * 252
        fig_scatter.add_hline(y=rf_mean, line_dash="dot", line_color="gray", annotation_text=f"Risk-Free ({rf_mean:.1%})", annotation_position="bottom right")
        
        st.plotly_chart(fig_scatter, width='stretch')

# --------------------------------------------------------------------------
with tab_features:
    st.header("Feature Impact Analysis")
    st.markdown("Understand how the XGBoost model uses different features to identify bear market regimes.")
    
    # --------------------------------------------------------------------------
    # Chart 5: Time-Series Feature Contribution (Local SHAP)
    # --------------------------------------------------------------------------
    st.subheader(
        "Time-Series Feature Contribution (Local SHAP)",
        help="**Calculated:** Average Daily SHAP values aggregated to a monthly frequency.\n\n**Data:** XGBoost SHAP (SHapley Additive exPlanations) values for the top 7 features.\n\n**Interpret:** Shows exactly how much each feature contributed to the model's 'Bear' prediction at any point in time. Positive values push the model toward calling a Bear regime."
    )
    shap_cols = [c for c in jm_xgb_df.columns if c.startswith('SHAP_') and c != 'SHAP_Base_Value']
    
    if shap_cols and not jm_xgb_df[shap_cols].empty:
        fig_shap_ts = create_base_fig("Local Feature Impact (XGBoost)", "SHAP Value (Log-Odds Impact)", height=600)
        # Resample to monthly to improve rendering performance and visual clarity
        shap_df = jm_xgb_df[shap_cols].groupby(pd.Grouper(freq='ME')).mean().dropna(how='all')
        if shap_df.empty: # Fallback to generic month freq if ME fails in older pandas
            shap_df = jm_xgb_df[shap_cols].groupby(pd.Grouper(freq='M')).mean().dropna(how='all')
        
        import plotly.colors
        colors_cycle = plotly.colors.qualitative.Plotly
        
        # Identify top 7 features by mean absolute impact to avoid cluttering the chart
        mean_abs_impact = shap_df.abs().mean().sort_values(ascending=False)
        top_7_cols = mean_abs_impact.head(7).index.tolist()
        other_cols = [c for c in shap_cols if c not in top_7_cols]
        
        # Add traces for Top 7 features
        for i, col in enumerate(top_7_cols):
            feature_name = col.replace('SHAP_', '')
            fig_shap_ts.add_trace(go.Bar(
                x=shap_df.index, y=shap_df[col], 
                name=feature_name,
                marker_color=colors_cycle[i % len(colors_cycle)],
                showlegend=True,
                legendgroup="SHAP"
            ))
            
        # Group remaining features into 'Other Features'
        if other_cols:
            shap_df['Other_Features'] = shap_df[other_cols].sum(axis=1)
            fig_shap_ts.add_trace(go.Bar(
                x=shap_df.index, y=shap_df['Other_Features'], 
                name='Other Features',
                marker_color='gray',
                showlegend=True,
                legendgroup="SHAP"
            ))
                
        fig_shap_ts.update_layout(barmode='relative')
        apply_bear_shading(fig_shap_ts)
        st.plotly_chart(fig_shap_ts, width='stretch')



# --------------------------------------------------------------------------
# Global SHAP Summary Plot & Detailed Feature Analysis
# --------------------------------------------------------------------------
with tab_features:
    if shap_cols and not jm_xgb_df[shap_cols].empty:
        st.subheader(
            "Global Feature Importance (SHAP Summary)",
            help="**Calculated:** Mean Absolute SHAP value for each feature over the entire period.\n\n**Data:** XGBoost SHAP values.\n\n**Interpret:** Identifies the single most important features for the model on average. Features at the top drive the majority of the model's regime classification decisions."
        )
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
        # New Feature Analysis Tools
        # --------------------------------------------------------------------------
        st.markdown("---")
        
        col_f1, col_f2 = st.columns(2)
        
        # Top 5 Features Averages by Regime
        with col_f1:
            st.subheader(
                "Feature Averages by Regime (Raw Values)",
                help="**Calculated:** Simple average of raw feature values grouped by the predicted regime.\n\n**Data:** Raw feature data and predicted XGBoost states.\n\n**Interpret:** Helps ground the SHAP values by showing what the features actually look like in Bull vs Bear markets (e.g. is VIX higher in Bear regimes?)."
            )
            st.markdown("How do the most important features behave in Bull vs Bear regimes?")
            
            top_features = mean_abs_shap.tail(5).index.tolist()[::-1] # Reverse to get top 5 descending
            
            # Reconstruct the raw feature DataFrame from jm_xgb_df (excluding Target_Return, etc)
            exclude_avg_cols = ['Target_Return', 'RF_Rate', 'Forecast_State', 'Strat_Return', 'Trades', 'State_Prob'] + shap_cols + ['SHAP_Base_Value']
            raw_feats_df = jm_xgb_df.drop(columns=[c for c in jm_xgb_df.columns if c in exclude_avg_cols])
            
            if not raw_feats_df.empty:
                # Ensure top_features actually exist in raw_feats_df (accounting for 'Feature_' prefix)
                feature_to_raw_col = {f: f"Feature_{f}" for f in top_features if f"Feature_{f}" in raw_feats_df.columns}
                
                if feature_to_raw_col:
                    bull_avg = raw_feats_df.loc[jm_xgb_df['Forecast_State'] == 0, list(feature_to_raw_col.values())].mean()
                    bear_avg = raw_feats_df.loc[jm_xgb_df['Forecast_State'] == 1, list(feature_to_raw_col.values())].mean()
                    
                    regime_comp_df = pd.DataFrame({
                        'Feature': list(feature_to_raw_col.keys()),
                        'Bull Regime Avg': bull_avg.values,
                        'Bear Regime Avg': bear_avg.values
                    })
                    
                    # Formatting for readability
                    st.dataframe(
                        regime_comp_df.style.format({
                            'Bull Regime Avg': '{:.4f}',
                            'Bear Regime Avg': '{:.4f}',
                        }), 
                        hide_index=True, 
                        width='stretch'
                    )
                else:
                    st.info("Raw feature data not found for the top SHAP features.")
                    
        # Feature Dependence Plot
        with col_f2:
            st.subheader(
                "Feature Dependence (SHAP Impact vs Raw Value)",
                help="**Calculated:** Scatter plot of a feature's raw value (X) vs its SHAP value (Y).\n\n**Data:** Raw feature data, SHAP values, and predicted regimes (colors).\n\n**Interpret:** Reveals non-linear relationships and thresholds. For example, you can see the exact numerical level where VIX spikes from being a 'Bull' signal to a 'Bear' signal."
            )
            st.markdown("Observe non-linear thresholds where a feature triggers a regime shift.")
            
            # Select feature to plot
            raw_feature_cols = [c for c in raw_feats_df.columns if c.startswith('Feature_')]
            all_raw_feats = [c.replace('Feature_', '') for c in raw_feature_cols if f"SHAP_{c.replace('Feature_', '')}" in shap_cols]
            
            if all_raw_feats:
                selected_dep_feat = st.selectbox("Select Feature for Dependence Plot", all_raw_feats, index=all_raw_feats.index(top_features[0]) if top_features and top_features[0] in all_raw_feats else 0)
                
                fig_dep = go.Figure()
                
                feat_raw_vals = raw_feats_df[f"Feature_{selected_dep_feat}"]
                feat_shap_vals = jm_xgb_df[f"SHAP_{selected_dep_feat}"]
                
                fig_dep.add_trace(go.Scatter(
                    x=feat_raw_vals,
                    y=feat_shap_vals,
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=jm_xgb_df['Forecast_State'], # Color by regime
                        colorscale=[[0, 'deepskyblue'], [1, 'crimson']],
                        showscale=False,
                        opacity=0.6,
                        line=dict(width=0.5, color='white')
                    ),
                    text=[f"Date: {d.date()}<br>Regime: {'Bear' if r==1 else 'Bull'}" for d, r in zip(jm_xgb_df.index, jm_xgb_df['Forecast_State'])],
                    hoverinfo="text+x+y"
                ))
                
                fig_dep.update_layout(
                    height=350,
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis_title=f"{selected_dep_feat} (Raw Value)",
                    yaxis_title="SHAP Value (Impact)",
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                # Add horizontal line at 0
                fig_dep.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                
                st.plotly_chart(fig_dep, width='stretch')
                
        st.markdown("---")
        
        # Point-in-Time Explanation (Waterfall alternative using Bar)
        st.subheader(
            "Point-in-Time Explanation",
            help="**Calculated:** Static bar chart of SHAP values for a single specific day.\n\n**Data:** XGBoost SHAP values extracted by date.\n\n**Interpret:** Extremely useful for auditing specific market crashes or fake-outs. Pick a date to see exactly which indicator triggered the model's action."
        )
        st.markdown("Examine exactly which features drove the model prediction on a specific day.")
        
        # Date selection
        available_dates = jm_xgb_df.index.tolist()
        
        col_date1, col_date2, _ = st.columns([1, 1, 2])
        with col_date1:
            # Let user select a date that was marked as a bear regime to make it interesting
            bear_dates = jm_xgb_df[jm_xgb_df['Forecast_State'] == 1].index.tolist()
            default_date = bear_dates[0] if bear_dates else available_dates[-1]
            selected_date = st.selectbox("Select Date to Explain", available_dates, index=available_dates.index(default_date), format_func=lambda x: x.strftime('%Y-%m-%d'))
        
        with col_date2:
            st.metric(
                "Predicted Regime", 
                "Bear (High Risk)" if jm_xgb_df.loc[selected_date, 'Forecast_State'] == 1 else "Bull (Low Risk)",
                f"Prob: {jm_xgb_df.loc[selected_date, 'State_Prob']:.1%}" if 'State_Prob' in jm_xgb_df.columns else ""
            )
            
        # Get SHAP values for that specific date
        specific_shap = jm_xgb_df.loc[selected_date, shap_cols]
        base_val = jm_xgb_df.loc[selected_date, 'SHAP_Base_Value'] if 'SHAP_Base_Value' in jm_xgb_df.columns else 0
        
        # Create a Bar chart (simulating a waterfall profile since Plotly Waterfall can be fragile without base values)
        # Sort by absolute impact
        specific_shap_sorted = specific_shap[specific_shap.abs().sort_values(ascending=False).index]
        
        # Take top 10 impacts to avoid clutter
        top_n_impacts = specific_shap_sorted.head(10)
        
        # If there are residual impacts, sum them
        other_impacts = specific_shap_sorted.iloc[10:].sum()
        if abs(other_impacts) > 0.001:
            top_n_impacts['SHAP_Other_Features'] = other_impacts
            
        # Sort values ascending for horizontal bar chart
        top_n_impacts = top_n_impacts.sort_values(ascending=True)
        
        bar_colors = ['crimson' if val > 0 else 'deepskyblue' for val in top_n_impacts.values]
        
        # Clean labels
        clean_labels = [idx.replace('SHAP_', '') for idx in top_n_impacts.index]
        
        fig_point = go.Figure(go.Bar(
            x=top_n_impacts.values,
            y=clean_labels,
            orientation='h',
            marker_color=bar_colors,
            text=[f"{val:+.3f}" for val in top_n_impacts.values],
            textposition='auto',
        ))
        
        fig_point.update_layout(
            title=f"Top Feature Impacts on {selected_date.strftime('%Y-%m-%d')}",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis_title="Log-Odds Impact (Positive = Pushing towards Bear Regime)",
            template="plotly_dark",
            yaxis=dict(autorange="reversed") # Put biggest at top (because we sorted ascending earlier, actually wait, let's keep it standard)
        )
        # Fix axis range: If we sorted ascending, the largest is at the bottom of the dataframe, so if we don't reverse, it plots at the top... Plotly plots bottom-up for Series.
        fig_point.update_yaxes(autorange="reversed", title="") # To have biggest at top

        fig_point.add_vline(x=0, line_width=1, line_color="white")
        
        st.plotly_chart(fig_point, width='stretch')

# --------------------------------------------------------------------------
# Feature Charts Tab
# --------------------------------------------------------------------------
with tab_feat_charts:
    st.header("Feature Time Series")
    
    # Re-display Cumulative Wealth
    st.subheader("Cumulative Wealth")
    st.plotly_chart(fig_wealth, width='stretch', key="wealth_chart_features_tab")

    # --------------------------------------------------------------------------
    # Transactions & Lambda
    # --------------------------------------------------------------------------
    st.subheader(
        "Transactions & Lambda Penalty",
        help="**Calculated:** Sum of regime-switching trades aggregated over 6-month periods, overlaid with the selected Lambda (Jump Penalty) parameter.\n\n**Data:** Number of trades triggered by regime changes and the history of the Jump Penalty hyperparameter.\n\n**Interpret:** Shows how actively the strategy trades. Higher Lambda values penalize jumping between states, theoretically reducing the transaction count."
    )
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
                   line=dict(color='#FF00FF', width=4, dash='solid')),
        secondary_y=True
    )
    
    fig_trades.update_yaxes(title_text="Transactions", secondary_y=False)
    fig_trades.update_yaxes(title_text="Lambda Penalty", secondary_y=True)

    apply_bear_shading(fig_trades)
    st.plotly_chart(fig_trades, width='stretch')
    
    st.subheader("Feature Values Over Time")
    
    # Identify feature columns
    # We exclude core columns, SHAP columns, and other metadata
    exclude_cols = ['Target_Return', 'RF_Rate', 'Forecast_State', 'Strat_Return', 'Trades', 'State_Prob', 'SHAP_Base_Value']
    feature_cols = [c for c in jm_xgb_df.columns if c not in exclude_cols and not c.startswith('SHAP_')]
    
    feature_charts = {}
    for feat in feature_cols:
        shap_col = feat.replace('Feature_', 'SHAP_') if feat.startswith('Feature_') else f"SHAP_{feat}"
        has_shap = shap_col in jm_xgb_df.columns
        
        if has_shap:
            fig_feat = make_subplots(specs=[[{"secondary_y": True}]])
            fig_feat.update_layout(
                title=f"{feat} over Time with SHAP Impact",
                height=350,
                template="plotly_dark",
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(orientation="h", yanchor="top", y=-0.15, x=0.5, xanchor="center", bgcolor="rgba(0,0,0,0)")
            )
            
            # Secondary y-axis: SHAP Impact (Area chart)
            fig_feat.add_trace(go.Scatter(
                x=jm_xgb_df.index, y=jm_xgb_df[shap_col], mode='lines', 
                name="SHAP Impact",
                line=dict(width=0),
                fill='tozeroy',
                fillcolor='rgba(255, 140, 0, 0.3)'
            ), secondary_y=True)

            # Primary y-axis: Raw Feature
            fig_feat.add_trace(go.Scatter(
                x=jm_xgb_df.index, y=jm_xgb_df[feat], mode='lines', 
                name=f"Raw {feat}",
                line=dict(color='dodgerblue', width=1.5)
            ), secondary_y=False)
            
            fig_feat.update_yaxes(title_text="Raw Value", secondary_y=False, showgrid=False)
            fig_feat.update_yaxes(title_text="SHAP Impact", secondary_y=True, showgrid=False)
            
        else:
            fig_feat = create_base_fig(f"{feat} over Time", feat, height=350)
            
            fig_feat.add_trace(go.Scatter(
                x=jm_xgb_df.index, y=jm_xgb_df[feat], mode='lines', 
                name=feat,
                line=dict(color='dodgerblue', width=1.5)
            ))
        
        apply_bear_shading(fig_feat)
        feature_charts[feat] = fig_feat
        st.plotly_chart(fig_feat, width='stretch', key=f"feat_chart_{feat}")

# --------------------------------------------------------------------------
# JM Audit Tab
# --------------------------------------------------------------------------
with tab_jm_audit:
    st.header("JM Audit")
    st.markdown("Assess how well the Jump Model mathematically separates market returns.", help="The true 'Bear' and 'Bull' regimes are determined out-of-sample by the Jump Model. This tab compares the true JM labels.")
    
    if 'JM_Target_State' not in jm_xgb_df.columns:
        st.warning("The selected backtest cache does not contain True JM Labels. Please run a new backtest to view this tab.")
    else:
        st.subheader(
            "1. Jump Model Regime Audit",
            help="**Calculated:** Statistical summary of returns within Ground Truth regimes.\n\n**Data:** Out-of-sample `Target_Return` grouped by `JM_Target_State`.\n\n**Interpret:** Confirms the jump model successfully identified two distinct regimes. Bear (1) should ideally have a negative average return and higher volatility than Bull (0)."
        )
        st.markdown("Analyze the true financial returns for the out-of-sample periods categorized as Bull (0) or Bear (1) by the JM algorithm.", help="A good regime classification model should show distinctly lower or negative average returns and higher volatility in the Bear (1) regime compared to the Bull (0) regime.")
        
        # Calculate JM Actual Stats
        jm_returns = jm_xgb_df.groupby('JM_Target_State')['Target_Return'].agg(
            Count='count',
            Mean='mean',
            Std_Dev='std',
            Min='min',
            Max='max'
        ).rename(index={0: 'Bull (0)', 1: 'Bear (1)'})
        
        jm_returns['Ann. Return'] = jm_returns['Mean'] * 252
        jm_returns['Ann. Volatility'] = jm_returns['Std_Dev'] * np.sqrt(252)
        
        # Display as formatted dataframe
        st.dataframe(jm_returns[['Count', 'Ann. Return', 'Ann. Volatility', 'Min', 'Max']].style.format({
            'Ann. Return': '{:.2%}',
            'Ann. Volatility': '{:.2%}',
            'Min': '{:.2%}',
            'Max': '{:.2%}'
        }), width='stretch')

        # Return Distribution BoxPlot
        fig_box = go.Figure()
        
        df_bull = jm_xgb_df[jm_xgb_df['JM_Target_State'] == 0]
        df_bear = jm_xgb_df[jm_xgb_df['JM_Target_State'] == 1]
        
        fig_box.add_trace(go.Box(y=df_bull['Target_Return'], name="Bull (0)", marker_color="deepskyblue", boxmean='sd'))
        fig_box.add_trace(go.Box(y=df_bear['Target_Return'], name="Bear (1)", marker_color="crimson", boxmean='sd'))
        
        fig_box.update_layout(
            title="Distribution of Daily Returns by True JM Regime",
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis_title="Daily Return",
            template="plotly_dark",
            height=400,
            yaxis=dict(tickformat=".1%", zerolinecolor='gray')
        )
        st.plotly_chart(fig_box, width='stretch')

        # NEW: JM Regime Timeline (Price Overlay)
        st.subheader(
            "1b. JM Regime Timeline (Price Index Overlay)",
            help="**Calculated:** Benchmark price curve with background shading matching exactly when the Jump Model detected a Bear state.\n\n**Data:** S&P 500 wealth curve and True `JM_Target_State`.\n\n**Interpret:** Visually audits the 'timing' of the math. Does the red shading cover the major drawdowns perfectly without excessive false alarms?"
        )
        st.markdown("Visualize where exactly the JM model placed its regime labels on the market price curve.", help="The background shading indicates the TRUE Jump Model regime detected for that day. This helps identify if the model is 'late' to switch or if it's mis-labeling a rally as a bear regime.")
        
        fig_jm_price = create_base_fig("Benchmark Price index vs. JM Ground Truth", "Price Multiplier", height=450)
        fig_jm_price.add_trace(go.Scatter(
            x=bh_wealth.index, y=bh_wealth, mode='lines', 
            name="S&P 500",
            line=dict(color='white', width=1.5)
        ))
        
        # Apply True JM Shading
        jm_starts = jm_xgb_df.index[(jm_xgb_df['JM_Target_State'] == 1) & (~jm_xgb_df['JM_Target_State'].shift(1).fillna(0).astype(bool))]
        jm_ends = jm_xgb_df.index[(jm_xgb_df['JM_Target_State'] == 1) & (~jm_xgb_df['JM_Target_State'].shift(-1).fillna(0).astype(bool))]
        for s, e in zip(jm_starts, jm_ends):
            fig_jm_price.add_vrect(x0=s, x1=e, fillcolor="crimson", opacity=0.15, line_width=0, layer="below")
            
        st.plotly_chart(fig_jm_price, width='stretch')
        
        # NEW: Periodic Return Breakdown (Bull vs Bear)
        st.subheader(
            "1c. Periodic Return Comparison (Audit)",
            help="**Calculated:** Annualized return of the True Bull regime minus the annualized return of the True Bear regime for every 6-month window.\n\n**Data:** `Target_Return` and `JM_Target_State`.\n\n**Interpret:** Bar above 0 (Green) means the regimes separated correctly. Bar below 0 (Red) indicates a 'Regime Breakdown' where the Bear state actually rallied harder than the Bull state."
        )
        st.markdown("Examine the annualized return difference (Bull Return - Bear Return) for each 6-month evaluation chunk.", help="A POSITIVE bar means the model correctly separated a higher-yielding Bull regime from a lower-yielding Bear regime in that period. A NEGATIVE bar indicates a 'Regime Breakdown' where the identified Bear state actually had higher returns than the Bull state.")
        
        # Calculate returns by regime per period
        periodic_stats = []
        for period_start, chunk in jm_xgb_df.groupby(pd.Grouper(freq='6M')):
            if chunk.empty: continue
            
            bull_ret = chunk.loc[chunk['JM_Target_State'] == 0, 'Target_Return']
            bear_ret = chunk.loc[chunk['JM_Target_State'] == 1, 'Target_Return']
            
            # Use geometric annualized return for the chunk
            ann_bull = (1 + bull_ret).prod() ** (252 / len(chunk)) - 1 if not bull_ret.empty else 0
            ann_bear = (1 + bear_ret).prod() ** (252 / len(chunk)) - 1 if not bear_ret.empty else 0
            
            periodic_stats.append({
                'Period': period_start,
                'Bull Ann.': ann_bull,
                'Bear Ann.': ann_bear,
                'Difference': ann_bull - ann_bear
            })
            
        if periodic_stats:
            p_df = pd.DataFrame(periodic_stats)
            fig_p_break = go.Figure()
            
            fig_p_break.add_trace(go.Bar(
                x=p_df['Period'], y=p_df['Difference'],
                marker_color=['green' if v > 0 else 'red' for v in p_df['Difference']],
                name='Bull - Bear Return Delta',
                text=[f"{v:+.1%}" for v in p_df['Difference']],
                textposition='outside'
            ))
            
            fig_p_break.update_layout(
                title="Regime Separation Quality (Annualized Return Delta)",
                yaxis_title="Return Difference",
                yaxis_tickformat=".0%",
                template="plotly_dark",
                height=400,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            fig_p_break.add_hline(y=0, line_dash="solid", line_color="white", line_width=1)
            
            st.plotly_chart(fig_p_break, width='stretch')

        # NEW: Risk-Return Scatter Plot
        st.subheader(
            "1d. Regime Risk-Return separation",
            help="**Calculated:** Scatter plot of Trailing 21-Day Volatility vs Trailing 21-Day Return, colored by Ground Truth regime.\n\n**Data:** Rolling 21-Day stats of `Target_Return` colored by `JM_Target_State`.\n\n**Interpret:** A healthy model shows two distinct blobs. The Bear (red) blob should be lower and wider (choppy, low return) while the Bull (blue) blob should be clustered higher and far to the left (steady, high return)."
        )
        st.markdown("Compare the daily return and volatility characteristics of the two identified regimes.", help="This scatter plot shows the trade-off. Even if the 'Bear' regime has higher returns, it usually sits much further to the right (higher volatility), resulting in a lower Sortino or Sharpe ratio.")
        
        # Calculate trailing 21-day volatility and return for each point
        jm_xgb_df['Trailing_Vol'] = jm_xgb_df['Target_Return'].rolling(21).std() * np.sqrt(252)
        jm_xgb_df['Trailing_Ret'] = jm_xgb_df['Target_Return'].rolling(21).mean() * 252
        
        fig_scatter = go.Figure()
        
        for state, label, color in [(0, 'Bull (0)', 'deepskyblue'), (1, 'Bear (1)', 'crimson')]:
            mask = jm_xgb_df['JM_Target_State'] == state
            fig_scatter.add_trace(go.Scatter(
                x=jm_xgb_df.loc[mask, 'Trailing_Vol'],
                y=jm_xgb_df.loc[mask, 'Trailing_Ret'],
                mode='markers',
                name=label,
                marker=dict(color=color, opacity=0.4, size=6),
                text=[f"Date: {d.date()}" for d in jm_xgb_df.index[mask]],
                hoverinfo="text+x+y"
            ))
            
        fig_scatter.update_layout(
            title="Trailing 21-Day Risk vs Return by Regime",
            xaxis_title="Annualized Volatility (Trailing 21D)",
            yaxis_title="Annualized Return (Trailing 21D)",
            yaxis_tickformat=".0%",
            xaxis_tickformat=".0%",
            template="plotly_dark",
            height=450,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig_scatter, width='stretch')
        
        # NEW: Regime History Audit Table
        st.subheader(
            "1e. JM Regime History Audit Table",
            help="**Calculated:** Contiguous chunks of identical JM regimes are collapsed into single rows with aggregate stats.\n\n**Data:** Continuous segments of `JM_Target_State`.\n\n**Interpret:** Useful to see if the XGBoost algorithm struggles during long regimes vs short choppy regimes."
        )
        st.markdown("A granular breakdown of every contiguous regime detected by the Jump Model, including performance and XGBoost accuracy during those specific windows.", help="This table groups consecutive days of the same JM regime into 'segments'. It allows you to see if XGBoost performs better in long, stable regimes versus choppy, short-lived ones.")
        
        # Identify contiguous segments
        jm_states = jm_xgb_df['JM_Target_State']
        changes = (jm_states != jm_states.shift(1)).cumsum()
        
        # Ensure we have the lambda time series mapped to the daily index
        lambda_ts = pd.Series(lambda_history, index=pd.to_datetime(lambda_dates))
        lambda_daily = lambda_ts.reindex(jm_xgb_df.index.union(lambda_ts.index)).ffill().bfill().loc[jm_xgb_df.index]
        
        # Calculate XGB Accuracy properly (aligned t vs t+1)
        jm_xgb_df['Next_True_JM'] = jm_xgb_df['JM_Target_State'].shift(-1)
        jm_xgb_df['XGB_Correct'] = (jm_xgb_df['Forecast_State'] == jm_xgb_df['Next_True_JM']).astype(float)
        
        regime_segments = []
        for segment_id, group in jm_xgb_df.groupby(changes):
            start_dt = group.index[0]
            end_dt = group.index[-1]
            state = group['JM_Target_State'].iloc[0]
            
            # Segment returns
            seg_returns = group['Target_Return']
            total_ret = (1 + seg_returns).prod() - 1
            
            # Segment Drawdown
            wealth = (1 + seg_returns).cumprod()
            peak = wealth.cummax()
            dd = (wealth - peak) / peak
            max_dd = dd.min()
            
            # Forecast accuracy in THIS segment
            valid_forecasts = group['XGB_Correct'].dropna()
            acc = valid_forecasts.mean() if not valid_forecasts.empty else 0
            
            # Segment Volatility
            seg_vol = seg_returns.std() * np.sqrt(252)
            
            # Segment Lambda (Average)
            seg_lambda = lambda_daily.loc[group.index].mean()
            
            regime_segments.append({
                'Start': start_dt.date(),
                'End': end_dt.date(),
                'Days': len(group),
                'Regime': 'Bull (0)' if state == 0 else 'Bear (1)',
                'Return': total_ret,
                'Vol': seg_vol,
                'Max DD': max_dd,
                'Lambda': seg_lambda,
                'XGB Accuracy': acc
            })
            
        regime_history_df = pd.DataFrame(regime_segments).sort_values('Start', ascending=False)
        
        # Add filtering UI
        col_f1, _ = st.columns([2, 3])
        with col_f1:
            filter_regimes = st.multiselect("Filter by Regime", options=['Bull (0)', 'Bear (1)'], default=['Bull (0)', 'Bear (1)'])
            
        filtered_history_df = regime_history_df[regime_history_df['Regime'].isin(filter_regimes)]
        
        st.dataframe(
            filtered_history_df.style.format({
                'Return': '{:.2%}',
                'Vol': '{:.2%}',
                'Max DD': '{:.2%}',
                'Lambda': '{:.1f}',
                'XGB Accuracy': '{:.1%}'
            }).background_gradient(subset=['XGB Accuracy'], cmap='RdYlGn', vmin=0.3, vmax=0.7)
            .apply(lambda x: [
                'background-color: rgba(220, 20, 60, 0.2); color: crimson' if v == 'Bear (1)' 
                else 'background-color: rgba(0, 191, 255, 0.15); color: deepskyblue' for v in x
            ], subset=['Regime']),
            hide_index=True,
            width='stretch',
            column_config={
                "Start": st.column_config.DateColumn("Start", help="The date this regime segment began."),
                "End": st.column_config.DateColumn("End", help="The date this regime segment ended."),
                "Days": st.column_config.NumberColumn("Days", help="Total number of trading days in this segment."),
                "Regime": st.column_config.TextColumn("Regime", help="The Ground Truth regime identified by the Jump Model."),
                "Return": st.column_config.NumberColumn("Return", help="Total geometric return during this specific segment."),
                "Vol": st.column_config.NumberColumn("Vol", help="Annualized volatility of daily returns during this segment."),
                "Max DD": st.column_config.NumberColumn("Max DD", help="The maximum peak-to-trough decline experienced during this segment."),
                "Lambda": st.column_config.NumberColumn("Lambda", help="The average Lambda penalty active during this period (higher = more regime stability)."),
                "XGB Accuracy": st.column_config.NumberColumn(
                    "XGB Accuracy",
                    help="Directional accuracy of XGBoost: % of days where prediction at (t) correctly anticipated the JM label at (t+1)."
                )
            }
        )
        
        st.markdown("---")
        
        # NEW: 1f. Feature Distributions by JM Regime
        st.subheader(
            "1f. Feature Distributions by JM Regime",
            help="**Calculated:** Box plot of raw input features separated by ground truth regime.\n\n**Data:** User-selected raw feature grouped by `JM_Target_State`.\n\n**Interpret:** Box plots without much overlap prove the feature engineer created a variable that easily separates the two market states."
        )
        st.markdown("Analyze how the raw input features are distributed across the True JM regimes.", help="This helps verify if the feature engineering step produces features that are distinctly different between the two regimes. If the distributions overlap heavily, the clustering algorithm will struggle.")
        
        # Get raw features (excluding SHAP and other metadata)
        exclude_dist_cols = ['Target_Return', 'RF_Rate', 'Forecast_State', 'Strat_Return', 'Trades', 'State_Prob', 'SHAP_Base_Value', 'JM_Target_State', 'Trailing_Vol', 'Trailing_Ret', 'Next_True_JM', 'XGB_Correct']
        dist_features = [c for c in jm_xgb_df.columns if c.startswith('Feature_')]
        
        if dist_features:
            selected_dist_feat = st.selectbox("Select Feature to view Distribution", [f.replace('Feature_', '') for f in dist_features])
            
            fig_dist = go.Figure()
            feat_col = f"Feature_{selected_dist_feat}"
            
            fig_dist.add_trace(go.Box(
                y=jm_xgb_df.loc[jm_xgb_df['JM_Target_State'] == 0, feat_col], 
                name="Bull (0)", 
                marker_color="deepskyblue", 
                boxmean='sd'
            ))
            fig_dist.add_trace(go.Box(
                y=jm_xgb_df.loc[jm_xgb_df['JM_Target_State'] == 1, feat_col], 
                name="Bear (1)", 
                marker_color="crimson", 
                boxmean='sd'
            ))
            
            fig_dist.update_layout(
                title=f"Distribution of {selected_dist_feat} by True JM Regime",
                margin=dict(l=20, r=20, t=50, b=20),
                yaxis_title=selected_dist_feat,
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig_dist, width='stretch')
        
        # NEW: 1g. Regime Feature Scatter Plot
        st.subheader(
            "1g. Regime Feature Scatter Plot",
            help="**Calculated:** 2D Scatter plot mapping raw feature A vs feature B.\n\n**Data:** `JM_Target_State` labels mapped onto a 2D feature space.\n\n**Interpret:** Shows the exact multidimensional boundaries drawn by the Jump Model's clustering algorithm to define Bull vs Bear markets."
        )
        st.markdown("Visualize the multi-dimensional space where the Jump Model draws its clustering boundaries.", help="Select two features to see how they separate the regimes. The K-means algorithm tries to find clusters in this higher-dimensional space.")
        
        if len(dist_features) >= 2:
            col_scat1, col_scat2 = st.columns(2)
            clean_dist_features = [f.replace('Feature_', '') for f in dist_features]
            
            with col_scat1:
                feat_x = st.selectbox("Select X-Axis Feature", clean_dist_features, index=0)
            with col_scat2:
                # Try to default to a volatility/risk feature for the Y-axis if available
                default_y_idx = 1
                for i, f in enumerate(clean_dist_features):
                    if 'DD' in f or 'Vol' in f or 'VIX' in f:
                        default_y_idx = i
                        break
                feat_y = st.selectbox("Select Y-Axis Feature", clean_dist_features, index=default_y_idx)
                
            fig_feat_scatter = go.Figure()
            
            for state, label, color in [(0, 'Bull (0)', 'deepskyblue'), (1, 'Bear (1)', 'crimson')]:
                mask = jm_xgb_df['JM_Target_State'] == state
                fig_feat_scatter.add_trace(go.Scatter(
                    x=jm_xgb_df.loc[mask, f"Feature_{feat_x}"],
                    y=jm_xgb_df.loc[mask, f"Feature_{feat_y}"],
                    mode='markers',
                    name=label,
                    marker=dict(color=color, opacity=0.5, size=6),
                    text=[f"Date: {d.date()}<br>Ret: {r:.2%}" for d, r in zip(jm_xgb_df.index[mask], jm_xgb_df.loc[mask, 'Target_Return'])],
                    hoverinfo="text+x+y"
                ))
                
            fig_feat_scatter.update_layout(
                title=f"{feat_y} vs {feat_x} by Regime",
                xaxis_title=feat_x,
                yaxis_title=feat_y,
                template="plotly_dark",
                height=500,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_feat_scatter, width='stretch')

        # NEW: 1h. Misclassified Regimes Deep Dive (The "Why" Table)
        st.subheader(
            "1h. Misclassified Regimes Deep Dive (Anomalies)",
            help="**Calculated:** Filters data for days where the True JM Regime label opposes the actual daily return (e.g. Bear states that have positive returns).\n\n**Data:** `JM_Target_State` and `Target_Return`.\n\n**Interpret:** Audits the jump model's confidence. If a Bear day had a +3% return, looking at the features might reveal it was due to extreme underlying volatility, justifying the Bear label despite the rally."
        )
        st.markdown("Investigate specific days where the True JM Regime label seems counter-intuitive based solely on return.", help="Filter for Bear regimes with positive returns, or Bull regimes with negative returns, to see the underlying feature values that drove the clustering decision.")
        
        anomaly_type = st.radio(
            "Select Anomaly Type to Investigate:",
            options=["Bear Regimes with Positive Returns", "Bull Regimes with Negative Returns"]
        )
        
        if anomaly_type == "Bear Regimes with Positive Returns":
            anomaly_mask = (jm_xgb_df['JM_Target_State'] == 1) & (jm_xgb_df['Target_Return'] > 0)
        else:
            anomaly_mask = (jm_xgb_df['JM_Target_State'] == 0) & (jm_xgb_df['Target_Return'] < 0)
            
        anomaly_df = jm_xgb_df[anomaly_mask].copy()
        
        if not anomaly_df.empty:
            st.write(f"Found {len(anomaly_df)} days matching this criteria.")
            
            # Add a 'Return Magnitude' filter to focus on the worst offenders
            min_mag = st.slider("Minimum Return Magnitude Threshold (Absolute %)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
            
            filtered_anomaly_df = anomaly_df[anomaly_df['Target_Return'].abs() >= (min_mag / 100)]
            
            if not filtered_anomaly_df.empty:
                 # Select columns to display: Date, Regime, Return, and Top Features
                 # Assuming SHAP values can tell us what features were important to the *model* globally
                 # For the *clustering*, we just show the raw features.
                 display_cols = ['Target_Return', 'JM_Target_State'] + [c for c in dist_features if 'Sortino' in c or 'DD' in c or 'Avg_Ret' in c]
                 
                 disp_df = filtered_anomaly_df[display_cols].copy()
                 disp_df.index = disp_df.index.date
                 disp_df.index.name = 'Date'
                 
                 # Clean up feature names for display
                 disp_df = disp_df.rename(columns={c: c.replace('Feature_', '') for c in disp_df.columns})
                 disp_df = disp_df.rename(columns={'JM_Target_State': 'Regime', 'Target_Return': 'Return'})
                 
                 st.dataframe(
                     disp_df.style.format({
                         'Return': '{:.2%}',
                         **{c: '{:.4f}' for c in disp_df.columns if c not in ['Return', 'Regime']}
                     }).apply(lambda x: [
                        'background-color: rgba(220, 20, 60, 0.2); color: crimson' if v == 1 
                        else 'background-color: rgba(0, 191, 255, 0.15); color: deepskyblue' for v in x
                    ], subset=['Regime']),
                    width='stretch'
                 )
            else:
                st.info(f"No anomalies found with magnitude >= {min_mag}%.")
        else:
            st.success("No anomalies of this type found in the current dataset.")

        # NEW: 1i. Regime Feature Drift Over Time
        st.subheader(
            "1i. Regime Feature Drift Over Time",
            help="**Calculated:** Tracks the moving average of a raw feature within each regime bucket over time.\n\n**Data:** Selected feature grouped by 6-month intervals and `JM_Target_State`.\n\n**Interpret:** Identifies regime drift. If the red line moves significantly over the years, it implies the definition of a 'Bear Market' according to this feature is slowly changing."
        )
        st.markdown("Track the average value of a specific feature within Bull vs Bear regimes across each 6-month evaluation period.", help="This helps identify if the definition of a 'Bear' market is changing over time. For example, does a Bear market in 2020 have a different average Sortino ratio than a Bear market in 2008?")
        
        if dist_features:
            selected_drift_feat = st.selectbox("Select Feature to view Drift", [f.replace('Feature_', '') for f in dist_features], index=0)
            feat_col = f"Feature_{selected_drift_feat}"
            
            drift_stats = []
            for period_start, chunk in jm_xgb_df.groupby(pd.Grouper(freq='6M')):
                if chunk.empty: continue
                
                bull_vals = chunk.loc[chunk['JM_Target_State'] == 0, feat_col]
                bear_vals = chunk.loc[chunk['JM_Target_State'] == 1, feat_col]
                
                drift_stats.append({
                    'Period': period_start,
                    'Bull Avg': bull_vals.mean() if not bull_vals.empty else np.nan,
                    'Bear Avg': bear_vals.mean() if not bear_vals.empty else np.nan
                })
                
            if drift_stats:
                drift_df = pd.DataFrame(drift_stats)
                
                fig_drift = create_base_fig(f"Average {selected_drift_feat} over Time by Regime", selected_drift_feat, height=400)
                
                fig_drift.add_trace(go.Scatter(
                    x=drift_df['Period'], y=drift_df['Bull Avg'],
                    mode='lines+markers', name='Bull (0) Avg',
                    line=dict(color='deepskyblue', width=2)
                ))
                
                fig_drift.add_trace(go.Scatter(
                    x=drift_df['Period'], y=drift_df['Bear Avg'],
                    mode='lines+markers', name='Bear (1) Avg',
                    line=dict(color='crimson', width=2)
                ))
                
                st.plotly_chart(fig_drift, width='stretch')
        
        # NEW: 1j. Export JM Audit to LLM
        st.subheader("1j. Export JM Audit to LLM")
        st.markdown("Generate a minimal JSON export of the regime labeling and anomaly data to audit with an LLM.")
        
        dialog_func = getattr(st, "dialog", getattr(st, "experimental_dialog", None))
        
        def generate_jm_audit_export():
            # 1. JM Overall Stats
            jm_stats = jm_returns[['Count', 'Ann. Return', 'Ann. Volatility', 'Min', 'Max']].copy()
            jm_stats.index = [str(x) for x in jm_stats.index]
            
            # 2. Regime History (limit to save tokens, or round)
            history_export = regime_history_df.copy()
            history_export['Start'] = history_export['Start'].astype(str)
            history_export['End'] = history_export['End'].astype(str)
            for c in ['Return', 'Vol', 'Max DD', 'Lambda', 'XGB Accuracy']:
                if c in history_export.columns:
                    history_export[c] = history_export[c].round(4)
                    
            # 3. Top Anomalies
            bear_pos = jm_xgb_df[(jm_xgb_df['JM_Target_State'] == 1) & (jm_xgb_df['Target_Return'] > 0)].copy()
            bull_neg = jm_xgb_df[(jm_xgb_df['JM_Target_State'] == 0) & (jm_xgb_df['Target_Return'] < 0)].copy()
            
            # Get largest anomalies (Top 10 of each) to save tokens
            bear_pos = bear_pos.sort_values('Target_Return', ascending=False).head(10)
            bull_neg = bull_neg.sort_values('Target_Return', ascending=True).head(10)
            
            anomaly_cols = ['Target_Return', 'JM_Target_State'] + [c for c in dist_features if 'Sortino' in c or 'DD' in c or 'Avg_Ret' in c]
            
            def format_anomaly_df(df):
                if df.empty: return []
                d = df[anomaly_cols].copy()
                d.index = d.index.astype(str)
                d = d.round(4)
                d = d.rename(columns={c: c.replace('Feature_', '') for c in d.columns})
                return d.reset_index().to_dict('records')
                
            export_data = {
                "jm_overall_stats": jm_stats.round(4).to_dict('index'),
                "regime_history_segments": history_export.to_dict('records'),
                "top_10_bear_pos_anomalies": format_anomaly_df(bear_pos),
                "top_10_bull_neg_anomalies": format_anomaly_df(bull_neg)
            }
            return export_data
            
        if dialog_func:
            @dialog_func("JM Audit LLM Export")
            def show_jm_audit_export():
                st.markdown("Copy the JSON below and share it with your LLM.")
                export_dict = generate_jm_audit_export()
                st.code(json.dumps(export_dict, separators=(',', ':')), language="json")
                
            if st.button("🤖 Export JM Audit Context to LLM"):
                show_jm_audit_export()
        else:
            with st.expander("🤖 Export JM Audit Context to LLM"):
                st.markdown("Copy the JSON below and share it with your LLM.")
                export_dict = generate_jm_audit_export()
                st.code(json.dumps(export_dict, separators=(',', ':')), language="json")
                
        st.markdown("---")
# --------------------------------------------------------------------------
# XGBoost Forecast Evaluation Tab
# --------------------------------------------------------------------------
with tab_xgb_eval:
    st.header("XGBoost Forecast Evaluation")
    st.markdown("Assess how accurately XGBoost predicts the underlying regimes.", help="XGBoost attempts to forecast these regimes one day in advance. This tab compares the true JM labels against the XGBoost forecasts.")
    
    if 'JM_Target_State' not in jm_xgb_df.columns:
        st.warning("The selected backtest cache does not contain True JM Labels. Please run a new backtest to view this tab.")
    else:
        # XGBoost Evaluation
        st.subheader(
            "1. XGBoost Forecast Quality",
            help="**Calculated:** Classification metrics (Accuracy, Precision, Recall, F1) comparing XGBoost prediction at (t) vs True JM Label at (t+1).\n\n**Data:** `Forecast_State` compared against a 1-day shifted `JM_Target_State`.\n\n**Interpret:** High precision means when it calls a Bear market, it's usually right. High recall means it rarely misses an actual Bear market. The Confusion Matrix visualized the Raw Trade-Off."
        )
        st.markdown("Evaluate the classification performance of XGBoost in predicting the next day's true JM regime.", help="Remember that XGBoost predicts the JM regime shifted by 1 day (tomorrow's state). Thus, a perfect forecast matches today's prediction with tomorrow's true state.")
        
        # Align XGBoost Forecast (t) with JM Regime (t+1)
        # The true target for the forecast made at time t is the JM state at t+1.
        true_labels = jm_xgb_df['JM_Target_State'].shift(-1).dropna()
        pred_labels = jm_xgb_df['Forecast_State'].loc[true_labels.index]
        
        from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
        
        if len(true_labels) > 0:
            acc = accuracy_score(true_labels, pred_labels)
            prec = precision_score(true_labels, pred_labels, zero_division=0)
            rec = recall_score(true_labels, pred_labels, zero_division=0)
            f1 = f1_score(true_labels, pred_labels, zero_division=0)
            
            # Metrics Row
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Accuracy", f"{acc:.2%}", help="Overall proportion of correct predictions.")
            col_m2.metric("Precision (Bear)", f"{prec:.2%}", help="When XGBoost predicts Bear, how often is it actually Bear?")
            col_m3.metric("Recall (Bear)", f"{rec:.2%}", help="Out of all actual Bear days, how many did XGBoost correctly identify?")
            col_m4.metric("F1-Score (Bear)", f"{f1:.2f}", help="Harmonic mean of Precision and Recall.")
            
            # Confusion Matrix Plotly
            cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm[::-1], # Reverse rows for display matching standard layout
                x=["Predicted Bull", "Predicted Bear"],
                y=["Actual Bear", "Actual Bull"], # Reverse y labels correspondingly
                hoverongaps=False,
                colorscale="Blues",
                text=[[str(v) for v in row] for row in cm[::-1]],
                texttemplate="%{text}",
                textfont={"size": 16}
            ))
            fig_cm.update_layout(
                title="Confusion Matrix (Predicting t+1 Regime)",
                height=400,
                width=400,
                template="plotly_dark",
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            # Center the confusion matrix
            col_c1, col_c2, col_c3 = st.columns([1, 2, 1])
            with col_c2:
                st.plotly_chart(fig_cm, width='stretch')
                
            # --------------------------------------------------------------------------
            # Chart 2: Regime Probability vs Benchmark Price
            # --------------------------------------------------------------------------
            st.subheader(
                "2. Regime Probability vs. Market Price",
                help="**Calculated:** Overlay of the S&P 500 wealth index and the XGBoost model's predicted bear regime probability.\n\n**Data:** Market price index and predicted `State_Prob` from the XGBoost model.\n\n**Interpret:** Check if high-conviction bear calls (red area spikes) precede or coincide with actual market drops."
            )
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

            # Rolling Metrics
            st.subheader(
                "3. Rolling Forecast Accuracy",
                help="**Calculated:** 126-day (approx 6-month) simple moving average of a boolean 'Correct / Incorrect' array.\n\n**Data:** Daily boolean match between XGBoost (t) and True JM (t+1).\n\n**Interpret:** Shows model stability over time. Dips below the 50% line mean the model is performing worse than a coin toss during that specific 6 month window."
            )
            st.markdown("Track the 6-month rolling accuracy of XGBoost's regime predictions.", help="Helps identify periods where the model struggles to accurately forecast the underlying regimes.")
            
            # Create a dataframe for rolling calculation
            roll_df = pd.DataFrame({'True': true_labels, 'Pred': pred_labels})
            roll_df['Correct'] = (roll_df['True'] == roll_df['Pred']).astype(int)
            rolling_acc = roll_df['Correct'].rolling(window=126).mean()  # 126 days is approx 6 months
            
            fig_roll = create_base_fig("6-Month Rolling Classification Accuracy", "Accuracy", height=350)
            fig_roll.add_trace(go.Scatter(
                x=rolling_acc.index, y=rolling_acc, mode='lines', 
                name="Rolling Accuracy",
                line=dict(color='orange', width=2)
            ))
            fig_roll.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Coin Toss (50%)")
            fig_roll.update_yaxes(tickformat=".0%", range=[0, 1])
            
            apply_bear_shading(fig_roll)
            st.plotly_chart(fig_roll, width='stretch')


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
        
        
        # Format the duration string dynamically
        duration_str = "N/A"
        if backtest_duration is not None:
            if backtest_duration >= 3600:
                h = int(backtest_duration // 3600)
                m = int((backtest_duration % 3600) // 60)
                s = int(backtest_duration % 60)
                duration_str = f"{h}h {m}m {s}s"
            elif backtest_duration >= 60:
                m = int(backtest_duration // 60)
                s = int(backtest_duration % 60)
                duration_str = f"{m}m {s}s"
            else:
                duration_str = f"{backtest_duration:.2f} seconds"
        
        def compress_fig(fig, max_points=6):
            if fig is None: return {}
            compressed = {}
            for trace in getattr(fig, 'data', []):
                name = getattr(trace, 'name', None) or "Unnamed"
                if getattr(trace, 'showlegend', True) is False and name == "Unnamed": 
                    continue
                
                trace_type = getattr(trace, 'type', 'scatter')
                
                # Handling Heatmap
                if trace_type == 'heatmap':
                    z = getattr(trace, 'z', None)
                    if z is not None:
                        try:
                            z_flat = [float(round(v, 3)) if isinstance(v, (float, np.floating)) else int(v) for row in z for v in row]
                            compressed[name] = z_flat[:max_points*max_points]
                        except Exception:
                            pass
                    continue
                    
                # Handling Box
                if trace_type == 'box':
                    y = getattr(trace, 'y', None)
                    if y is not None and len(y) > 0:
                        y_num = [val for val in y if isinstance(val, (int, float, np.integer, np.floating))]
                        if y_num:
                            compressed[name] = [
                                float(round(np.min(y_num), 3)),
                                float(round(np.percentile(y_num, 25), 3)),
                                float(round(np.median(y_num), 3)),
                                float(round(np.percentile(y_num, 75), 3)),
                                float(round(np.max(y_num), 3))
                            ]
                    continue

                # Handling horizontal bar charts (like SHAP summary)
                if trace_type == 'bar' and getattr(trace, 'orientation', 'v') == 'h':
                    x = getattr(trace, 'x', [])
                    y = getattr(trace, 'y', [])
                    if x is not None and y is not None:
                        pairs = list(zip(y, x))
                        if len(pairs) > max_points * 2:
                            pairs = pairs[:max_points] + pairs[-max_points:]
                        compressed[name] = {str(k): float(round(v, 3)) if isinstance(v, (float, np.floating)) else v for k, v in pairs}
                    continue
                
                # Handling standard scatter/lines/bar
                y = getattr(trace, 'y', None)
                if y is None or len(y) == 0: 
                    continue
                
                y_list = list(y)
                y_num = [val for val in y_list if isinstance(val, (int, float, np.integer, np.floating))]
                if not y_num:
                    continue
                
                if len(y_num) > max_points:
                    step = (len(y_num) - 1) / max(1, max_points - 1)
                    sampled_y = [y_num[int(round(i * step))] for i in range(max_points)]
                else:
                    sampled_y = y_num
                
                sampled_y = [float(round(val, 3)) if isinstance(val, (float, np.floating)) else int(val) if isinstance(val, np.integer) else val for val in sampled_y]
                compressed[name] = sampled_y
                
            return {k: v for k, v in compressed.items() if v}

        g = globals()
        figs = {
            "Wealth": g.get('fig_wealth'),
            "Drawdown": g.get('fig_dd'),
            "Regime_Prob": g.get('fig_prob'),
            "Sortino": g.get('fig_sortino'),
            "Monthly_Heatmap": g.get('fig_heatmap'),
            "Return_Dist": g.get('fig_dist'),
            "Risk_Return_Scatter": g.get('fig_scatter'),
            "Trades_Lambda": g.get('fig_trades'),
            "SHAP_TS": g.get('fig_shap_ts'),
            "SHAP_Summary": g.get('fig_shap_summary'),
            "Feature_Dependence": g.get('fig_dep'),
            "SHAP_Point_in_Time": g.get('fig_point'),
            "JM_Return_Dist": g.get('fig_box'),
            "JM_Regime_Timeline": g.get('fig_jm_price'),
            "JM_Periodic_Breakdown": g.get('fig_p_break'),
            "JM_Risk_Return": g.get('fig_scatter'),
            "XGB_Confusion_Matrix": g.get('fig_cm'),
            "XGB_Rolling_Acc": g.get('fig_roll'),
        }
        
        fc = g.get('feature_charts', {})
        for k, v in fc.items():
            figs[f"Feat_{k}"] = v

        exported_charts = {k: compress_fig(v) for k, v in figs.items() if v is not None and compress_fig(v)}
        
        return {
            "params": {
                "tgt": backend.TARGET_TICKER, 
                "bnd": backend.BOND_TICKER, 
                "rf": backend.RISK_FREE_TICKER,
                "vix": backend.VIX_TICKER,
                "cost": backend.TRANSACTION_COST, 
                "oos": f"{filter_start} to {filter_end}",
                "lambda_sel": lambda_history[-1] if lambda_history else "N/A",
                "backtest_duration": duration_str
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
            ],
            "charts": exported_charts
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
            "lambda_sel": "Final Lambda Penalty",
            "backtest_duration": "Backtest Duration"
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
            "Rolling 1-Year Sortino": fig_sortino,
            "Monthly Returns Heatmap": fig_heatmap,
            "Return Distribution": fig_dist,
            "Risk / Return Profile": fig_scatter,
            "Transactions & Lambda": fig_trades,
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
                import base64
                import streamlit.components.v1 as components
                b64 = base64.b64encode(pdf_bytes).decode('utf-8')
                href = f"""
                <a href="data:application/pdf;base64,{b64}" download="xgb_{current_time_file}.pdf" id="download_link"></a>
                <script>
                    document.getElementById("download_link").click();
                </script>
                """
                components.html(href, height=0)

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
