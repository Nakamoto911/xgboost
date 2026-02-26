import sys
import os
import pickle
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Mock the dataframe and strategies to create the layout
jm_xgb_df = pd.DataFrame({'State_Prob': [0.1, 0.9], 'Trades': [1, 2]}, index=pd.date_range('2020-01-01', periods=2))
bh_wealth = pd.Series([1, 1.1], index=pd.date_range('2020-01-01', periods=2))
lambda_history_full = [10, 20]
lambda_dates_full = pd.date_range('2020-01-01', periods=2)
trades_by_period = pd.Series([1, 2], index=pd.date_range('2020-01-01', periods=2))
ret_per = pd.Series([0.1, -0.05], index=pd.date_range('2020-01-01', periods=2))
vol_per = pd.Series([0.15, 0.12], index=pd.date_range('2020-01-01', periods=2))

fig_plotly = make_subplots(
    rows=7, cols=1, shared_xaxes=True, 
    vertical_spacing=0.03,
    row_heights=[0.15, 0.15, 0.15, 0.20, 0.10, 0.125, 0.125],
    specs=[
        [{"secondary_y": False}],
        [{"secondary_y": True}],
        [{"secondary_y": False}],
        [{"secondary_y": False}],
        [{"secondary_y": True}],
        [{"secondary_y": False}],
        [{"secondary_y": False}]
    ]
)

# Row 1
fig_plotly.add_trace(go.Scatter(x=jm_xgb_df.index, y=[0, -0.1], mode='lines'), row=1, col=1)

# Row 2
fig_plotly.add_trace(go.Scatter(x=bh_wealth.index, y=bh_wealth, mode='lines'), row=2, col=1, secondary_y=False)
fig_plotly.add_trace(go.Scatter(x=jm_xgb_df.index, y=jm_xgb_df['State_Prob'], mode='lines'), row=2, col=1, secondary_y=True)

# Row 3
fig_plotly.add_trace(go.Scatter(x=jm_xgb_df.index, y=[0.5, 0.8], mode='lines'), row=3, col=1)

# Row 4
fig_plotly.add_trace(go.Bar(x=jm_xgb_df.index, y=[0.1, 0.2]), row=4, col=1)

# Row 5
fig_plotly.add_trace(go.Scatter(x=trades_by_period.index, y=trades_by_period), row=5, col=1, secondary_y=False)
fig_plotly.add_trace(go.Scatter(x=lambda_dates_full, y=lambda_history_full), row=5, col=1, secondary_y=True)

# Row 6
fig_plotly.add_trace(go.Scatter(x=ret_per.index, y=ret_per), row=6, col=1)

# Row 7
fig_plotly.add_trace(go.Scatter(x=vol_per.index, y=vol_per), row=7, col=1)


layout = fig_plotly.to_dict()['layout']
for key, value in layout.items():
    if key.startswith('yaxis'):
        print(f"{key}: {value.get('domain', 'No domain set')}")
