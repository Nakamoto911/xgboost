import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=1)

fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2], name="Tr1"), row=1, col=1)
fig.data[-1].legend = "legend"

fig.add_trace(go.Scatter(x=[1, 2], y=[2, 1], name="Tr2"), row=2, col=1)
fig.data[-1].legend = "legend2"

fig.update_layout(
    legend=dict(y=0.8, x=1.05),
    legend2=dict(y=0.2, x=1.05)
)

print(fig.to_dict()['layout'].keys())
print("Test passed without errors")
