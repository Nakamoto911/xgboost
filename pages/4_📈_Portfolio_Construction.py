"""
Portfolio Construction — reproduces Tables 6, 7 and Figure 3 from
Shu, Yu & Mulvey (2024) "Dynamic Asset Allocation with Asset-Specific Regime
Forecasts" (arXiv:2406.09578v2).

Default data source: Bloomberg total return indices (cache/DATA PAUL.xlsx).
Alternative: 12 Yahoo ETFs.
"""
import os
import sys
import importlib

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Ensure project root is on path so `import portfolio` works regardless of CWD
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import portfolio
import main as backend
importlib.reload(portfolio)

st.set_page_config(layout="wide", page_title="Portfolio Construction (Tables 6/7, Fig 3)")

st.title("📈 Portfolio Construction — Tables 6 & 7, Figure 3")
st.caption("Reproduces the Shu/Yu/Mulvey (2024) MVO/MV/EW portfolios over 12 assets. "
           "Default: Bloomberg total return indices (matches paper). "
           "Heavy per-asset signals are cached on disk; click **Force Refresh** to recompute.")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    universe = st.selectbox(
        "Asset universe",
        options=["bloomberg", "yahoo"],
        index=0,
        format_func=lambda x: {"bloomberg": "Bloomberg (DATA PAUL.xlsx) — 12 assets",
                               "yahoo": "Yahoo ETFs — 12 ETFs"}[x],
        help="Bloomberg matches the paper. Yahoo ETFs use IVV/IJH/IWM/EFA/EEM/IYR/AGG/SPTL/HYG/SPBO/DBC/GLD.",
    )

    if universe == "bloomberg":
        oos_start_default = "2007-01-01"
        oos_end_default   = "2023-12-31"
    else:
        # Yahoo ETFs: start pinned at 2010 by ETF inception (SPBO inception 2012, others 2007-2010),
        # but end matched to the paper's 2023-12-31 for apples-to-apples comparison.
        oos_start_default = "2010-01-01"
        oos_end_default   = "2023-12-31"

    oos_start = st.text_input("OOS start", value=oos_start_default,
                              help="Out-of-sample start date (paper: 2007-01-01).")
    oos_end   = st.text_input("OOS end",   value=oos_end_default,
                              help="Out-of-sample end date (paper: 2023-12-31).")

    st.markdown("**TC mode**")
    tc_mode = st.radio(
        "Transaction cost reporting",
        options=["net", "gross"],
        index=0,
        format_func=lambda x: {"net":   "Net (5 bps) — realistic investor",
                               "gross": "Gross (paper-faithful, TC=0)"}[x],
        horizontal=False,
        help="Net: 5 bps applied during walk-forward λ-selection AND portfolio "
             "rebalancing — what you would actually earn. "
             "Gross: TC=0 in both layers — apples-to-apples with paper Tables 4/6/7 "
             "(reported gross of TC, Session 19 finding). Gross signals cache "
             "separately and require a one-time recompute.",
    )

    rebal_freq = st.selectbox(
        "Rebalancing frequency",
        options=["daily", "monthly", "quarterly", "biannually", "yearly"],
        index=0,
        help="The paper rebalances daily.",
    )

    st.markdown("**MVO parameters**")
    gamma_risk_minvar = st.number_input("γ_risk (MinVar)",   value=10.0, step=1.0)
    gamma_risk_mv_b   = st.number_input("γ_risk (MV base)",  value=5.0,  step=1.0)
    gamma_risk_mv_j   = st.number_input("γ_risk (MV JM-XGB)", value=10.0, step=1.0)
    gamma_trade       = st.number_input("γ_trade",            value=1.0,  step=0.25,
                                        help="Trading-cost penalty multiplier (paper default 1.0). "
                                             "Reduces turnover. With OOS-forecast μ this can collapse "
                                             "leverage; with in-sample μ (paper-faithful, see toggle "
                                             "below) the larger μ values support full deployment.")
    # Portfolio-level TC default mirrors the selected mode; user can still override.
    _tc_default = 0.0 if tc_mode == "gross" else 0.0005
    tc_oneway = st.number_input("Transaction cost (one-way, decimal)",
                                value=_tc_default, step=0.0001, format="%.4f",
                                help=f"Defaults to {_tc_default} for tc_mode='{tc_mode}'. "
                                     "Override to decouple portfolio-level TC from signal-level.")
    w_ub              = st.number_input("w_ub (max weight per asset)",
                                        value=0.40, step=0.05, format="%.2f")
    cov_hl_days       = st.number_input("Σ EWM halflife (days)", value=252, step=21,
                                        help="EWM covariance halflife (paper: 252 days).")
    mu_baseline_hl_yr = st.number_input("MV baseline μ EWM hl (years)", value=5.0,
                                        step=1.0, format="%.1f")
    mu_jmxgb_lb_yr    = st.number_input("MV(JM-XGB) μ lookback (years)", value=11.0,
                                        step=1.0, format="%.1f",
                                        help="Window for regime-conditional historical mean (paper: 11y).")

    st.markdown("**μ specification for MV(JM-XGB)**")
    use_insample_mu = st.checkbox(
        "Use in-sample JM regime means (paper-faithful)",
        value=True,
        help="Paper Section 4.5 conditions on the in-sample JM regime labels — clean "
             "bull-day vs bear-day separation. When off, falls back to conditioning on "
             "the OOS-forecast labels (noisier, smaller μ → lower leverage).",
    )

    st.divider()
    force_refresh = st.button("🔄 Force Refresh (recompute heavy signals)",
                              help="Clears on-disk signal caches and re-runs walk-forward "
                                   "backtests for all 12 assets. Takes several minutes.")
    if st.button("Clear signal cache only (don't recompute now)"):
        deleted = portfolio.clear_signal_cache()
        st.success(f"Removed {len(deleted)} cache files.")
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — compute (or load) per-asset signals (heavy, cached on disk)
# ─────────────────────────────────────────────────────────────────────────────

cache_path = portfolio._signal_cache_path(universe, oos_start, oos_end, tc_mode=tc_mode)
cache_exists = os.path.exists(cache_path)

if force_refresh:
    portfolio.clear_signal_cache(universe)
    cache_exists = False

st.subheader("1. Per-asset regime signals")
mode_label = "Gross (TC=0, paper-faithful)" if tc_mode == "gross" else "Net (5 bps)"
if cache_exists:
    st.success(f"Loaded cached **{mode_label}** signals from `cache/{os.path.basename(cache_path)}`")
else:
    st.warning(
        f"No cached **{mode_label}** signals for **{universe} {oos_start}→{oos_end}**. "
        f"Click below to compute (this will run walk-forward on each of 12 assets — "
        f"several minutes total). Net and gross caches are stored separately."
    )
    if not st.button("▶️ Compute signals now"):
        st.stop()


@st.cache_resource(show_spinner=False)
def _load_signals(universe, oos_start, oos_end, tc_mode, force_refresh_flag):
    progress_bar = st.progress(0.0)
    status_box   = st.empty()

    def _cb(i, n, name):
        progress_bar.progress(min(1.0, (i + 1) / n))
        status_box.write(f"Asset {i+1}/{n}: **{name}**")

    signals = portfolio.compute_asset_signals(universe=universe,
                                              oos_start=oos_start, oos_end=oos_end,
                                              tc_mode=tc_mode,
                                              force_refresh=force_refresh_flag,
                                              progress_callback=_cb)
    progress_bar.empty(); status_box.empty()
    return signals


with st.spinner(f"Computing per-asset signals for **{universe}** ({mode_label})…"):
    signals = _load_signals(universe, oos_start, oos_end, tc_mode, force_refresh)

if not signals:
    st.error("Signal computation produced no output. Check the data files and logs.")
    st.stop()

st.write(f"✓ Computed signals for **{len(signals)} / 12** assets.")
with st.expander("Signal summary per asset", expanded=False):
    summary_rows = []
    for name, df in signals.items():
        oos = df[(df.index >= oos_start) & (df.index <= oos_end)]
        bear_pct = float((oos['Forecast_State'] == 1).mean()) if 'Forecast_State' in oos.columns else np.nan
        shifts = int(oos['Forecast_State'].diff().abs().fillna(0).sum()) if 'Forecast_State' in oos.columns else 0
        lh = oos.attrs.get('lambda_history', df.attrs.get('lambda_history', []))
        summary_rows.append({
            'Asset':      name,
            'Days':       len(oos),
            '% Bear':     f"{bear_pct*100:.1f}%",
            'Shifts':     shifts,
            'λ̄':          f"{np.mean(lh):.1f}" if len(lh) else "—",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — assemble panel + run all 7 portfolios
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("2. Portfolio backtests")

@st.cache_resource(show_spinner=False)
def _load_insample_mu(universe, oos_start, oos_end, tc_mode, force_refresh_flag):
    progress_bar = st.progress(0.0)
    status_box   = st.empty()

    def _cb(i, n, name):
        progress_bar.progress(min(1.0, (i + 1) / n))
        status_box.write(f"In-sample μ {i+1}/{n}: **{name}**")

    try:
        out = portfolio.compute_insample_regime_means(
            universe=universe, oos_start=oos_start, oos_end=oos_end,
            tc_mode=tc_mode,
            force_refresh=force_refresh_flag, progress_callback=_cb)
    except FileNotFoundError:
        out = {}
    progress_bar.empty(); status_box.empty()
    return out


insample_mu = None
if use_insample_mu:
    insample_mu_path = portfolio._insample_mu_cache_path(universe, oos_start, oos_end,
                                                        tc_mode=tc_mode)
    if not os.path.exists(insample_mu_path) and not force_refresh:
        st.info(
            f"In-sample regime-mean cache not found for **{mode_label}** — computing now "
            "(refits JM at each biannual anchor × 12 assets, ~2 min). "
            "This is the paper's exact MV(JM-XGB) μ spec."
        )
    with st.spinner("Loading in-sample regime means…"):
        insample_mu = _load_insample_mu(universe, oos_start, oos_end, tc_mode, force_refresh)
    if insample_mu:
        st.caption(f"✓ In-sample μ loaded for {len(insample_mu)} assets "
                   f"(`cache/{os.path.basename(insample_mu_path)}`).")

panel = portfolio.build_asset_panel(signals, oos_start, oos_end, insample_mu=insample_mu)
st.write(f"Panel: **{len(panel.returns)} trading days × {len(panel.asset_order)} assets** "
         f"({panel.returns.index.min().date()} → {panel.returns.index.max().date()})")


@st.cache_resource(show_spinner=False)
def _run_all_portfolios(_panel, rebal_freq, gamma_risk_minvar, gamma_risk_mv_b,
                        gamma_risk_mv_j, gamma_trade, tc_oneway, w_ub,
                        cov_hl_days, mu_baseline_hl_yr, mu_jmxgb_lb_yr,
                        cache_buster):
    """Cache_buster is the signal cache file's mtime — invalidates results when signals change."""
    progress_bar = st.progress(0.0)
    status_box   = st.empty()

    def _cb(i, n, label):
        progress_bar.progress(min(1.0, (i + 1) / n))
        status_box.write(f"Portfolio {i+1}/{n}: **{label}**")

    out = portfolio.run_all_portfolios(
        _panel,
        rebal_freq=rebal_freq,
        gamma_risk_minvar=gamma_risk_minvar,
        gamma_risk_mv_baseline=gamma_risk_mv_b,
        gamma_risk_mv_jmxgb=gamma_risk_mv_j,
        gamma_trade=gamma_trade,
        tc_oneway=tc_oneway,
        w_ub=w_ub,
        cov_hl_days=int(cov_hl_days),
        mu_baseline_hl_years=float(mu_baseline_hl_yr),
        mu_jmxgb_lookback_years=float(mu_jmxgb_lb_yr),
        progress_callback=_cb,
    )
    progress_bar.empty(); status_box.empty()
    return out


cache_buster = (os.path.getmtime(cache_path) if os.path.exists(cache_path) else 0,
                bool(insample_mu))
with st.spinner("Solving MVO portfolios…"):
    results = _run_all_portfolios(panel, rebal_freq,
                                  gamma_risk_minvar, gamma_risk_mv_b, gamma_risk_mv_j,
                                  gamma_trade, tc_oneway, w_ub,
                                  cov_hl_days, mu_baseline_hl_yr, mu_jmxgb_lb_yr,
                                  cache_buster)

# ─────────────────────────────────────────────────────────────────────────────
# Table 6 — performance metrics
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("3. Table 6 — Portfolio performance vs paper")

with st.expander("ℹ️ Why ~3% absolute-return gap remains — and what we fixed", expanded=False):
    st.markdown("""
### Paper-faithful spec is now implemented

The MVO follows paper Section 4.1 exactly:
- **μ is the forecast of excess returns** (not total — Section 4.1 last paragraph)
- **MV(JM-XGB) μ from in-sample JM regime means**: refit JM at each biannual
  anchor on the 11-year window with the selected λ; mean excess return per
  in-sample state. For LargeCap: bull μ = +29.2%/yr, bear μ = −46.1%/yr (capped
  at −10 bps/day per Section 4.5).
- **MinVar(JM-XGB) μ**: 10 bps for bullish, 0 for bearish (Section 4.2)
- **MV baseline μ**: EWM (hl=5y) of excess returns (Section 4.3)
- **L1 trade penalty** in the convex objective, solved with scipy-SLSQP (and
  cvxpy/CLARABEL as a cross-check). Mathematically equivalent to paper's Gurobi.
- **No active-mask restriction** — bearish-forecast assets have μ ≤ 0 and the
  optimizer naturally drives their weights to 0.
- **≤3-bullish → 100% cash** fallback rule (Section 4.5)

### What's left and why

The remaining gap (~3% absolute return for MV(JM-XGB), ~0.3 Sharpe) is
**structural regime-forecast instability**, not a portfolio-layer issue:

| Asset    | Our %bear | Paper Fig 2 | Our shifts | Paper |
|----------|-----------|-------------|------------|-------|
| LargeCap | 27.5%     | 20.9%       | 64         | 46    |
| REIT     | **42.2%** | **18.4%**   | 75         | 46    |
| AggBond  | 42.9%     | 41.5%       | 74         | 97    |

Our REIT alone flags 24 percentage points more bear days than the paper.
EM/EAFE/Gold/Treasury show similar over-bearishness. This drives the
**≤3-bullish cash trigger to fire on 18.9% of days** (paper presumably ~14%),
which mechanically caps average leverage at ~0.7 vs paper's 0.86.

Per-asset 0/1 Sharpe **matches paper closely** (avg gap −0.023 across 12
assets, per `MEMORY.md`) — the JM-XGB pipeline is correctly catching bad days.
It's just more aggressive at calling bear regimes for several assets, so the
*joint* distribution differs from paper's. This is the same gap documented for
Table 4 / Figure 2 in `MEMORY.md` — a per-asset λ-grid + EWMA-halflife
calibration issue that lives below the portfolio layer.

### Possible deeper fixes (out of paper-faithful scope)
- Tighter λ grid bounds for hl=0 assets (REIT, EM, EAFE) to reduce switching
- Longer EWMA halflife on probability smoothing (deviates from PAPER_EWMA_HL)
- Soften the cash trigger from ≤3 to ≤2 (deviates from paper rule)
""")


rows = {}
for label in portfolio.STRATEGY_LABELS:
    if label not in results:
        continue
    res = results[label]
    rows[label] = portfolio.portfolio_metrics(res['returns'], panel.rf_daily,
                                              res['weights'], res['turnover_daily'])

ours = pd.DataFrame(rows)
paper = portfolio.PAPER_TABLE_6.reindex(columns=ours.columns)


def _fmt(metric, val):
    if pd.isna(val):
        return "—"
    if metric in ('Return', 'Volatility', 'MDD'):
        return f"{val*100:.1f}%"
    if metric in ('Sharpe', 'Calmar', 'Leverage'):
        return f"{val:.2f}"
    if metric == 'Turnover':
        return f"{val:.2f}"
    return str(val)


display = pd.DataFrame(index=ours.index, columns=ours.columns, dtype=object)
for m in ours.index:
    for c in ours.columns:
        ours_v  = ours.loc[m, c]
        paper_v = paper.loc[m, c] if (m in paper.index and c in paper.columns) else np.nan
        display.loc[m, c] = f"{_fmt(m, ours_v)}  ({_fmt(m, paper_v)})"

st.markdown("**Format: ours (paper)**. Annualized excess return (rf≈1.1%/yr in 2007–2023).")
st.dataframe(display, use_container_width=True)

with st.expander("Raw numeric tables (ours vs paper)", expanded=False):
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Ours**")
        st.dataframe(ours.style.format("{:.4f}"), use_container_width=True)
    with cc2:
        st.markdown("**Paper Table 6**")
        st.dataframe(paper.style.format("{:.4f}"), use_container_width=True)

# Sharpe gap summary
sharpe_ours  = ours.loc['Sharpe']
sharpe_paper = paper.loc['Sharpe']
deltas = (sharpe_ours - sharpe_paper).round(3)
gap_summary = pd.DataFrame({'Sharpe (ours)': sharpe_ours.round(2),
                            'Sharpe (paper)': sharpe_paper.round(2),
                            'Δ': deltas}).T
st.write("**Sharpe gap vs paper**")
st.dataframe(gap_summary, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — cumulative wealth
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("4. Figure 3 — Cumulative wealth")

fig = go.Figure()
for label in portfolio.STRATEGY_LABELS:
    if label not in results:
        continue
    r = results[label]['returns']
    wealth = (1 + r).cumprod()
    line_style = dict(width=2.5)
    if 'JM-XGB' in label:
        line_style['width'] = 2.5
    elif label in ('60/40',):
        line_style['dash'] = 'dash'
    else:
        line_style['dash'] = 'dot'
    fig.add_trace(go.Scatter(x=wealth.index, y=wealth.values, name=label, line=line_style))

fig.update_layout(title=f"Cumulative wealth (1 = ${oos_start}), rebalanced {rebal_freq}",
                  xaxis_title="Date", yaxis_title="Wealth multiplier (log scale)",
                  yaxis_type="log", height=520, hovermode="x unified",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Table 7 — Forecast correlation
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("5. Table 7 — Return forecast correlation with realized")

ewma_mu  = results['MV']['mu_forecast']         if 'MV' in results else pd.DataFrame()
jmxgb_mu = results['MV(JM-XGB)']['mu_forecast'] if 'MV(JM-XGB)' in results else pd.DataFrame()

tbl7_ewma  = portfolio.forecast_correlation(panel, ewma_mu)
tbl7_jmxgb = portfolio.forecast_correlation(panel, jmxgb_mu)

idx_order = ['Overall'] + panel.asset_order
tbl7_ours = pd.DataFrame({
    'EWMA':   tbl7_ewma.reindex(idx_order),
    'JM-XGB': tbl7_jmxgb.reindex(idx_order),
})
tbl7_paper = portfolio.PAPER_TABLE_7.reindex(idx_order)

display7 = pd.DataFrame(index=idx_order, columns=['EWMA (ours)', 'EWMA (paper)',
                                                  'JM-XGB (ours)', 'JM-XGB (paper)'],
                        dtype=object)
for r in idx_order:
    for col, src in [('EWMA (ours)', tbl7_ours.loc[r, 'EWMA']),
                     ('EWMA (paper)', tbl7_paper.loc[r, 'EWMA']),
                     ('JM-XGB (ours)', tbl7_ours.loc[r, 'JM-XGB']),
                     ('JM-XGB (paper)', tbl7_paper.loc[r, 'JM-XGB'])]:
        display7.loc[r, col] = "—" if pd.isna(src) else f"{src*100:+.2f}%"

st.markdown("Pearson correlation between forecasted return at each rebalance and realized "
            "return over the next ~21 trading days. Paper baseline = EWM (hl=5y); "
            "JM-XGB = regime-conditional historical mean.")
st.dataframe(display7, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("Diagnostics — weights, turnover, regime calls", expanded=False):
    sel = st.selectbox("Strategy", options=list(results.keys()),
                       index=min(2, len(results) - 1))
    res = results[sel]
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Average weights (over OOS period)**")
        avg_w = res['weights'].mean().sort_values(ascending=False)
        st.dataframe(avg_w.to_frame(name='avg weight').style.format("{:.3f}"),
                     use_container_width=True)
    with col2:
        st.markdown("**Turnover over time (annualized rolling)**")
        roll_to = res['turnover_daily'].rolling(252).sum()
        st.line_chart(roll_to)

    st.markdown("**Daily weights (last 250 days)**")
    st.dataframe(res['weights'].tail(250).style.format("{:.3f}"),
                 use_container_width=True)

    st.markdown("**Bullish-asset count (JM-XGB strategies)**")
    bull_count = (panel.forecast == 0).sum(axis=1)
    st.line_chart(bull_count)


with st.expander("Regime call panel — bear-state percentage per asset", expanded=False):
    bear_pct = pd.DataFrame({
        a: [(panel.forecast[a] == 1).mean() * 100] for a in panel.asset_order
    }, index=['% bear days']).T
    st.bar_chart(bear_pct)


st.caption(
    f"Cache file: `cache/{os.path.basename(cache_path)}` — "
    f"size ≈ {os.path.getsize(cache_path)/1024:.0f} KB. "
    "Click **Force Refresh** in the sidebar to recompute."
)
