import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import glob
import datetime
import importlib

import main as backend
importlib.reload(backend)

st.set_page_config(layout="wide", page_title="Data Quality Audit")

st.title("Data Quality Audit")
st.caption("Go/no-go checks before trusting backtest results. Reads from existing caches — does not re-run the pipeline.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

# Mutual fund → ETF proxy pairs (from asset_lists.md)
PROXY_PAIRS = {
    "VIMSX": {"etf": "IJH",  "name": "Mid-Cap 400"},
    "NAESX": {"etf": "IWM",  "name": "Small-Cap 2000"},
    "FDIVX": {"etf": "EFA",  "name": "MSCI EAFE"},
    "VEIEX": {"etf": "EEM",  "name": "Emerging Markets"},
    "VBMFX": {"etf": "AGG",  "name": "Aggregate Bond"},
    "VUSTX": {"etf": "SPTL", "name": "Long-Term Treasury"},
    "VWEHX": {"etf": "HYG",  "name": "High-Yield Corp"},
    "VWESX": {"etf": "SPBO", "name": "Corporate Bond"},
    "FRESX": {"etf": "IYR",  "name": "Real Estate"},
    "PCASX": {"etf": "DBC",  "name": "Commodity"},
    "GC=F":  {"etf": "GLD",  "name": "Gold"},
}


def _load_data_cache(ticker: str):
    """Load the data cache for a given ticker, trying multiple naming conventions."""
    safe = ticker.replace("^", "").replace("=", "")
    candidates = sorted(glob.glob(os.path.join(CACHE_DIR, f"data_cache_{safe}*.pkl")))
    if not candidates and ticker == "^SP500TR":
        legacy = os.path.join(CACHE_DIR, "data_cache.pkl")
        if os.path.exists(legacy):
            candidates = [legacy]
    if not candidates:
        return None
    return pd.read_pickle(candidates[-1])  # newest


def _cache_file_mtime(ticker: str) -> float | None:
    """Return mtime of the cache file for a ticker, or None."""
    safe = ticker.replace("^", "").replace("=", "")
    candidates = sorted(glob.glob(os.path.join(CACHE_DIR, f"data_cache_{safe}*.pkl")))
    if not candidates and ticker == "^SP500TR":
        legacy = os.path.join(CACHE_DIR, "data_cache.pkl")
        if os.path.exists(legacy):
            candidates = [legacy]
    if not candidates:
        return None
    return os.path.getmtime(candidates[-1])


# ---------------------------------------------------------------------------
# Discover available cached tickers
# ---------------------------------------------------------------------------
available_tickers = []
for f in sorted(glob.glob(os.path.join(CACHE_DIR, "data_cache_*.pkl"))):
    base = os.path.basename(f)
    parts = base.replace("data_cache_", "").replace(".pkl", "").split("_")
    if parts:
        tkr = parts[0]
        if tkr == "SP500TR":
            tkr = "^SP500TR"
        elif tkr == "IRX":
            tkr = "^IRX"
        elif tkr == "VIX":
            tkr = "^VIX"
        elif tkr.startswith("GCF"):
            tkr = "GC=F"
        if tkr not in available_tickers:
            available_tickers.append(tkr)

if not available_tickers:
    if os.path.exists(os.path.join(CACHE_DIR, "data_cache.pkl")):
        available_tickers.append("^SP500TR")

if not available_tickers:
    st.error("No data caches found in `cache/`. Run the backtest first to populate caches.")
    st.stop()

# ---------------------------------------------------------------------------
# Ticker selector
# ---------------------------------------------------------------------------
selected_ticker = st.selectbox("Ticker", available_tickers, index=0)
df = _load_data_cache(selected_ticker)
if df is None:
    st.error(f"Could not load data cache for {selected_ticker}")
    st.stop()

# =========================================================================
# CACHE FRESHNESS TABLE
# =========================================================================
now = datetime.datetime.now()


def _fmt_mtime(mtime):
    """Format mtime as age string + date, with status icon."""
    if mtime is None:
        return "—", "Not found", ""
    dt = datetime.datetime.fromtimestamp(mtime)
    age_days = (now - dt).days
    age_str = f"{age_days}d ago" if age_days > 0 else "today"
    status = "🟢" if age_days <= 1 else ("🟡" if age_days <= 7 else "🔴")
    return status, age_str, dt.strftime("%Y-%m-%d %H:%M")


def _cache_stats(ticker: str):
    """Return (start_date, end_date, rows, nan_pct) for a ticker's cache, or Nones."""
    cache_df = _load_data_cache(ticker)
    if cache_df is None or cache_df.empty:
        return "—", "—", "—", "—"
    start = cache_df.index.min().strftime("%Y-%m-%d")
    end = cache_df.index.max().strftime("%Y-%m-%d")
    rows = len(cache_df)
    total_cells = cache_df.size
    nan_cells = int(cache_df.isna().sum().sum())
    nan_pct = f"{nan_cells / total_cells * 100:.1f}%" if total_cells > 0 else "—"
    return start, end, f"{rows:,}", nan_pct


def _fred_stats():
    """Return (start_date, end_date, rows, nan_pct) for FRED cache."""
    path = os.path.join(CACHE_DIR, "fred_cache.pkl")
    if not os.path.exists(path):
        return "—", "—", "—", "—"
    fdf = pd.read_pickle(path)
    if fdf.empty:
        return "—", "—", "—", "—"
    start = fdf.index.min().strftime("%Y-%m-%d")
    end = fdf.index.max().strftime("%Y-%m-%d")
    rows = len(fdf)
    total_cells = fdf.size
    nan_cells = int(fdf.isna().sum().sum())
    nan_pct = f"{nan_cells / total_cells * 100:.1f}%" if total_cells > 0 else "—"
    return start, end, f"{rows:,}", nan_pct


# Build rows: selected ticker cache + all supporting series
_ticker_list = [
    (selected_ticker, "Target Asset", _cache_file_mtime(selected_ticker)),
    (backend.BOND_TICKER, "Bond (Stock-Bond Corr)", _cache_file_mtime(backend.BOND_TICKER)),
    (backend.RISK_FREE_TICKER, "Risk-Free Rate", _cache_file_mtime(backend.RISK_FREE_TICKER)),
    (backend.VIX_TICKER, "VIX", _cache_file_mtime(backend.VIX_TICKER)),
]

table_rows = []
short_lookback_warnings = []
oos_start = pd.to_datetime(backend.OOS_START_DATE)

for ticker, role, mtime in _ticker_list:
    status, age, date_str = _fmt_mtime(mtime)
    start, end, rows, nan_pct = _cache_stats(ticker)
    table_rows.append({"Ticker": ticker, "Series": role, "Status": status,
                       "Age": age, "Fetched": date_str,
                       "Start": start, "End": end, "Rows": rows, "NaN%": nan_pct})
    # Check lookback for target asset
    if start != "—":
        years_before_oos = (oos_start - pd.to_datetime(start)).days / 365.25
        if years_before_oos < 11:
            short_lookback_warnings.append(
                f"**{ticker}** ({role}): only {years_before_oos:.1f}y before OOS start "
                f"({backend.OOS_START_DATE}) — need 11y for full JM lookback")

# FRED row
fred_mtime = (os.path.getmtime(os.path.join(CACHE_DIR, "fred_cache.pkl"))
              if os.path.exists(os.path.join(CACHE_DIR, "fred_cache.pkl")) else None)
fred_status, fred_age, fred_date = _fmt_mtime(fred_mtime)
fred_start, fred_end, fred_rows, fred_nan = _fred_stats()
table_rows.append({"Ticker": "DGS2 / DGS10", "Series": "FRED Yields", "Status": fred_status,
                   "Age": fred_age, "Fetched": fred_date,
                   "Start": fred_start, "End": fred_end, "Rows": fred_rows, "NaN%": fred_nan})

# Backtest cache row (no date range / NaN — it's not a time series)
bt_mtime = (os.path.getmtime(os.path.join(CACHE_DIR, "backtest_cache.pkl"))
            if os.path.exists(os.path.join(CACHE_DIR, "backtest_cache.pkl")) else None)
bt_status, bt_age, bt_date = _fmt_mtime(bt_mtime)
table_rows.append({"Ticker": "—", "Series": "Backtest Cache", "Status": bt_status,
                   "Age": bt_age, "Fetched": bt_date,
                   "Start": "—", "End": "—", "Rows": "—", "NaN%": "—"})

st.dataframe(pd.DataFrame(table_rows).set_index("Ticker"),
             use_container_width=True, hide_index=False)

if short_lookback_warnings:
    for w in short_lookback_warnings:
        st.warning(w)


# =========================================================================
# SECTION 1 — Raw Data Health
# =========================================================================
st.header("Section 1 — Raw Data Health",
          help="Stale streaks: longest run of |return| < 0.001% (stale NAVs corrupt EWMA features). "
               "Outliers: 0.1st/99.9th percentile and days > ±10% (catches Yahoo adjustment artifacts). "
               "Coverage: per-year completeness (JM needs 11-year lookback).")

returns = df["Target_Return"] if "Target_Return" in df.columns else df.iloc[:, 0].pct_change()

# --- Compute stale streak ---
is_stale = returns.abs() < 0.00001
stale_groups = (is_stale != is_stale.shift()).cumsum()
stale_streaks = is_stale.groupby(stale_groups).agg(["sum", "first"])
stale_only = stale_streaks[stale_streaks["first"] == True]
longest_streak = int(stale_only["sum"].max()) if not stale_only.empty else 0
if longest_streak > 0:
    longest_group = stale_only["sum"].idxmax()
    streak_dates = is_stale[stale_groups == longest_group].index
    streak_period = f"{streak_dates[0].strftime('%Y-%m-%d')} → {streak_dates[-1].strftime('%Y-%m-%d')}"
else:
    streak_period = "—"
total_stale = int(is_stale.sum())
is_index = selected_ticker.startswith("^")
stale_thresh = 1 if is_index else 3
stale_status = "🔴" if longest_streak > stale_thresh else ("🟡" if longest_streak > 1 else "🟢")

# --- Compute outliers ---
p01 = returns.quantile(0.001)
p999 = returns.quantile(0.999)
extreme_days = int((returns.abs() > 0.10).sum())
is_bond = selected_ticker in backend.DD_EXCLUDE_TICKERS
extreme_thresh = 0.05 if is_bond else 0.15
flag_extreme = abs(p01) > extreme_thresh or abs(p999) > extreme_thresh
outlier_status = "🔴" if flag_extreme else "🟢"

# --- Build summary table ---
health_rows = [
    {"Check": "Stale Return Streak",
     "Value": f"{longest_streak}d (total: {total_stale}/{len(returns)}, {100*total_stale/len(returns):.2f}%)",
     "Detail": streak_period,
     "Status": stale_status},
    {"Check": "Return Outliers",
     "Value": f"P0.1={p01*100:+.2f}%  P99.9={p999*100:+.2f}%",
     "Detail": f"{extreme_days} days > ±10% (thresh ±{extreme_thresh*100:.0f}%)",
     "Status": outlier_status},
]

# --- Coverage check ---
oos_start_yr = int(backend.OOS_START_DATE[:4])
data_start_yr = df.index.year.min()
lookback_available = oos_start_yr - data_start_yr
coverage_status = "🟢" if lookback_available >= 11 else "🔴"
health_rows.append({
    "Check": "Lookback Coverage",
    "Value": f"{lookback_available}y before OOS ({backend.OOS_START_DATE})",
    "Detail": f"Need 11y; have {df.index.year.min()}–{df.index.year.max()}",
    "Status": coverage_status,
})

# --- Gap years (mid-series missing years) ---
df_yearly = df.groupby(df.index.year).size()
all_years = range(df.index.year.min(), df.index.year.max() + 1)
gap_years = [yr for yr in all_years if df_yearly.get(yr, 0) == 0]
partial_years = [yr for yr in all_years if 0 < df_yearly.get(yr, 0) < 200]
gap_status = "🔴" if gap_years else ("🟡" if partial_years else "🟢")
gap_detail = ""
if gap_years:
    gap_detail += f"Missing: {', '.join(str(y) for y in gap_years)}. "
if partial_years:
    gap_detail += f"Partial (<200d): {', '.join(str(y) for y in partial_years)}"
health_rows.append({
    "Check": "Year Gaps",
    "Value": f"{len(gap_years)} missing, {len(partial_years)} partial",
    "Detail": gap_detail or "All years complete",
    "Status": gap_status,
})

st.dataframe(pd.DataFrame(health_rows).set_index("Check"), use_container_width=True)

# Compact coverage bar (keep it small)
fig3, ax3 = plt.subplots(figsize=(14, 1.0))
colors_cov = []
counts_cov = []
for yr in all_years:
    n = df_yearly.get(yr, 0)
    counts_cov.append(n)
    colors_cov.append("#2ecc71" if n >= 200 else ("#f39c12" if n > 0 else "#e74c3c"))

ax3.barh(0, counts_cov, left=[0] + list(np.cumsum(counts_cov[:-1])),
         color=colors_cov, height=0.5, edgecolor="none")
cum = 0
for i, yr in enumerate(all_years):
    if counts_cov[i] > 0 and i % 4 == 0:
        ax3.text(cum + counts_cov[i] / 2, 0, str(yr), ha="center", va="center",
                 fontsize=6, color="white", fontweight="bold")
    cum += counts_cov[i]
ax3.set_yticks([])
ax3.set_xlabel("Trading Days", fontsize=8)
ax3.spines[["top", "right", "left"]].set_visible(False)
ax3.tick_params(axis="x", labelsize=7)
fig3.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.35)
st.pyplot(fig3)
plt.close(fig3)

# --- Indicator 3b: FRED Yield Data Health ---
st.subheader("Indicator 3b — FRED Yield Data Health",
             help="FRED Treasury yields (DGS2, DGS10) feed three macro features: "
                  "Yield_2Y_EWMA_diff, Yield_Slope_EWMA_10, and Yield_Slope_EWMA_diff_21. "
                  "These are fetched separately from Yahoo data and merged by date.\n\n"
                  "**What can go wrong:** FRED API timeouts cause missing data. "
                  "Yields are only published on business days, so weekends/holidays are expected NaNs "
                  "(forward-filled during merge). But mid-week NaN gaps or stale values indicate a "
                  "download failure. Yield_2Y_diff near zero for extended periods means DGS2 didn't update.\n\n"
                  "**Expected:** Yield_2Y_diff should be nonzero on most trading days (typical magnitude 0.01–0.05). "
                  "Yield_Slope_EWMA_10 ranges roughly -1.0 to +3.0 (negative = inverted curve). "
                  "No large NaN gaps in the derived features.")

fred_cache_path = os.path.join(CACHE_DIR, "fred_cache.pkl")
has_raw_fred = os.path.exists(fred_cache_path)
if has_raw_fred:
    fred_df = pd.read_pickle(fred_cache_path)
else:
    fred_df = None

yield_derived = ["Yield_2Y_EWMA_diff", "Yield_Slope_EWMA_10", "Yield_Slope_EWMA_diff_21"]
yield_raw_diff = "Yield_2Y_diff"
has_derived = all(c in df.columns for c in yield_derived)

if has_raw_fred:
    st.markdown("**Raw FRED Series (from `fred_cache.pkl`)**")

    raw_col1, raw_col2 = st.columns(2)
    for i, series_id in enumerate(["DGS2", "DGS10"]):
        col = raw_col1 if i == 0 else raw_col2
        if series_id not in fred_df.columns:
            col.warning(f"{series_id} not found in FRED cache.")
            continue
        s = fred_df[series_id]
        n_total = len(s)
        n_nan = int(s.isna().sum())
        n_valid = n_total - n_nan

        is_stale_yield = (s.diff().abs() < 1e-10) & s.notna() & s.shift(1).notna()
        stale_groups_y = (is_stale_yield != is_stale_yield.shift()).cumsum()
        stale_streaks_y = is_stale_yield.groupby(stale_groups_y).agg(["sum", "first"])
        stale_only_y = stale_streaks_y[stale_streaks_y["first"] == True]
        longest_stale_y = int(stale_only_y["sum"].max()) if not stale_only_y.empty else 0

        col.markdown(f"**{series_id}**")
        c1, c2, c3 = col.columns(3)
        c1.metric("Valid / Total", f"{n_valid} / {n_total}")
        c2.metric("NaN Days", f"{n_nan} ({100*n_nan/n_total:.1f}%)")
        stale_status = "🟡" if longest_stale_y > 5 else "🟢"
        c3.metric("Longest Stale", f"{longest_stale_y}d {stale_status}",
                  help="Consecutive days with identical yield values. "
                       "1–3 days is normal (weekends/holidays after ffill). > 5 days is suspicious.")

    fig_fred, (ax_f1, ax_f2) = plt.subplots(2, 1, figsize=(14, 3.5), sharex=True)
    for ax_f, sid, color in [(ax_f1, "DGS2", "#e67e22"), (ax_f2, "DGS10", "#2980b9")]:
        if sid in fred_df.columns:
            s = fred_df[sid].dropna()
            ax_f.plot(s.index, s.values, linewidth=0.5, color=color)
            ax_f.set_ylabel(f"{sid} (%)")
            ax_f.spines[["top", "right"]].set_visible(False)
    ax_f1.set_title("Raw FRED Yields", fontsize=10)
    ax_f2.xaxis.set_major_locator(mdates.YearLocator(5))
    ax_f2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig_fred.tight_layout()
    st.pyplot(fig_fred)
    plt.close(fig_fred)

    st.caption("Visual sanity check: DGS2 and DGS10 should show smooth yield curves. "
               "Sudden spikes to 0 or jumps > 2% in a single day indicate download artifacts.")

if has_derived:
    if has_raw_fred:
        st.markdown("**Derived Yield Features (in data cache)**")
    else:
        st.markdown("**Derived Yield Features** (raw FRED cache not found — checking derived features only)")

    if yield_raw_diff in df.columns:
        y2d = df[yield_raw_diff]
        is_zero_diff = y2d.abs() < 1e-10
        zero_groups = (is_zero_diff != is_zero_diff.shift()).cumsum()
        zero_streaks = is_zero_diff.groupby(zero_groups).agg(["sum", "first"])
        zero_only = zero_streaks[zero_streaks["first"] == True]
        longest_zero = int(zero_only["sum"].max()) if not zero_only.empty else 0
        total_zero = int(is_zero_diff.sum())
    else:
        longest_zero = 0
        total_zero = 0

    nan_counts = {feat: int(df[feat].isna().sum()) for feat in yield_derived}

    fc1, fc2, fc3 = st.columns(3)
    zero_status = "🔴" if longest_zero > 10 else ("🟡" if longest_zero > 5 else "🟢")
    fc1.metric("Yield_2Y_diff Zero Streak", f"{longest_zero}d {zero_status}",
               help="Longest run of consecutive days where DGS2 yield didn't change. "
                    "Weekends/holidays account for 1–3d. > 5d means FRED data was stale during that period.")
    fc2.metric("Total Zero-Diff Days", f"{total_zero} / {len(df)} ({100*total_zero/len(df):.1f}%)",
               help="Days where the 2Y yield was unchanged. ~30–40% is typical "
                    "(weekends, holidays, minimal yield movement days). > 50% warrants investigation.")
    total_nans = sum(nan_counts.values())
    nan_status = "🔴" if total_nans > 0 else "🟢"
    fc3.metric("NaN in Derived Features", f"{total_nans} {nan_status}",
               help="Should be 0 after feature engineering (NaNs are forward-filled during merge). "
                    "Any NaN here means the FRED-Yahoo date merge left gaps.")

    range_data = []
    expected_ranges = {
        "Yield_2Y_EWMA_diff": (-0.15, 0.15),
        "Yield_Slope_EWMA_10": (-1.5, 3.5),
        "Yield_Slope_EWMA_diff_21": (-0.10, 0.10),
    }
    for feat in yield_derived:
        s = df[feat].dropna()
        lo, hi = s.min(), s.max()
        exp_lo, exp_hi = expected_ranges.get(feat, (-999, 999))
        in_range = lo >= exp_lo * 1.5 and hi <= exp_hi * 1.5
        range_data.append({
            "Feature": feat,
            "Min": f"{lo:.4f}",
            "Max": f"{hi:.4f}",
            "Expected Range": f"[{exp_lo}, {exp_hi}]",
            "Status": "🟢" if in_range else "🟡",
        })
    st.dataframe(pd.DataFrame(range_data).set_index("Feature"), use_container_width=True)

elif not has_raw_fred:
    st.info("No FRED cache (`fred_cache.pkl`) and no derived yield features found in data cache. "
            "Run the backtest to fetch FRED data and populate caches.")


# =========================================================================
# SECTION 2 — Feature Health
# =========================================================================
st.header("Section 2 — Feature Health",
          help="Catches feature engineering artifacts that corrupt JM and XGB inputs. "
               "The JM uses z-scored return features — outliers directly damage cluster stability. "
               "Macro features (yield-derived, VIX, Stock-Bond Corr) feed XGB raw — distributional "
               "anomalies there cause noisy or biased probability forecasts. "
               "Sortino clipping and Stock-Bond Correlation get their own dedicated checks.")

# --- Indicator 4: Missing Data Summary ---
st.subheader("Indicator 4 — Missing Data Summary",
             help="Shows the exact NaN percentage for every feature and price series in the cache. "
                  "Forward-filling can mask large data gaps — this indicator exposes the true picture.\n\n"
                  "**Expected:** 0% NaN for price and return columns after feature engineering. "
                  "Early rows may have NaN due to rolling window warm-up (e.g., first 252 rows for "
                  "Stock_Bond_Corr). Features with >1% NaN outside the warm-up period warrant investigation.\n\n"
                  "**Reading the chart:** Bars show % NaN per column. Columns are grouped by type: "
                  "price/return, return features, macro features, and other.")

# Compute NaN percentages for all columns
nan_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
nan_pct = nan_pct[nan_pct > 0]  # only show columns with NaN

if nan_pct.empty:
    st.success(f"No NaN values in any of the {len(df.columns)} columns ({len(df)} rows).")
else:
    # Categorize columns
    def _categorize(col_name):
        if col_name in ("Target_Return", "Close", "Adj Close", "Open", "High", "Low", "Volume"):
            return "Price/Return"
        if col_name.startswith(("DD_", "Avg_Ret_", "Sortino_")):
            return "Return Feature"
        if col_name.startswith(("Yield_", "VIX_", "Stock_Bond")):
            return "Macro Feature"
        return "Other"

    nan_df = pd.DataFrame({
        "Column": nan_pct.index,
        "NaN %": nan_pct.values.round(2),
        "NaN Count": df[nan_pct.index].isna().sum().values,
        "Category": [_categorize(c) for c in nan_pct.index],
    })

    # Color-coded bar chart
    cat_colors = {
        "Price/Return": "#e74c3c",
        "Return Feature": "#3498db",
        "Macro Feature": "#e67e22",
        "Other": "#95a5a6",
    }

    fig_nan, ax_nan = plt.subplots(figsize=(14, max(2.5, 0.35 * len(nan_pct))))
    bar_colors = [cat_colors.get(_categorize(c), "#95a5a6") for c in nan_pct.index]
    y_pos = range(len(nan_pct))
    ax_nan.barh(y_pos, nan_pct.values, color=bar_colors, edgecolor="none", height=0.7)
    ax_nan.set_yticks(y_pos)
    ax_nan.set_yticklabels(nan_pct.index, fontsize=8)
    ax_nan.set_xlabel("NaN %")
    ax_nan.set_title(f"Missing Values by Column ({len(nan_pct)} columns with NaN)", fontsize=10)
    ax_nan.invert_yaxis()
    ax_nan.spines[["top", "right"]].set_visible(False)

    # Add percentage labels
    for i, (val, col_name) in enumerate(zip(nan_pct.values, nan_pct.index)):
        count = int(df[col_name].isna().sum())
        ax_nan.text(val + 0.3, i, f"{val:.1f}% ({count})", va="center", fontsize=7)

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=c, label=l) for l, c in cat_colors.items()
                      if any(_categorize(col) == l for col in nan_pct.index)]
    if legend_patches:
        ax_nan.legend(handles=legend_patches, fontsize=7, loc="lower right")

    fig_nan.tight_layout()
    st.pyplot(fig_nan)
    plt.close(fig_nan)

    # Flag any price/return NaN (should be zero after pipeline)
    price_nan = nan_df[nan_df["Category"] == "Price/Return"]
    if not price_nan.empty:
        st.warning(f"Price/Return columns with NaN: {', '.join(price_nan['Column'].tolist())}. "
                   "This should not happen after feature engineering — investigate the data source.")


# --- Indicator 5: Feature Z-Score Extremes ---
st.subheader("Indicator 5 — Feature Z-Score Extremes (rolling 252-day)",
             help="Each feature's value is z-scored against its own trailing 252-day (1-year) "
                  "rolling mean and std. Points beyond ±4σ are extreme outliers.\n\n"
                  "**Return features** (DD, Avg_Ret, Sortino) are z-scored before feeding to the JM — "
                  "outliers directly damage cluster stability because the alternating optimization "
                  "uses squared Euclidean distance.\n\n"
                  "**Macro features** (Yield_2Y_EWMA_diff, Yield_Slope_EWMA_10, Yield_Slope_EWMA_diff_21, "
                  "VIX_EWMA_log_diff) feed XGB raw — outliers there can cause spurious splits "
                  "and noisy probability forecasts.\n\n"
                  "**Expected:** Occasional ±4σ spikes during market crashes (GFC, COVID) or "
                  "rate shock episodes (2022 hiking cycle) are legitimate. "
                  "Persistent clusters of exceedances, or exceedances in calm markets, suggest data artifacts.\n\n"
                  "**Reading the charts:** Red-shaded regions show where features exceeded ±4σ. "
                  "The y-axis label shows exceedance count per feature. "
                  "Stock_Bond_Corr is excluded here — it has its own dedicated chart (Indicator 7).")

return_features = [c for c in df.columns if c.startswith(("DD_", "Avg_Ret_", "Sortino_"))]
macro_features = [c for c in ["Yield_2Y_EWMA_diff", "Yield_Slope_EWMA_10",
                               "Yield_Slope_EWMA_diff_21", "VIX_EWMA_log_diff"]
                  if c in df.columns]


def _plot_zscore_panel(feature_list, title, color):
    """Plot rolling z-score panel for a list of features. Returns exceedance Series."""
    zscore_df = pd.DataFrame(index=df.index)
    for feat in feature_list:
        rolling_mean = df[feat].rolling(252).mean()
        rolling_std = df[feat].rolling(252).std()
        zscore_df[feat] = (df[feat] - rolling_mean) / rolling_std.replace(0, np.nan)
    zscore_df = zscore_df.dropna()

    exceedance_counts = (zscore_df.abs() > 4).sum()
    total_obs = len(zscore_df)

    fig, axes = plt.subplots(len(feature_list), 1,
                             figsize=(14, 2.0 * len(feature_list)), sharex=True)
    if len(feature_list) == 1:
        axes = [axes]

    for i, feat in enumerate(feature_list):
        ax = axes[i]
        series = zscore_df[feat]
        ax.plot(series.index, series.values, linewidth=0.4, color=color, alpha=0.8)
        ax.axhline(4, color="#e74c3c", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axhline(-4, color="#e74c3c", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.fill_between(series.index, 4, series.values, where=series.values > 4,
                        color="#e74c3c", alpha=0.3)
        ax.fill_between(series.index, -4, series.values, where=series.values < -4,
                        color="#e74c3c", alpha=0.3)
        n_exc = int(exceedance_counts[feat])
        label = f"{feat} ({n_exc} exc.)" if n_exc > 0 else feat
        ax.set_ylabel(label, fontsize=8)
        ax.set_ylim(-8, 8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(5))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    return exceedance_counts, total_obs


all_exceedances = pd.Series(dtype=int)
all_total_obs = 0

if not return_features:
    st.info("No return features found in cached data.")
else:
    exc, nobs = _plot_zscore_panel(return_features,
                                   "Return Feature Rolling Z-Scores (252d window) — fed to JM + XGB",
                                   "#3498db")
    all_exceedances = pd.concat([all_exceedances, exc])
    all_total_obs = max(all_total_obs, nobs)

if macro_features:
    exc, nobs = _plot_zscore_panel(macro_features,
                                   "Macro Feature Rolling Z-Scores (252d window) — fed to XGB only",
                                   "#e67e22")
    all_exceedances = pd.concat([all_exceedances, exc])
    all_total_obs = max(all_total_obs, nobs)
elif not return_features:
    pass
else:
    st.info("No macro features (Yield, VIX) found in cached data.")

if not all_exceedances.empty:
    exc_table = all_exceedances[all_exceedances > 0]
    if not exc_table.empty:
        st.dataframe(
            pd.DataFrame({"Feature": exc_table.index, "±4σ Exceedances": exc_table.values,
                          "% of Obs": (exc_table.values / all_total_obs * 100).round(3)}).set_index("Feature"),
            use_container_width=True,
        )
    else:
        st.success("No features exceed ±4σ in the rolling window.")

# --- Indicator 6: Sortino Clipping Frequency ---
st.subheader("Indicator 6 — Sortino Clipping Frequency",
             help="Sortino ratios are clipped to [-10, 10] to prevent extreme values from dominating "
                  "JM fitting. This indicator shows how often observations hit the clip boundary.\n\n"
                  "**Expected:** < 1% clipping is normal — the boundary is generous. "
                  "1–2% is borderline. > 2% means the clip is materially reshaping the feature distribution, "
                  "which could bias JM cluster centers.\n\n"
                  "**Interpretation:** High clipping on Sortino_5 (short halflife) is more common "
                  "than Sortino_21 because short-window ratios are noisier. "
                  "High-side clipping (≥10) during low-vol bull markets is expected; "
                  "low-side clipping (≤-10) during crashes is expected.")

sortino_features = [c for c in df.columns if c.startswith("Sortino_")]
if not sortino_features:
    st.info("No Sortino features found.")
else:
    clip_data = []
    for feat in sortino_features:
        s = df[feat]
        n_total = len(s.dropna())
        n_clipped_high = int((s >= 9.99).sum())
        n_clipped_low = int((s <= -9.99).sum())
        n_clipped = n_clipped_high + n_clipped_low
        pct = n_clipped / n_total * 100 if n_total > 0 else 0
        status = "🔴" if pct > 2.0 else ("🟡" if pct > 1.0 else "🟢")
        clip_data.append({
            "Feature": feat,
            "Clipped High (≥10)": n_clipped_high,
            "Clipped Low (≤-10)": n_clipped_low,
            "Total Clipped": n_clipped,
            "% of Obs": round(pct, 3),
            "Status": status,
        })

    clip_df = pd.DataFrame(clip_data).set_index("Feature")
    st.dataframe(clip_df, use_container_width=True)

    if any(row["% of Obs"] > 2.0 for row in clip_data):
        st.warning("Clipping exceeds 2% for one or more Sortino features — the clip boundary is materially shaping the distribution.")

# --- Indicator 7: Stock-Bond Correlation Rolling Series ---
st.subheader("Indicator 7 — Stock-Bond Correlation (252-day rolling)",
             help="Rolling 252-day correlation between LargeCap returns and AggBond returns. "
                  "This is a macro feature shared across ALL assets (paper Table 3 specifies "
                  "Stock-Bond Corr always uses LargeCap vs AggBond, regardless of target asset).\n\n"
                  "**Expected:** Negative correlation (-0.3 to -0.1) for most of 2000–2021, "
                  "flipping positive (~+0.3) around 2022 as rates rose — this is a well-known "
                  "macro regime shift and a legitimate signal, not an artifact.\n\n"
                  "**Red flags:** Week-over-week jumps > 0.3 (red dots) suggest stale VBMFX NAV pricing "
                  "or bond data gaps. A handful of alerts near market dislocations (Mar 2020) is OK; "
                  "persistent clusters are not.")

if "Stock_Bond_Corr" in df.columns:
    sbc = df["Stock_Bond_Corr"].dropna()

    fig6, ax6 = plt.subplots(figsize=(14, 3))
    ax6.plot(sbc.index, sbc.values, linewidth=0.7, color="#2c3e50")
    ax6.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    wow_change = sbc.diff(5)
    alert_dates = wow_change[wow_change.abs() > 0.3].index
    if len(alert_dates) > 0:
        ax6.scatter(alert_dates, sbc.loc[alert_dates], color="#e74c3c", s=15, zorder=5, label=f"WoW Δ > 0.3 ({len(alert_dates)} alerts)")
        ax6.legend(fontsize=8)

    ax6.set_ylabel("Correlation")
    ax6.set_title("Stock-Bond Correlation (LargeCap vs AggBond)", fontsize=10)
    ax6.xaxis.set_major_locator(mdates.YearLocator(3))
    ax6.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax6.spines[["top", "right"]].set_visible(False)
    fig6.tight_layout()
    st.pyplot(fig6)
    plt.close(fig6)

    if len(alert_dates) > 0:
        st.warning(f"{len(alert_dates)} dates with week-over-week correlation change > 0.3 detected. "
                   "Could indicate VBMFX NAV stale pricing artifacts.")
    else:
        st.success("No erratic week-over-week jumps (> 0.3) detected.")
else:
    st.info("Stock_Bond_Corr feature not found in cached data.")


# =========================================================================
# SECTION 3 — Proxy Reliability vs. Reference Index
# =========================================================================
st.header("Section 3 — Proxy Reliability vs. Reference Index",
          help="The paper chains mutual funds to ETFs (e.g., VIMSX as a proxy for IJH before 2000). "
               "If the mutual fund is a poor proxy, the model learns from corrupted pre-ETF data.\n\n"
               "This section pairs each mutual fund with its target ETF using the overlap period "
               "(dates where both exist) to measure proxy quality.\n\n"
               "**Metrics:**\n"
               "- **Daily Correlation:** Should be > 0.99 for a good proxy.\n"
               "- **Annualized Tracking Error:** Std of daily return differences × √252. Should be < 2% for equity, < 1% for bond.\n"
               "- **Total Return Drift:** Cumulative return difference over the overlap. Shows fee drag or tracking failure.\n\n"
               "If a mutual fund has high tracking error or low correlation, the pre-ETF data may be teaching the model wrong patterns.")

# Check which proxy pairs have both sides cached
proxy_results = []
proxy_pairs_available = []

for fund_ticker, info in PROXY_PAIRS.items():
    etf_ticker = info["etf"]
    fund_df = _load_data_cache(fund_ticker)
    etf_df = _load_data_cache(etf_ticker)
    if fund_df is not None and etf_df is not None:
        proxy_pairs_available.append((fund_ticker, etf_ticker, info["name"]))

if not proxy_pairs_available:
    st.info("No proxy pairs available — need both mutual fund and ETF caches. "
            "Run `benchmark_assets.py` for both 'Default ETFs' and 'Long History' asset lists to populate caches.")
else:
    st.caption(f"Found {len(proxy_pairs_available)} proxy pairs with cached data.")

    summary_rows = []

    for fund_ticker, etf_ticker, pair_name in proxy_pairs_available:
        fund_df_full = _load_data_cache(fund_ticker)
        etf_df_full = _load_data_cache(etf_ticker)

        # Extract daily returns
        fund_ret_col = "Target_Return" if "Target_Return" in fund_df_full.columns else None
        etf_ret_col = "Target_Return" if "Target_Return" in etf_df_full.columns else None

        if fund_ret_col is None or etf_ret_col is None:
            continue

        fund_ret = fund_df_full[fund_ret_col].dropna()
        etf_ret = etf_df_full[etf_ret_col].dropna()

        # Find overlap period
        overlap_start = max(fund_ret.index.min(), etf_ret.index.min())
        overlap_end = min(fund_ret.index.max(), etf_ret.index.max())

        fund_overlap = fund_ret[(fund_ret.index >= overlap_start) & (fund_ret.index <= overlap_end)]
        etf_overlap = etf_ret[(etf_ret.index >= overlap_start) & (etf_ret.index <= overlap_end)]

        # Align on common dates
        common_idx = fund_overlap.index.intersection(etf_overlap.index)
        if len(common_idx) < 60:  # need at least ~3 months of overlap
            continue

        fund_aligned = fund_overlap.loc[common_idx]
        etf_aligned = etf_overlap.loc[common_idx]

        # Metrics
        daily_corr = fund_aligned.corr(etf_aligned)
        return_diff = fund_aligned - etf_aligned
        tracking_error_ann = return_diff.std() * np.sqrt(252) * 100  # in %

        # Cumulative return drift
        fund_cum = (1 + fund_aligned).cumprod()
        etf_cum = (1 + etf_aligned).cumprod()
        total_return_drift = (fund_cum.iloc[-1] - etf_cum.iloc[-1]) / etf_cum.iloc[-1] * 100  # in %

        # Years of overlap
        overlap_years = (overlap_end - overlap_start).days / 365.25

        # Status thresholds
        is_bond_pair = fund_ticker in ("VBMFX", "VUSTX", "VWEHX", "VWESX")
        te_thresh = 1.0 if is_bond_pair else 2.0
        corr_status = "🟢" if daily_corr > 0.99 else ("🟡" if daily_corr > 0.95 else "🔴")
        te_status = "🟢" if tracking_error_ann < te_thresh else ("🟡" if tracking_error_ann < te_thresh * 2 else "🔴")
        drift_status = "🟢" if abs(total_return_drift) < 5 else ("🟡" if abs(total_return_drift) < 15 else "🔴")

        summary_rows.append({
            "Pair": f"{fund_ticker} → {etf_ticker}",
            "Asset Class": pair_name,
            "Overlap": f"{overlap_start.strftime('%Y-%m')} to {overlap_end.strftime('%Y-%m')} ({overlap_years:.1f}y)",
            "Days": len(common_idx),
            "Correlation": f"{daily_corr:.4f} {corr_status}",
            "Tracking Err (ann %)": f"{tracking_error_ann:.2f}% {te_status}",
            "Return Drift": f"{total_return_drift:+.2f}% {drift_status}",
        })

        proxy_results.append({
            "fund_ticker": fund_ticker,
            "etf_ticker": etf_ticker,
            "pair_name": pair_name,
            "common_idx": common_idx,
            "fund_aligned": fund_aligned,
            "etf_aligned": etf_aligned,
            "daily_corr": daily_corr,
            "tracking_error_ann": tracking_error_ann,
            "total_return_drift": total_return_drift,
            "fund_cum": fund_cum,
            "etf_cum": etf_cum,
        })

    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows).set_index("Pair"), use_container_width=True)

        # Detailed charts for each pair
        with st.expander("Detailed Proxy Charts", expanded=False):
            for pr in proxy_results:
                st.markdown(f"### {pr['fund_ticker']} vs {pr['etf_ticker']} ({pr['pair_name']})")

                fig_px, (ax_cum, ax_diff) = plt.subplots(2, 1, figsize=(14, 4.5), sharex=True,
                                                          gridspec_kw={"height_ratios": [2, 1]})

                # Cumulative return comparison
                ax_cum.plot(pr["fund_cum"].index, pr["fund_cum"].values, linewidth=0.8,
                            color="#2980b9", label=pr["fund_ticker"])
                ax_cum.plot(pr["etf_cum"].index, pr["etf_cum"].values, linewidth=0.8,
                            color="#e67e22", label=pr["etf_ticker"], linestyle="--")
                ax_cum.set_ylabel("Cumulative Return")
                ax_cum.legend(fontsize=8)
                ax_cum.set_title(f"Proxy Comparison: {pr['fund_ticker']} vs {pr['etf_ticker']} "
                                 f"(corr={pr['daily_corr']:.4f}, TE={pr['tracking_error_ann']:.2f}%)",
                                 fontsize=10)
                ax_cum.spines[["top", "right"]].set_visible(False)

                # Rolling return difference
                ret_diff = pr["fund_aligned"] - pr["etf_aligned"]
                rolling_diff = ret_diff.rolling(63).mean() * 252 * 100  # annualized rolling 3-month diff
                ax_diff.plot(rolling_diff.index, rolling_diff.values, linewidth=0.7, color="#8e44ad")
                ax_diff.axhline(0, color="gray", linewidth=0.5, linestyle="--")
                ax_diff.set_ylabel("Ann. Ret Diff (%)")
                ax_diff.set_xlabel("")
                ax_diff.spines[["top", "right"]].set_visible(False)
                ax_diff.xaxis.set_major_locator(mdates.YearLocator(2))
                ax_diff.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

                fig_px.tight_layout()
                st.pyplot(fig_px)
                plt.close(fig_px)

    # Flag problematic proxies
    bad_proxies = [r for r in proxy_results if r["daily_corr"] < 0.95 or r["tracking_error_ann"] > 4.0]
    if bad_proxies:
        st.error("Unreliable proxies detected: " +
                 ", ".join(f"{r['fund_ticker']} (corr={r['daily_corr']:.3f}, TE={r['tracking_error_ann']:.1f}%)"
                           for r in bad_proxies) +
                 ". Pre-ETF data from these funds may corrupt model training.")
