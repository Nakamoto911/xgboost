import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import pickle
import glob
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


def _load_data_cache(ticker: str):
    """Load the data cache for a given ticker, trying multiple naming conventions."""
    safe = ticker.replace("^", "").replace("=", "")
    # Try exact match first, then glob for any date suffix (includes _noDD variants)
    candidates = sorted(glob.glob(os.path.join(CACHE_DIR, f"data_cache_{safe}*.pkl")))
    # Also try the legacy data_cache.pkl if ticker is ^SP500TR
    if not candidates and ticker == "^SP500TR":
        legacy = os.path.join(CACHE_DIR, "data_cache.pkl")
        if os.path.exists(legacy):
            candidates = [legacy]
    if not candidates:
        return None
    return pd.read_pickle(candidates[-1])  # newest


def _load_backtest_cache():
    path = os.path.join(CACHE_DIR, "backtest_cache.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# Paper-reported Bear% per asset (from paper Section 4.2 / Figure 2)
PAPER_BEAR_PCT = {
    "^SP500TR": 0.209, "IVV": 0.209,
    "IJH": 0.22, "VIMSX": 0.22,
    "IWM": 0.23, "NAESX": 0.23,
    "IYR": 0.25, "FRESX": 0.25,
    "AGG": 0.30, "VBMFX": 0.30,
    "SPTL": 0.28, "VUSTX": 0.28, "TLT": 0.28, "VGLT": 0.28,
    "DBC": 0.35, "PCASX": 0.35,
    "GLD": 0.30, "GC=F": 0.30, "IAU": 0.30,
    "SPBO": 0.27, "VWESX": 0.27,
    "EEM": 0.32, "VEIEX": 0.32,
    "EFA": 0.29, "FDIVX": 0.29,
    "HYG": 0.26, "VWEHX": 0.26,
}

# ---------------------------------------------------------------------------
# Discover available cached tickers
# ---------------------------------------------------------------------------
available_tickers = []
for f in sorted(glob.glob(os.path.join(CACHE_DIR, "data_cache_*.pkl"))):
    base = os.path.basename(f)
    # data_cache_{ticker}_{date_stuff}.pkl  or  data_cache_{ticker}_{start}_{end}.pkl
    parts = base.replace("data_cache_", "").replace(".pkl", "").split("_")
    if parts:
        tkr = parts[0]
        # Reverse the safe-ticker transform for common index tickers
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

# Also check legacy cache
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
# SECTION 1 — Raw Data Health
# =========================================================================
st.header("Section 1 — Raw Data Health",
          help="Catches Yahoo Finance and FRED download artifacts before they infect features. "
               "Stale prices produce zero returns that corrupt EWMA features; "
               "price adjustment errors create outliers that destabilize JM fitting; "
               "insufficient data history degrades the 11-year lookback window; "
               "FRED yield gaps corrupt the three yield-derived macro features.")

# --- Indicator 1: Zero / Stale Return Streak ---
st.subheader("Indicator 1 — Zero / Stale Return Streak",
             help="Finds the longest run of consecutive days where |daily return| < 0.001%. "
                  "Yahoo Finance mutual fund NAVs (VIMSX, NAESX, etc.) sometimes report stale prices "
                  "that produce zero returns, which corrupt EWMA-based features downstream.\n\n"
                  "**Expected:** 0–1 days for index tickers (^SP500TR), 0–3 days for mutual funds. "
                  "Longer streaks are suspicious — especially mid-week runs, which indicate stale NAVs "
                  "rather than market holidays.\n\n"
                  "**Thresholds:** Green < 2d | Yellow 2–3d (funds) or 2d (indices) | Red > threshold.")

returns = df["Target_Return"] if "Target_Return" in df.columns else df.iloc[:, 0].pct_change()
is_stale = returns.abs() < 0.00001  # |return| < 0.001%

# Compute longest consecutive stale streak
stale_groups = (is_stale != is_stale.shift()).cumsum()
stale_streaks = is_stale.groupby(stale_groups).agg(["sum", "first"])
stale_only = stale_streaks[stale_streaks["first"] == True]
longest_streak = int(stale_only["sum"].max()) if not stale_only.empty else 0

# Find the date range of the longest streak
if longest_streak > 0:
    longest_group = stale_only["sum"].idxmax()
    streak_dates = is_stale[stale_groups == longest_group].index
    streak_start = streak_dates[0].strftime("%Y-%m-%d")
    streak_end = streak_dates[-1].strftime("%Y-%m-%d")
else:
    streak_start = streak_end = "N/A"

# Threshold: index tickers > 1 day, equity/fund tickers > 3 days
is_index = selected_ticker.startswith("^")
threshold = 1 if is_index else 3
status = "🔴" if longest_streak > threshold else ("🟡" if longest_streak > 1 else "🟢")

col1, col2, col3 = st.columns(3)
col1.metric("Longest Stale Streak", f"{longest_streak} days", help="|return| < 0.001%")
col2.metric("Streak Period", f"{streak_start} → {streak_end}")
col3.metric("Status", f"{status} (threshold: {threshold}d)")

# Also show count of all stale days
total_stale = int(is_stale.sum())
st.caption(f"Total stale days: {total_stale} / {len(returns)} ({100*total_stale/len(returns):.2f}%)")

# --- Indicator 2: Return Distribution Outliers ---
st.subheader("Indicator 2 — Return Distribution Outliers",
             help="Shows the 0.1st and 99.9th percentile of daily returns and counts days beyond ±10%. "
                  "Catches Yahoo price adjustment artifacts like incorrect dividend reinvestment or "
                  "stock split errors that create spurious extreme returns.\n\n"
                  "**Expected:** Equities typically have 0.1st/99.9th percentiles within ±8%. "
                  "Bonds within ±3%. Days > ±10% should be rare (< 5 for equities over 30+ years).\n\n"
                  "**Thresholds:** Red if extremes exceed ±15% (equities) or ±5% (bonds/gold).")

p01 = returns.quantile(0.001)
p999 = returns.quantile(0.999)
extreme_days = int((returns.abs() > 0.10).sum())

# Thresholds: ±15% for equities, ±5% for bonds
is_bond = selected_ticker in backend.DD_EXCLUDE_TICKERS
extreme_thresh = 0.05 if is_bond else 0.15
flag_extreme = abs(p01) > extreme_thresh or abs(p999) > extreme_thresh

col1, col2, col3 = st.columns(3)
col1.metric("0.1st percentile", f"{p01*100:.2f}%")
col2.metric("99.9th percentile", f"{p999*100:.2f}%")
status2 = "🔴" if flag_extreme else "🟢"
col3.metric(f"Days > ±10%", f"{extreme_days} {status2}")

if flag_extreme:
    st.warning(f"Extreme percentile exceeds ±{extreme_thresh*100:.0f}% threshold for this asset class.")

# --- Indicator 3: Data Coverage Timeline ---
st.subheader("Indicator 3 — Data Coverage Timeline",
             help="Shows per-year data completeness. The JM requires an 11-year lookback window, "
                  "so partial early years degrade the first OOS chunks.\n\n"
                  "**Expected:** ~252 trading days per complete year. Partial years at the start/end "
                  "of the series are normal. Mid-series gaps are a problem — they create NaN features.\n\n"
                  "**Watch for:** Tickers like VIMSX (starts 1998), GC=F (2000), VEIEX (1994) have "
                  "fewer than 11 years before the 2007 OOS start, meaning early JM fits use truncated windows.")

df_yearly = df.groupby(df.index.year).size()
all_years = range(df.index.year.min(), df.index.year.max() + 1)

# Expected trading days per year (~252, but allow 200 as minimum for "complete")
fig3, ax3 = plt.subplots(figsize=(14, 1.5))
colors = []
counts = []
for yr in all_years:
    n = df_yearly.get(yr, 0)
    counts.append(n)
    if n >= 200:
        colors.append("#2ecc71")  # green: complete
    elif n > 0:
        colors.append("#f39c12")  # yellow: partial
    else:
        colors.append("#e74c3c")  # red: missing

ax3.barh(0, counts, left=[0] + list(np.cumsum(counts[:-1])), color=colors, height=0.5, edgecolor="none")
# Add year labels
cum = 0
for i, yr in enumerate(all_years):
    if counts[i] > 0 and i % 3 == 0:  # label every 3rd year
        ax3.text(cum + counts[i] / 2, 0, str(yr), ha="center", va="center", fontsize=7, color="white", fontweight="bold")
    cum += counts[i]

ax3.set_yticks([])
ax3.set_xlabel("Trading Days")
ax3.set_title(f"{selected_ticker} — Data Coverage ({df.index.year.min()}–{df.index.year.max()})", fontsize=10)
ax3.spines[["top", "right", "left"]].set_visible(False)
st.pyplot(fig3)
plt.close(fig3)

st.caption("Green = 200+ trading days | Yellow = partial year | Red = missing")

# Count years with < 11 years of data before OOS start
oos_start_yr = int(backend.OOS_START_DATE[:4])
data_start_yr = df.index.year.min()
lookback_available = oos_start_yr - data_start_yr
if lookback_available < 11:
    st.warning(f"Only {lookback_available} years of data before OOS start ({backend.OOS_START_DATE}). "
               f"First OOS chunks will use shorter-than-11yr training windows, degrading JM fit quality.")

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

# Try to load raw FRED cache if available
fred_cache_path = os.path.join(CACHE_DIR, "fred_cache.pkl")
has_raw_fred = os.path.exists(fred_cache_path)
if has_raw_fred:
    fred_df = pd.read_pickle(fred_cache_path)
else:
    fred_df = None

# Derived yield features are always in the main data cache
yield_derived = ["Yield_2Y_EWMA_diff", "Yield_Slope_EWMA_10", "Yield_Slope_EWMA_diff_21"]
yield_raw_diff = "Yield_2Y_diff"
has_derived = all(c in df.columns for c in yield_derived)

if has_raw_fred:
    # --- Raw FRED health ---
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

        # Stale detection: consecutive days with identical values
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

    # Range sanity check
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
    # --- Derived yield feature health ---
    if has_raw_fred:
        st.markdown("**Derived Yield Features (in data cache)**")
    else:
        st.markdown("**Derived Yield Features** (raw FRED cache not found — checking derived features only)")

    # Stale Yield_2Y_diff: proxy for whether raw DGS2 was updating
    if yield_raw_diff in df.columns:
        y2d = df[yield_raw_diff]
        # Zero-diff streaks: consecutive days where yield didn't change
        is_zero_diff = y2d.abs() < 1e-10
        zero_groups = (is_zero_diff != is_zero_diff.shift()).cumsum()
        zero_streaks = is_zero_diff.groupby(zero_groups).agg(["sum", "first"])
        zero_only = zero_streaks[zero_streaks["first"] == True]
        longest_zero = int(zero_only["sum"].max()) if not zero_only.empty else 0
        total_zero = int(is_zero_diff.sum())
    else:
        longest_zero = 0
        total_zero = 0

    # NaN check on derived features
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

    # Derived feature ranges
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
        in_range = lo >= exp_lo * 1.5 and hi <= exp_hi * 1.5  # 50% margin
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

# --- Indicator 4: Feature Z-Score Extremes ---
st.subheader("Indicator 4 — Feature Z-Score Extremes (rolling 252-day)",
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
                  "Stock_Bond_Corr is excluded here — it has its own dedicated chart (Indicator 6).")

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
    pass  # already showed info above
else:
    st.info("No macro features (Yield, VIX) found in cached data.")

# Combined summary table
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

# --- Indicator 5: Sortino Clipping Frequency ---
st.subheader("Indicator 5 — Sortino Clipping Frequency",
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
        n_clipped_high = int((s >= 9.99).sum())  # ~10 boundary
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

# --- Indicator 6: Stock-Bond Correlation Rolling Series ---
st.subheader("Indicator 6 — Stock-Bond Correlation (252-day rolling)",
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

    # Week-over-week change alerts (5 trading days)
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
# SECTION 3 — Regime Labeling Health
# =========================================================================
st.header("Section 3 — Regime Labeling Health",
          help="Verifies that the JM produces meaningful, stable regime labels across the walk-forward periods. "
               "Degenerate fits (all one state), extreme label imbalance, and erratic lambda selection "
               "are the main failure modes. Uses the last cached backtest run.")

bc = _load_backtest_cache()
if bc is None:
    st.warning("No backtest cache found (`cache/backtest_cache.pkl`). Run a backtest from the Performance Tracker to populate it.")
    st.stop()

cache_version = bc.get("cache_version", 1)
if cache_version >= 2:
    bt_df = bc["jm_xgb_df"]
else:
    st.warning("Legacy backtest cache format — please re-run the backtest to get v2 format.")
    st.stop()

if "JM_Target_State" not in bt_df.columns:
    st.warning("Backtest cache does not contain JM_Target_State column. Re-run the backtest.")
    st.stop()

lambda_history = bc.get("lambda_history", [])
lambda_dates_str = bc.get("lambda_dates", [])
lambda_dates = [pd.to_datetime(d) for d in lambda_dates_str] if lambda_dates_str else []
config_name = bc.get("config_name", "Unknown")

st.caption(f"Backtest config: **{config_name}** | EWMA halflife: {bc.get('best_ewma_hl', '?')}")

# Split backtest into 6-month chunks aligned with lambda_dates
chunks = []
if lambda_dates:
    for i, d in enumerate(lambda_dates):
        end_d = lambda_dates[i + 1] if i + 1 < len(lambda_dates) else bt_df.index[-1] + pd.Timedelta(days=1)
        chunk = bt_df[(bt_df.index >= d) & (bt_df.index < end_d)]
        if not chunk.empty:
            lmbda = lambda_history[i] if i < len(lambda_history) else np.nan
            chunks.append({"start": d, "end": end_d, "df": chunk, "lambda": lmbda})
else:
    # Fallback: split by 6-month intervals
    start = bt_df.index[0]
    while start < bt_df.index[-1]:
        end_d = start + pd.DateOffset(months=6)
        chunk = bt_df[(bt_df.index >= start) & (bt_df.index < end_d)]
        if not chunk.empty:
            chunks.append({"start": start, "end": end_d, "df": chunk, "lambda": np.nan})
        start = end_d

if not chunks:
    st.warning("Could not split backtest into chunks.")
    st.stop()

# --- Indicator 7: Bear Market Fraction per Period ---
st.subheader("Indicator 7 — Bear Fraction per OOS Chunk",
             help="For each 6-month OOS period, shows the fraction of days the JM's online Viterbi "
                  "assigned to the Bear state (State 1). The green dashed line is the paper's reported "
                  "overall Bear% for this asset class.\n\n"
                  "**Expected:** The paper reports ~20.9% Bear for LargeCap overall. Individual chunks "
                  "vary — GFC chunks (2008–2009) should be 50–80% Bear, late bull market chunks "
                  "(2013–2014, 2017–2018) should be near 0% Bear. This variation is expected.\n\n"
                  "**Red flags:** A chunk at exactly 0% or 100% Bear means JM assigned all days to one state "
                  "(degenerate fit). This typically happens when lambda is too high (over-penalizing transitions) "
                  "or the training window passed through a structural break.")

bear_fracs = []
chunk_labels = []
for c in chunks:
    jm_states = c["df"]["JM_Target_State"]
    bear_frac = (jm_states == 1).mean()
    bear_fracs.append(bear_frac)
    chunk_labels.append(c["start"].strftime("%Y-%m"))

paper_bear = PAPER_BEAR_PCT.get(selected_ticker, None)
# Also check backtest cache ticker if different
cache_ticker = bc.get("target_ticker", selected_ticker)
if paper_bear is None:
    paper_bear = PAPER_BEAR_PCT.get(cache_ticker, None)

fig7, ax7 = plt.subplots(figsize=(14, 3.5))
bar_colors = []
for bf in bear_fracs:
    if bf < 0.02 or bf > 0.80:
        bar_colors.append("#e74c3c")  # red: degenerate
    elif bf < 0.05 or bf > 0.60:
        bar_colors.append("#f39c12")  # yellow: suspicious
    else:
        bar_colors.append("#3498db")  # blue: normal

ax7.bar(range(len(bear_fracs)), bear_fracs, color=bar_colors, edgecolor="none", width=0.7)
if paper_bear is not None:
    ax7.axhline(paper_bear, color="#2ecc71", linewidth=1.5, linestyle="--", label=f"Paper: {paper_bear*100:.1f}%")
    ax7.legend(fontsize=8)
ax7.set_xticks(range(len(chunk_labels)))
ax7.set_xticklabels(chunk_labels, rotation=45, ha="right", fontsize=7)
ax7.set_ylabel("Bear Fraction")
ax7.set_title("JM Bear State Fraction per 6-Month OOS Chunk", fontsize=10)
ax7.set_ylim(0, 1)
ax7.spines[["top", "right"]].set_visible(False)
fig7.tight_layout()
st.pyplot(fig7)
plt.close(fig7)

# Flag degenerate chunks
degenerate = [(chunk_labels[i], bear_fracs[i]) for i in range(len(bear_fracs)) if bear_fracs[i] < 0.02 or bear_fracs[i] > 0.80]
if degenerate:
    st.warning(f"Degenerate chunks (0% or 80%+ Bear): {', '.join(f'{lbl} ({bf*100:.0f}%)' for lbl, bf in degenerate)}")

# --- Indicator 8: Regime Label Imbalance per Chunk (Training Window) ---
st.subheader("Indicator 8 — Training Label Imbalance per Chunk",
             help="Shows the Bull/Bear class split that XGBoost receives as training labels for each chunk. "
                  "XGBoost trains on JM-assigned regime labels from the 11-year lookback window; "
                  "extreme imbalance means XGB is fitting a near-constant classifier.\n\n"
                  "**Expected:** ~75–80% Bull / 20–25% Bear for LargeCap (matching paper's overall Bear%). "
                  "Per-chunk OOS states shown here are a proxy — actual training labels come from the "
                  "11-year window which is more stable.\n\n"
                  "**Red flags:** Minority class < 10% (red) means XGB can't learn meaningful discrimination. "
                  "10–15% (yellow) is borderline — XGB may overfit to the minority class. "
                  "This is especially common for low-lambda fits where JM produces fewer state transitions.")
st.caption("Cannot reconstruct exact training labels from cache — using OOS JM states as proxy.")

# We use the OOS JM_Target_State as the best available proxy since we don't store training labels.
# The overall bear fraction across the full OOS gives a sense of label balance.
overall_bear = (bt_df["JM_Target_State"] == 1).mean()
overall_bull = 1 - overall_bear

col1, col2, col3 = st.columns(3)
col1.metric("Overall OOS Bull%", f"{overall_bull*100:.1f}%")
col2.metric("Overall OOS Bear%", f"{overall_bear*100:.1f}%")
status8 = "🔴" if overall_bear < 0.10 or overall_bull < 0.10 else "🟢"
col3.metric("Balance Status", status8)

# Per-chunk view
imbalance_data = []
for i, c in enumerate(chunks):
    jm_states = c["df"]["JM_Target_State"]
    bear_pct = (jm_states == 1).mean() * 100
    bull_pct = 100 - bear_pct
    minority = min(bear_pct, bull_pct)
    flag = "🔴" if minority < 10 else ("🟡" if minority < 15 else "🟢")
    imbalance_data.append({
        "Period": chunk_labels[i],
        "Bull%": f"{bull_pct:.1f}%",
        "Bear%": f"{bear_pct:.1f}%",
        "Minority%": f"{minority:.1f}%",
        "Status": flag,
        "λ": f"{c['lambda']:.1f}" if not np.isnan(c["lambda"]) else "?",
    })

st.dataframe(pd.DataFrame(imbalance_data).set_index("Period"), use_container_width=True)

# --- Indicator 9: JM Objective Value Stability ---
st.subheader("Indicator 9 — Lambda Selection Stability",
             help="Lambda (the JM transition penalty) is re-tuned every 6 months by maximizing Sharpe "
                  "on the validation window. This chart shows which lambda was selected at each period.\n\n"
                  "**Expected:** Stable strategies pick a consistent lambda (CV < 0.3). "
                  "The Sub-Window Consensus method (Experiment 11) typically yields CV ~0.26. "
                  "Standard argmax selection can be more erratic (CV > 0.5).\n\n"
                  "**Interpretation:** Mean λ in the 15–50 range is typical for LargeCap. "
                  "Large jumps (> 100% change between periods) mean the validation Sharpe landscape is flat "
                  "or multi-modal — different lambdas look similarly good, so the selection is noise-driven. "
                  "Lambda smoothing (Experiment 5) or Sub-Window Consensus can stabilize this.")

if lambda_history and lambda_dates:
    fig9, ax9 = plt.subplots(figsize=(14, 3))
    ax9.plot(lambda_dates, lambda_history, marker="o", markersize=4, linewidth=1.2, color="#8e44ad")
    ax9.fill_between(lambda_dates, lambda_history, alpha=0.15, color="#8e44ad")
    ax9.set_ylabel("Lambda")
    ax9.set_title("Walk-Forward Lambda Selection Over Time", fontsize=10)
    ax9.xaxis.set_major_locator(mdates.YearLocator(2))
    ax9.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax9.spines[["top", "right"]].set_visible(False)
    fig9.tight_layout()
    st.pyplot(fig9)
    plt.close(fig9)

    # Stats
    lh = np.array(lambda_history)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean λ", f"{lh.mean():.1f}")
    col2.metric("Std λ", f"{lh.std():.1f}")
    cv = lh.std() / lh.mean() if lh.mean() > 0 else 0
    col3.metric("CV", f"{cv:.2f}")
    # Count large jumps (> 2x change between periods)
    diffs = np.abs(np.diff(lh))
    large_jumps = int((diffs > lh[:-1] * 1.0).sum())  # jump > 100% of previous
    col4.metric("Large Jumps", large_jumps)
else:
    st.info("No lambda history available in backtest cache.")
