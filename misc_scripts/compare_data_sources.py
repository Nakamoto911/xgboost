"""
Data Source Comparison: Yahoo ETFs vs Yahoo Mutual Funds vs Bloomberg Indices.

Loads daily Target_Return for each (paper_asset, source) triplet from the
existing per-ticker caches (data_cache_<TICKER>_<YYYYMMDD>[_noDD]_v2.pkl),
aligns on the 2007-2023 paper OOS window, and computes per-source statistics
plus pair-wise divergence metrics. Generates a markdown report with embedded
PNG charts.

Reads from caches only — does NOT re-fetch data. If a cache is missing, run
`benchmark_assets.py "Yahoo ETFs"`, `... "Yahoo Mutual Funds"`,
`... "Bloomberg Indices"` first (each populates one source's caches).

CLI:
    python misc_scripts/compare_data_sources.py              # all 12 assets
    python misc_scripts/compare_data_sources.py largecap     # 1 asset
    python misc_scripts/compare_data_sources.py tier1        # LargeCap
    python misc_scripts/compare_data_sources.py tier2        # +REIT, AggBond
    python misc_scripts/compare_data_sources.py list         # show asset names
"""

import base64
import io
import os
import pickle
import sys
import time
import warnings
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, 'cache')
BENCHMARKS_DIR = os.path.join(PROJECT_ROOT, 'benchmarks')
CHARTS_DIR = os.path.join(BENCHMARKS_DIR, 'data_comparison_charts')
os.makedirs(BENCHMARKS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

# ── Asset registry ────────────────────────────────────────────────────────────
# Maps paper-asset → 3 sources × (ticker, data_start_used_for_cache_filename, dd_excluded)
# data_start strings come from asset_lists.md (Yahoo ETFs=1993, MF=1975, BBG=1989)
# dd_excluded matches benchmark_assets.DD_EXCLUDE_TICKERS.

DD_EXCLUDE = {
    'AGG', 'VBMFX', 'SPTL', 'VUSTX', 'IEF', 'TLT', 'VGLT',
    'GLD', 'GC=F', 'IAU',
    'LBUSTRUU', 'LUTLTRUU', 'GOLDLNPM',
}

ASSET_REGISTRY = {
    'LargeCap':  {'Yahoo ETF': 'IVV',  'Yahoo MF': '^SP500TR', 'Bloomberg': 'SPTR'},
    'MidCap':    {'Yahoo ETF': 'IJH',  'Yahoo MF': 'VIMSX',    'Bloomberg': 'SPTRMDCP'},
    'SmallCap':  {'Yahoo ETF': 'IWM',  'Yahoo MF': 'NAESX',    'Bloomberg': 'RU20INTR'},
    'EAFE':      {'Yahoo ETF': 'EFA',  'Yahoo MF': 'FDIVX',    'Bloomberg': 'NDDUEAFE'},
    'EM':        {'Yahoo ETF': 'EEM',  'Yahoo MF': 'VEIEX',    'Bloomberg': 'NDUEEGF'},
    'AggBond':   {'Yahoo ETF': 'AGG',  'Yahoo MF': 'VBMFX',    'Bloomberg': 'LBUSTRUU'},
    'Treasury':  {'Yahoo ETF': 'SPTL', 'Yahoo MF': 'VUSTX',    'Bloomberg': 'LUTLTRUU'},
    'HighYield': {'Yahoo ETF': 'HYG',  'Yahoo MF': 'VWEHX',    'Bloomberg': 'IBOXHY'},
    'Corporate': {'Yahoo ETF': 'SPBO', 'Yahoo MF': 'VWESX',    'Bloomberg': 'LUACTRUU'},
    'REIT':      {'Yahoo ETF': 'IYR',  'Yahoo MF': 'FRESX',    'Bloomberg': 'DJUSRET'},
    'Commodity': {'Yahoo ETF': 'DBC',  'Yahoo MF': '^SPGSCI',  'Bloomberg': 'DBLCDBCE'},
    'Gold':      {'Yahoo ETF': 'GLD',  'Yahoo MF': 'GC=F',     'Bloomberg': 'GOLDLNPM'},
}
LIST_DATA_START = {'Yahoo ETF': '1993-01-01', 'Yahoo MF': '1975-01-01', 'Bloomberg': '1989-01-01'}

# Paper Table 4 B&H reference (2007-2023, Bloomberg, gross of TC)
PAPER_BH = {
    'LargeCap':  {'sharpe': 0.50, 'mdd': -0.5525},
    'MidCap':    {'sharpe': 0.45, 'mdd': -0.5515},
    'SmallCap':  {'sharpe': 0.36, 'mdd': -0.5889},
    'EAFE':      {'sharpe': 0.20, 'mdd': -0.6041},
    'EM':        {'sharpe': 0.20, 'mdd': -0.6525},
    'AggBond':   {'sharpe': 0.46, 'mdd': -0.1841},
    'Treasury':  {'sharpe': 0.26, 'mdd': -0.4691},
    'HighYield': {'sharpe': 0.67, 'mdd': -0.3287},
    'Corporate': {'sharpe': 0.54, 'mdd': -0.2204},
    'REIT':      {'sharpe': 0.27, 'mdd': -0.7423},
    'Commodity': {'sharpe': 0.03, 'mdd': -0.7554},
    'Gold':      {'sharpe': 0.43, 'mdd': -0.4462},
}

TIERS = {
    'tier1':  ['LargeCap'],
    'tier2':  ['LargeCap', 'REIT', 'AggBond'],
    'all':    list(ASSET_REGISTRY.keys()),
}

OOS_START = pd.Timestamp('2007-01-01')
OOS_END   = pd.Timestamp('2023-12-31')


# ── Cache loading ─────────────────────────────────────────────────────────────

def cache_path_for(ticker: str, source: str) -> str:
    data_start = LIST_DATA_START[source].replace('-', '')
    dd_suffix = '_noDD' if ticker in DD_EXCLUDE else ''
    return os.path.join(CACHE_DIR, f'data_cache_{ticker}_{data_start}{dd_suffix}_v2.pkl')


def load_returns(ticker: str, source: str):
    """Return (Target_Return Series, RF_Rate Series, full_start, full_end, mtime, full_df_for_excess).
    Returns None if cache missing."""
    p = cache_path_for(ticker, source)
    if not os.path.exists(p):
        return None
    df = pd.read_pickle(p)
    mtime = os.path.getmtime(p)
    full_start = df.index.min()
    full_end = df.index.max()
    rets = df['Target_Return']
    rf = df['RF_Rate']
    return rets, rf, full_start, full_end, mtime


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_drawdown(returns: pd.Series) -> float:
    """Max drawdown from daily simple returns."""
    wealth = (1.0 + returns).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())


def compute_stats(rets: pd.Series, rf: pd.Series) -> dict:
    """Return-based stats for a daily simple-return series, in given window."""
    if rets is None or len(rets) < 50:
        return {}
    excess = rets - rf
    n = len(rets)
    ann_ret = (1.0 + rets).prod() ** (252.0 / n) - 1.0
    ann_vol = rets.std() * np.sqrt(252)
    ann_excess = excess.mean() * 252
    sharpe = ann_excess / ann_vol if ann_vol > 0 else np.nan
    downside = np.minimum(excess, 0).std() * np.sqrt(252)
    sortino = ann_excess / downside if downside > 0 else np.nan
    mdd = compute_drawdown(rets)
    skew = rets.skew()
    kurt = rets.kurt()
    return {
        'n': n,
        'ann_ret': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'mdd': mdd,
        'skew': skew,
        'kurt': kurt,
    }


def compare_pair(a: pd.Series, b: pd.Series, label_a: str, label_b: str) -> dict:
    """Pairwise divergence metrics on aligned daily returns."""
    df = pd.concat([a.rename('a'), b.rename('b')], axis=1).dropna()
    if len(df) < 50:
        return {'n': len(df)}
    diff = df['a'] - df['b']
    n = len(df)
    ann_drift = diff.mean() * 252
    te_ann = diff.std() * np.sqrt(252)
    pearson = df['a'].corr(df['b'])
    spearman = df['a'].corr(df['b'], method='spearman')
    abs_diff_mean = diff.abs().mean()
    worst_days = diff.abs().sort_values(ascending=False).head(5)
    return {
        'pair': f'{label_a} vs {label_b}',
        'n': n,
        'pearson': pearson,
        'spearman': spearman,
        'ann_drift': ann_drift,
        'te_ann': te_ann,
        'abs_diff_mean': abs_diff_mean,
        'worst_days': [(d.strftime('%Y-%m-%d'), float(v)) for d, v in worst_days.items()],
    }


# ── Charts ────────────────────────────────────────────────────────────────────
# One 3×2 combined PNG per asset, saved to disk under
# benchmarks/data_comparison_charts/{asset}.png and served by Streamlit via
# st.image() — base64 inlining made the page freeze because browsers had to
# decode all 12 large data URIs eagerly inside st.expander.

SOURCE_COLORS = {'Yahoo ETF': '#1f77b4', 'Yahoo MF': '#2ca02c', 'Bloomberg': '#d62728'}
CHART_SCHEMA_VERSION = 7  # bump to invalidate per-asset caches when chart format changes


def _save_fig(fig, path: str, dpi: int = 110) -> str:
    fig.savefig(path, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def _downsample_for_plot(s: pd.Series, target_points: int = 2000) -> pd.Series:
    """Reduce points before plotting to shrink PNG size. Preserves first/last."""
    n = len(s)
    if n <= target_points:
        return s
    step = max(1, n // target_points)
    return s.iloc[::step]


def _style_ax(ax, title=None, ylabel=None, xlabel=None):
    """Consistent styling: bigger fonts, lighter spines, soft grid."""
    if title:
        ax.set_title(title, fontsize=11, pad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.6)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color('#888')


def chart_combined(returns_oos: dict, returns_full: dict, asset_name: str) -> str:
    """3×2 figure: price level · drift · rolling return · rolling vol · rolling corr · distribution.

    `returns_oos` covers the paper OOS window (2007-2023). `returns_full` covers
    each source's full available history — used for the price-level panel so the
    raw index level can be inspected back to inception.
    """
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 12.5))
    bbg = returns_oos.get('Bloomberg')

    # (0,0) Raw price level (full history) — all sources anchored at 100 on
    # the OOS start date so their trajectories are directly comparable. Sources
    # whose history starts AFTER OOS_START are anchored at their own first date
    # instead (e.g. HYG starts 2007-04). Pre-OOS values appear < 100 and show
    # where each source came from. Any vertical separation post-OOS is pure
    # data divergence.
    ax = axes[0, 0]
    for src, rets in returns_full.items():
        if rets is None or rets.empty:
            continue
        # Cumulative growth factor from each source's first date.
        growth = (1.0 + rets.fillna(0)).cumprod()
        # Anchor: value of `growth` on/at OOS_START (or first available date
        # if the source starts later). Rescale so growth[anchor] = 100.
        anchor_idx = growth.index.searchsorted(OOS_START)
        if anchor_idx >= len(growth):
            anchor_idx = len(growth) - 1
        anchor_value = growth.iloc[anchor_idx]
        if anchor_value <= 0 or np.isnan(anchor_value):
            continue
        price = 100.0 * growth / anchor_value
        price = _downsample_for_plot(price, target_points=2500)
        ax.plot(price.index, price.values, color=SOURCE_COLORS[src], label=src, lw=1.6)
    ax.axvline(OOS_START, color='black', lw=0.9, alpha=0.55, linestyle='--')
    ax.axhline(100, color='black', lw=0.6, alpha=0.3)
    ax.text(OOS_START, 0.97, ' OOS start (=100)',
            fontsize=8, color='#444', verticalalignment='top',
            transform=ax.get_xaxis_transform())
    ax.set_yscale('log')
    _style_ax(ax, title='Normalized price level — full history (=100 at OOS start, log scale)',
              ylabel='Index level (=100 at OOS start)')
    ax.legend(loc='best', fontsize=9, framealpha=0.85)

    # (0,1) Cumulative log-return drift vs Bloomberg
    ax = axes[0, 1]
    if bbg is not None and not bbg.empty:
        for src in ('Yahoo ETF', 'Yahoo MF'):
            r = returns_oos.get(src)
            if r is None or r.empty:
                continue
            df = pd.concat([r.rename('a'), bbg.rename('b')], axis=1).dropna()
            cum_drift = (np.log1p(df['a']) - np.log1p(df['b'])).cumsum() * 100
            cum_drift = _downsample_for_plot(cum_drift)
            ax.plot(cum_drift.index, cum_drift.values, color=SOURCE_COLORS[src], label=src, lw=1.6)
        ax.axhline(0, color='black', lw=0.8, alpha=0.5)
    _style_ax(ax, title='Cumulative log-return drift vs Bloomberg',
              ylabel='Σ log(1+r) − Σ log(1+r_BBG), % pts')
    ax.legend(loc='best', fontsize=9, framealpha=0.85)

    # (1,0) Rolling 1y annualized return — direct return comparison
    ax = axes[1, 0]
    for src, rets in returns_oos.items():
        if rets is None or rets.empty:
            continue
        roll_ret = rets.rolling(252).mean() * 252 * 100  # annualized %, arithmetic
        roll_ret = _downsample_for_plot(roll_ret)
        ax.plot(roll_ret.index, roll_ret.values, color=SOURCE_COLORS[src], label=src, lw=1.6)
    ax.axhline(0, color='black', lw=0.8, alpha=0.5)
    _style_ax(ax, title='Rolling 1-year annualized return',
              ylabel='Annualized return (%)')
    ax.legend(loc='best', fontsize=9, framealpha=0.85)

    # (1,1) Rolling 1y annualized volatility — direct vol comparison
    ax = axes[1, 1]
    for src, rets in returns_oos.items():
        if rets is None or rets.empty:
            continue
        roll_vol = rets.rolling(252).std() * np.sqrt(252) * 100  # annualized %
        roll_vol = _downsample_for_plot(roll_vol)
        ax.plot(roll_vol.index, roll_vol.values, color=SOURCE_COLORS[src], label=src, lw=1.6)
    _style_ax(ax, title='Rolling 1-year annualized volatility',
              ylabel='Annualized volatility (%)')
    ax.legend(loc='best', fontsize=9, framealpha=0.85)

    # (2,0) Rolling 1y correlation with Bloomberg
    ax = axes[2, 0]
    if bbg is not None and not bbg.empty:
        for src in ('Yahoo ETF', 'Yahoo MF'):
            r = returns_oos.get(src)
            if r is None or r.empty:
                continue
            df = pd.concat([r.rename('a'), bbg.rename('b')], axis=1).dropna()
            rc = df['a'].rolling(252).corr(df['b'])
            rc = _downsample_for_plot(rc)
            ax.plot(rc.index, rc.values, color=SOURCE_COLORS[src], label=src, lw=1.6)
    ax.set_ylim(-0.05, 1.05)
    _style_ax(ax, title='Rolling 1-year correlation with Bloomberg',
              ylabel='Pearson ρ')
    ax.legend(loc='best', fontsize=9, framealpha=0.85)

    # (2,1) Daily return distribution — KDE lines
    from scipy.stats import gaussian_kde
    ax = axes[2, 1]
    for src, rets in returns_oos.items():
        if rets is None or rets.empty:
            continue
        data = rets.dropna().values * 100
        xs = np.linspace(-5, 5, 400)
        kde = gaussian_kde(data, bw_method='scott')
        ax.plot(xs, kde(xs), label=src, color=SOURCE_COLORS[src], linewidth=1.8)
    ax.set_xlim(-5, 5)
    ax.set_ylim(bottom=0)
    _style_ax(ax, title='Daily return distribution',
              xlabel='Daily return (%)', ylabel='Density')
    ax.legend(loc='best', fontsize=9, framealpha=0.85)

    fig.suptitle(
        f'{asset_name} — Data Source Comparison (2007-2023)',
        fontsize=14, fontweight='bold', y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_path = os.path.join(CHARTS_DIR, f'{asset_name}.png')
    _save_fig(fig, out_path)
    # Return path relative to benchmarks/ so markdown image links work both in
    # the in-app viewer (resolved by Streamlit) and from the .md file's dir.
    return os.path.relpath(out_path, BENCHMARKS_DIR)


def chart_summary_divergence(results: list) -> str:
    """Two-panel horizontal bar chart: ΔReturn and ΔVolatility vs Bloomberg, all assets.
    Eagerly rendered at the top of the report — one-look diagnosis."""
    asset_names = [r['asset_name'] for r in results if r['per_source_stats'].get('Bloomberg')]
    if not asset_names:
        return ''

    # Pull deltas per source (in % units)
    def _delta(r, src, key):
        bbg = r['per_source_stats'].get('Bloomberg', {})
        s = r['per_source_stats'].get(src, {})
        if not bbg or not s:
            return np.nan
        return (s.get(key, np.nan) - bbg.get(key, np.nan)) * 100

    d_ret_etf = [_delta(r, 'Yahoo ETF', 'ann_ret') for r in results]
    d_ret_mf  = [_delta(r, 'Yahoo MF',  'ann_ret') for r in results]
    d_vol_etf = [_delta(r, 'Yahoo ETF', 'ann_vol') for r in results]
    d_vol_mf  = [_delta(r, 'Yahoo MF',  'ann_vol') for r in results]

    y = np.arange(len(asset_names))
    h = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13.5, max(5.5, 0.55 * len(asset_names))))

    # Δ Return
    ax = axes[0]
    ax.barh(y - h / 2, d_ret_etf, height=h, color=SOURCE_COLORS['Yahoo ETF'], label='Yahoo ETF − BBG')
    ax.barh(y + h / 2, d_ret_mf,  height=h, color=SOURCE_COLORS['Yahoo MF'],  label='Yahoo MF − BBG')
    ax.axvline(0, color='black', lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(asset_names)
    ax.invert_yaxis()
    _style_ax(ax, title='Δ Annualized return vs Bloomberg (% pts)',
              xlabel='Yahoo − Bloomberg (%)')
    ax.legend(loc='best', fontsize=9, framealpha=0.85)

    # Δ Vol
    ax = axes[1]
    ax.barh(y - h / 2, d_vol_etf, height=h, color=SOURCE_COLORS['Yahoo ETF'], label='Yahoo ETF − BBG')
    ax.barh(y + h / 2, d_vol_mf,  height=h, color=SOURCE_COLORS['Yahoo MF'],  label='Yahoo MF − BBG')
    ax.axvline(0, color='black', lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(asset_names)
    ax.invert_yaxis()
    _style_ax(ax, title='Δ Annualized volatility vs Bloomberg (% pts)',
              xlabel='Yahoo − Bloomberg (%)')
    ax.legend(loc='best', fontsize=9, framealpha=0.85)

    fig.suptitle('Return & Volatility divergence — Yahoo sources vs Bloomberg',
                 fontsize=13, fontweight='bold', y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = os.path.join(CHARTS_DIR, '_summary_divergence.png')
    _save_fig(fig, out_path)
    return os.path.relpath(out_path, BENCHMARKS_DIR)


# ── Per-asset orchestration ───────────────────────────────────────────────────

def analyze_asset(asset_name: str, use_cache: bool = True) -> dict:
    """Compute and cache full analysis result for one paper-asset."""
    cache_file = os.path.join(CACHE_DIR, f'data_comparison_{asset_name}.pkl')

    # Cache freshness: compare against source cache mtimes
    sources = ASSET_REGISTRY[asset_name]
    source_mtimes = {}
    for src, ticker in sources.items():
        p = cache_path_for(ticker, src)
        source_mtimes[src] = os.path.getmtime(p) if os.path.exists(p) else 0.0

    if use_cache and os.path.exists(cache_file):
        cached = pickle.load(open(cache_file, 'rb'))
        if (cached.get('source_mtimes') == source_mtimes
                and cached.get('schema_version') == CHART_SCHEMA_VERSION):
            return cached

    # Load returns for each source
    returns_full = {}     # full available history per source
    returns_oos = {}      # 2007-2023 window per source
    rf_oos = {}
    full_coverage = {}
    missing = []
    for src, ticker in sources.items():
        loaded = load_returns(ticker, src)
        if loaded is None:
            missing.append((src, ticker))
            returns_full[src] = None
            returns_oos[src] = None
            rf_oos[src] = None
            full_coverage[src] = None
            continue
        rets, rf, fstart, fend, _ = loaded
        returns_full[src] = rets
        oos_rets = rets.loc[OOS_START:OOS_END]
        oos_rf = rf.loc[OOS_START:OOS_END]
        returns_oos[src] = oos_rets
        rf_oos[src] = oos_rf
        full_coverage[src] = {'ticker': ticker, 'start': fstart, 'end': fend, 'n': len(rets)}

    # Per-source stats
    per_source_stats = {}
    for src, rets in returns_oos.items():
        rf = rf_oos.get(src)
        if rets is None or rf is None:
            per_source_stats[src] = {}
            continue
        per_source_stats[src] = compute_stats(rets, rf)
        per_source_stats[src]['ticker'] = sources[src]

    # Pair-wise divergence
    pair_keys = [('Yahoo ETF', 'Bloomberg'), ('Yahoo MF', 'Bloomberg'), ('Yahoo ETF', 'Yahoo MF')]
    pair_stats = []
    for a, b in pair_keys:
        if returns_oos.get(a) is None or returns_oos.get(b) is None:
            pair_stats.append({'pair': f'{a} vs {b}', 'n': 0})
            continue
        pair_stats.append(compare_pair(returns_oos[a], returns_oos[b], a, b))

    # Single combined chart (3×2), saved to disk. Path is relative to benchmarks/.
    chart_relpath = chart_combined(returns_oos, returns_full, asset_name)

    result = {
        'asset_name': asset_name,
        'sources': sources,
        'full_coverage': full_coverage,
        'missing': missing,
        'per_source_stats': per_source_stats,
        'pair_stats': pair_stats,
        'paper_bh': PAPER_BH.get(asset_name),
        'chart_relpath': chart_relpath,
        'source_mtimes': source_mtimes,
        'schema_version': CHART_SCHEMA_VERSION,
        'generated_at': datetime.now().isoformat(timespec='seconds'),
    }
    pickle.dump(result, open(cache_file, 'wb'))
    return result


# ── Report rendering ──────────────────────────────────────────────────────────

def _fmt_pct(x, places=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '—'
    return f'{x * 100:.{places}f}%'


def _fmt_num(x, places=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '—'
    return f'{x:.{places}f}'


def _one_line_headline(result: dict) -> str:
    """Compact headline used in the <details> summary so users can scan w/o expanding."""
    per = result['per_source_stats']
    bbg = per.get('Bloomberg', {})
    ye = per.get('Yahoo ETF', {})
    ym = per.get('Yahoo MF', {})

    def _s(d):
        v = d.get('sharpe', np.nan) if d else np.nan
        return '—' if np.isnan(v) else f'{v:.2f}'

    parts = [f"BBG={_s(bbg)}", f"Y_ETF={_s(ye)}", f"Y_MF={_s(ym)}"]
    pairs = {p['pair']: p for p in result['pair_stats']}
    pe = pairs.get('Yahoo ETF vs Bloomberg', {})
    pm = pairs.get('Yahoo MF vs Bloomberg', {})
    if pe.get('pearson') is not None and pm.get('pearson') is not None:
        parts.append(f"ρ_ETF={pe.get('pearson', float('nan')):.3f}")
        parts.append(f"ρ_MF={pm.get('pearson', float('nan')):.3f}")
    return ' · '.join(parts)


def render_asset_section(result: dict) -> str:
    asset = result['asset_name']
    sources = result['sources']
    paper = result['paper_bh']

    out = []
    # Collapsible per-asset section: only DOM-cheap summary + 1 PNG when expanded.
    # Streamlit's st.markdown renders <details>, and browsers defer rendering
    # of hidden image data — keeping the page responsive for All-12 reports.
    headline = _one_line_headline(result)
    out.append(f'<details>')
    out.append(
        f'<summary><b>{asset}</b> — Sharpe {headline} '
        f'(Yahoo ETF=<code>{sources["Yahoo ETF"]}</code> · '
        f'Yahoo MF=<code>{sources["Yahoo MF"]}</code> · '
        f'Bloomberg=<code>{sources["Bloomberg"]}</code>)</summary>'
    )
    out.append('')

    # Missing caches
    if result['missing']:
        out.append('**⚠️ Missing caches:**')
        for src, t in result['missing']:
            out.append(f'  - {src}: `{t}` (run benchmark_assets.py for this list)')
        out.append('')

    # Coverage
    out.append('### Coverage (full available history)')
    out.append('')
    out.append('| Source | Ticker | Start | End | Rows |')
    out.append('|---|---|---|---|---:|')
    for src in ('Yahoo ETF', 'Yahoo MF', 'Bloomberg'):
        c = result['full_coverage'].get(src)
        if c is None:
            out.append(f'| {src} | — | — | — | — |')
        else:
            out.append(f"| {src} | `{c['ticker']}` | {c['start'].date()} | {c['end'].date()} | {c['n']:,} |")
    out.append('')

    # Per-source stats in OOS window 2007-2023
    out.append('### B&H Statistics — 2007-01-01 to 2023-12-31 (paper OOS window)')
    out.append('')
    out.append('| Source | Ticker | Ann Ret | Ann Vol | Sharpe | Sortino | MDD | Skew | Kurt | n |')
    out.append('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for src in ('Yahoo ETF', 'Yahoo MF', 'Bloomberg'):
        s = result['per_source_stats'].get(src, {})
        if not s:
            out.append(f'| {src} | — | — | — | — | — | — | — | — | — |')
            continue
        out.append(
            f"| {src} | `{s['ticker']}` | {_fmt_pct(s['ann_ret'])} | {_fmt_pct(s['ann_vol'])} "
            f"| {_fmt_num(s['sharpe'])} | {_fmt_num(s['sortino'])} | {_fmt_pct(s['mdd'])} "
            f"| {_fmt_num(s['skew'], 2)} | {_fmt_num(s['kurt'], 2)} | {s['n']:,} |"
        )
    out.append('')

    # Explicit deltas vs Bloomberg — what matters most for MVO (μ, Σ).
    bbg = result['per_source_stats'].get('Bloomberg', {})
    if bbg:
        out.append('**Δ vs Bloomberg** (positive = Yahoo source higher than BBG)')
        out.append('')
        out.append('| Source | Δ Ann Ret | Δ Ann Vol | Δ Sharpe | Δ MDD |')
        out.append('|---|---:|---:|---:|---:|')
        for src in ('Yahoo ETF', 'Yahoo MF'):
            s = result['per_source_stats'].get(src, {})
            if not s:
                out.append(f'| {src} | — | — | — | — |')
                continue
            d_ret = s['ann_ret'] - bbg['ann_ret']
            d_vol = s['ann_vol'] - bbg['ann_vol']
            d_sh  = s['sharpe']  - bbg['sharpe']
            d_mdd = s['mdd']     - bbg['mdd']
            out.append(
                f"| {src} | {d_ret:+.2%} | {d_vol:+.2%} | {d_sh:+.3f} | {d_mdd:+.2%} |"
            )
        out.append('')

    # Paper B&H reference comparison
    if paper:
        out.append('### vs Paper Table 4 B&H reference (Bloomberg, gross of TC)')
        out.append('')
        out.append(f"Paper B&H Sharpe = **{paper['sharpe']:.2f}**, MDD = **{_fmt_pct(paper['mdd'])}**")
        out.append('')
        out.append('| Source | Sharpe (ours) | Sharpe gap | MDD (ours) | MDD gap |')
        out.append('|---|---:|---:|---:|---:|')
        for src in ('Yahoo ETF', 'Yahoo MF', 'Bloomberg'):
            s = result['per_source_stats'].get(src, {})
            if not s:
                out.append(f'| {src} | — | — | — | — |')
                continue
            sgap = s['sharpe'] - paper['sharpe']
            mgap = s['mdd'] - paper['mdd']
            out.append(
                f"| {src} | {_fmt_num(s['sharpe'])} | {sgap:+.3f} "
                f"| {_fmt_pct(s['mdd'])} | {mgap:+.3%} |"
            )
        out.append('')

    # Pair-wise
    out.append('### Pair-wise divergence (aligned daily returns, 2007-2023)')
    out.append('')
    out.append('| Pair | n | Pearson ρ | Spearman ρ | Ann return drift | Ann tracking error | Mean |Δr| |')
    out.append('|---|---:|---:|---:|---:|---:|---:|')
    for p in result['pair_stats']:
        if p.get('n', 0) < 50:
            out.append(f"| {p['pair']} | {p.get('n', 0)} | — | — | — | — | — |")
            continue
        out.append(
            f"| {p['pair']} | {p['n']:,} | {p['pearson']:.4f} | {p['spearman']:.4f} "
            f"| {p['ann_drift']:+.3%} | {p['te_ann']:.3%} | {p['abs_diff_mean'] * 100:.4f}% |"
        )
    out.append('')

    # Worst-tracking days
    out.append('<details><summary>Top-5 worst tracking days per pair</summary>')
    out.append('')
    for p in result['pair_stats']:
        if not p.get('worst_days'):
            continue
        out.append(f"**{p['pair']}**")
        out.append('')
        out.append('| Date | |a − b| |')
        out.append('|---|---:|')
        for d, v in p['worst_days']:
            out.append(f'| {d} | {abs(v) * 100:.2f}% |')
        out.append('')
    out.append('</details>')
    out.append('')

    # Single 2×2 chart on disk — referenced by relative path so the launcher
    # loads it via st.image() (no base64; no eager browser decode).
    chart_relpath = result.get('chart_relpath')
    if chart_relpath:
        out.append('### Charts')
        out.append('')
        out.append(f'![{asset} charts]({chart_relpath})')
        out.append('')

    out.append('</details>')
    out.append('')
    return '\n'.join(out)


def render_top_summary(results: list) -> str:
    """Two summary tables across all assets: Sharpe-and-correlation, and Return-and-Vol."""
    out = []
    out.append('## Summary: divergence at a glance (2007-2023)')
    out.append('')
    out.append('All gaps are computed against Bloomberg (paper reference).')
    out.append('')

    # ── Table 1: Sharpe + correlation/tracking error ───────────────────────────
    out.append('### Sharpe & alignment')
    out.append('')
    out.append('| Asset | Sharpe BBG | Sharpe Y_ETF (gap) | Sharpe Y_MF (gap) | ρ Y_ETF↔BBG | ρ Y_MF↔BBG | TE Y_ETF | TE Y_MF |')
    out.append('|---|---:|---:|---:|---:|---:|---:|---:|')
    for r in results:
        per = r['per_source_stats']
        bbg = per.get('Bloomberg', {})
        ye = per.get('Yahoo ETF', {})
        ym = per.get('Yahoo MF', {})
        if not bbg:
            out.append(f"| {r['asset_name']} | — | — | — | — | — | — | — |")
            continue
        s_bbg = bbg.get('sharpe', np.nan)

        def _gap_sharpe(src_stats):
            if not src_stats:
                return '—'
            v = src_stats.get('sharpe', np.nan)
            if np.isnan(v):
                return '—'
            return f'{v:.2f} ({v - s_bbg:+.2f})'

        pairs = {p['pair']: p for p in r['pair_stats']}
        pe_bbg = pairs.get('Yahoo ETF vs Bloomberg', {})
        pm_bbg = pairs.get('Yahoo MF vs Bloomberg', {})
        out.append(
            f"| {r['asset_name']} | {s_bbg:.2f} | {_gap_sharpe(ye)} | {_gap_sharpe(ym)} "
            f"| {pe_bbg.get('pearson', float('nan')):.3f} | {pm_bbg.get('pearson', float('nan')):.3f} "
            f"| {pe_bbg.get('te_ann', float('nan')):.2%} | {pm_bbg.get('te_ann', float('nan')):.2%} |"
        )
    out.append('')

    # ── Table 2: Return & Volatility (the MVO inputs μ, Σ) ─────────────────────
    out.append('### Return & Volatility (MVO inputs μ, σ)')
    out.append('')
    out.append('Annualized B&H return and volatility per source. Gaps in **Δ** columns are '
               '`Yahoo − Bloomberg` — large values indicate the joint return/risk distribution '
               'a portfolio optimizer sees differs between sources.')
    out.append('')
    out.append('| Asset | Ret BBG | Ret Y_ETF (Δ) | Ret Y_MF (Δ) | Vol BBG | Vol Y_ETF (Δ) | Vol Y_MF (Δ) |')
    out.append('|---|---:|---:|---:|---:|---:|---:|')
    for r in results:
        per = r['per_source_stats']
        bbg = per.get('Bloomberg', {})
        ye = per.get('Yahoo ETF', {})
        ym = per.get('Yahoo MF', {})
        if not bbg:
            out.append(f"| {r['asset_name']} | — | — | — | — | — | — |")
            continue
        r_bbg, v_bbg = bbg['ann_ret'], bbg['ann_vol']

        def _gap_pct(src_stats, key, ref):
            if not src_stats:
                return '—'
            v = src_stats.get(key, np.nan)
            if np.isnan(v):
                return '—'
            return f'{v * 100:.2f}% ({(v - ref) * 100:+.2f}%)'

        out.append(
            f"| {r['asset_name']} | {r_bbg * 100:.2f}% | {_gap_pct(ye, 'ann_ret', r_bbg)} | {_gap_pct(ym, 'ann_ret', r_bbg)} "
            f"| {v_bbg * 100:.2f}% | {_gap_pct(ye, 'ann_vol', v_bbg)} | {_gap_pct(ym, 'ann_vol', v_bbg)} |"
        )
    out.append('')
    return '\n'.join(out)


def render_report(results: list, scope: str) -> str:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = []
    out.append(f'# Data Source Comparison — {scope}')
    out.append('')
    out.append(f'**Generated:** {ts}')
    out.append(f'**OOS window:** {OOS_START.date()} → {OOS_END.date()}')
    out.append(f'**Sources:** Yahoo ETFs · Yahoo Mutual Funds · Bloomberg Indices (paper reference)')
    out.append('')
    out.append(
        'This report compares the three data sources used by the strategy. '
        'For each paper-asset class (LargeCap, REIT, …) we hold three ticker proxies and '
        'compare their daily Total-Return series on the paper OOS window. '
        '**Stats are computed on raw B&H returns** (not the JM-XGB strategy) — '
        'so any difference is purely *data*, not algorithm.'
    )
    out.append('')
    out.append(
        '**How to read the numbers:** *Sharpe gap vs paper* tells you whether the source\'s '
        'B&H return series matches the paper. *Pair-wise Pearson ρ* measures how similar the '
        'sources move day-to-day. *Tracking error* (annualized stdev of daily return differences) '
        'shows how much the sources drift apart over time.'
    )
    out.append('')

    # One-look bar chart of return/vol divergence across all assets.
    # Eagerly loaded — small (2 panels, ~80KB) and high-signal.
    if len(results) >= 2:
        summary_chart = chart_summary_divergence(results)
        if summary_chart:
            out.append('### Return & Volatility divergence — all assets')
            out.append('')
            out.append(f'![divergence summary]({summary_chart})')
            out.append('')

    out.append(render_top_summary(results))

    for r in results:
        out.append(render_asset_section(r))

    out.append('---')
    out.append('')
    out.append(
        f'Cache files: `cache/data_comparison_<asset>.pkl`. '
        f'Re-run skips recomputation if source per-ticker caches are unchanged.'
    )
    return '\n'.join(out)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_selection(args):
    if not args:
        return 'all', TIERS['all'], 'All 12 assets'
    arg = ' '.join(args).strip().lower()
    if arg in ('list', 'ls', '-h', '--help', 'help'):
        print('Available scopes:')
        print('  (no arg)       all 12 assets')
        print('  tier1          LargeCap')
        print('  tier2          LargeCap, REIT, AggBond')
        print('  all            all 12 assets')
        print('  <asset_name>   any one of: ' + ', '.join(ASSET_REGISTRY.keys()))
        return None, None, None
    if arg in TIERS:
        names = TIERS[arg]
        return arg, names, 'Tier 1 (LargeCap)' if arg == 'tier1' else (
            'Tier 2 (LargeCap, REIT, AggBond)' if arg == 'tier2' else 'All 12 assets'
        )
    # try exact name match (case-insensitive)
    for name in ASSET_REGISTRY:
        if name.lower() == arg:
            return name.lower(), [name], name
    print(f"Error: scope {arg!r} not recognized.")
    print(f"Run with 'list' to see options.")
    return None, None, None


def main():
    args = sys.argv[1:]
    use_cache = '--no-cache' not in args
    args = [a for a in args if a != '--no-cache']

    scope_key, assets, scope_label = parse_selection(args)
    if scope_key is None:
        return 0 if (args and args[0] in ('list', '-h', '--help', 'help')) else 1

    print(f'Comparing data sources for: {scope_label}')
    print(f'OOS window: {OOS_START.date()} → {OOS_END.date()}')
    print(f'Cache:      {"enabled (incremental)" if use_cache else "disabled (force recompute)"}')
    print()

    t0 = time.time()
    results = []
    for asset in assets:
        ts0 = time.time()
        sources = ASSET_REGISTRY[asset]
        print(f'[{asset}] {sources["Yahoo ETF"]} / {sources["Yahoo MF"]} / {sources["Bloomberg"]} ', end='', flush=True)
        try:
            res = analyze_asset(asset, use_cache=use_cache)
            results.append(res)
            if res['missing']:
                print(f'  (missing: {", ".join(s for s, _ in res["missing"])})  [{time.time() - ts0:.1f}s]')
            else:
                bbg = res['per_source_stats'].get('Bloomberg', {})
                ye = res['per_source_stats'].get('Yahoo ETF', {})
                ym = res['per_source_stats'].get('Yahoo MF', {})
                msg = f"Sharpe BBG={bbg.get('sharpe', float('nan')):.2f}"
                if ye: msg += f"  Y_ETF={ye.get('sharpe', float('nan')):.2f}"
                if ym: msg += f"  Y_MF={ym.get('sharpe', float('nan')):.2f}"
                print(f'  {msg}  [{time.time() - ts0:.1f}s]')
        except Exception as e:
            print(f'  FAILED: {e}')
            raise

    md = render_report(results, scope_label)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_name = f'data_comparison_{scope_key}_{ts}.md'
    out_path = os.path.join(BENCHMARKS_DIR, out_name)
    with open(out_path, 'w') as f:
        f.write(md)
    print()
    print(f'Total time: {time.time() - t0:.1f}s')
    print(f'Report:     {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
