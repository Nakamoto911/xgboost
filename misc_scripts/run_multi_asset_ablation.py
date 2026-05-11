"""
Multi-Asset Macro Ablation
==========================
For each ticker in the "Yahoo ETFs" asset list, runs a full walk-forward
backtest under two configurations:

  Pass A — feature_ablation="all"          (full JM-XGB: return + macro features)
  Pass B — feature_ablation="return_only"  (JM-XGB excluding macro features)

Reuses the data fetching, JM, walk-forward, and parallel-pool machinery from
benchmark_assets.py. Honours env-var overrides set by the Diagnostics Launcher
(OOS dates, lambda grid, transaction cost, XGB params, WF strategy options).

Output: timestamped MD report in benchmarks/ with a Consolidated Comparison
Table (Asset, Config, Ann. Return, Vol, Sharpe, Sortino, Max DD).
"""
import os
import sys
import time
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import benchmark_assets as ba  # noqa: E402  (env-driven constants resolved at import)
from config import StrategyConfig  # noqa: E402

warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────────

ASSET_LIST_NAME = 'Yahoo ETFs'

ABLATION_PASSES = [
    ('Full',        'all'),
    ('Return-Only', 'return_only'),
]

# Bloomberg paper-aligned assets (DATA PAUL.xlsx). Long history → no inception
# truncation issues. Schema mirrors test_bbg_assets.ASSET_CONFIGS.
BBG_XLSX_PATH = os.path.join(PROJECT_ROOT, 'cache', 'DATA PAUL.xlsx')
BBG_ASSETS = {
    'LargeCap':  {'col': 'SPTR',     'dd': True,  'hl_proxy': '^SP500TR'},
    'MidCap':    {'col': 'SPTRMDCP', 'dd': True,  'hl_proxy': '^SP500TR'},
    'SmallCap':  {'col': 'RU20INTR', 'dd': True,  'hl_proxy': '^SP500TR'},
    'EAFE':      {'col': 'NDDUEAFE', 'dd': True,  'hl_proxy': 'EFA'},
    'EM':        {'col': 'NDUEEGF',  'dd': True,  'hl_proxy': 'EEM'},
    'REIT':      {'col': 'DJUSRET',  'dd': True,  'hl_proxy': '^SP500TR'},
    'AggBond':   {'col': 'LBUSTRUU', 'dd': False, 'hl_proxy': '^SP500TR'},
    'Treasury':  {'col': 'LUTLTRUU', 'dd': False, 'hl_proxy': '^SP500TR'},
    'HighYield': {'col': 'IBOXHY',   'dd': True,  'hl_proxy': 'HYG'},
    'Corporate': {'col': 'LUACTRUU', 'dd': False, 'hl_proxy': 'SPBO'},
    'Commodity': {'col': 'DBLCDBCE', 'dd': True,  'hl_proxy': 'DBC'},
    'Gold':      {'col': 'GOLDLNPM', 'dd': False, 'hl_proxy': 'GLD'},
}

def _env(key, default=None):
    """Return env var value, treating missing/empty/whitespace as fallback to default."""
    raw = os.environ.get(key, '')
    return raw.strip() if raw and raw.strip() else default


_yesterday = (pd.Timestamp.today() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
OOS_START_DATE = _env('XGB_OOS_START_DATE', '2007-01-01')
OOS_END_DATE = _env('XGB_END_DATE', _yesterday)
DATA_START_OVERRIDE = _env('XGB_START_DATE_DATA')  # None if missing → falls back to asset_lists.md
VALIDATION_WINDOW_YRS = int(_env('XGB_VALIDATION_WINDOW_YRS', '5'))
DATA_SOURCE = (_env('XGB_DATA_SOURCE', 'yahoo') or 'yahoo').lower()  # 'yahoo' | 'bloomberg'

BENCHMARKS_DIR = os.path.join(PROJECT_ROOT, 'benchmarks')
os.makedirs(BENCHMARKS_DIR, exist_ok=True)


def build_config(name, ablation):
    """Build a StrategyConfig for one ablation pass, applying env-var overrides."""
    kwargs = {'name': name, 'feature_ablation': ablation}
    env_map = {
        'XGB_TUNING_METRIC': ('tuning_metric', str),
        'XGB_VALIDATION_WINDOW_TYPE': ('validation_window_type', str),
        'XGB_LAMBDA_SMOOTHING': ('lambda_smoothing', lambda v: v.lower() == 'true'),
        'XGB_PROB_THRESHOLD': ('prob_threshold', float),
        'XGB_ALLOCATION_STYLE': ('allocation_style', str),
        'XGB_LAMBDA_ENSEMBLE_K': ('lambda_ensemble_k', int),
        'XGB_LAMBDA_SELECTION': ('lambda_selection', str),
        'XGB_LAMBDA_SUBWINDOW_CONSENSUS': ('lambda_subwindow_consensus', lambda v: v.lower() == 'true'),
        'XGB_EWMA_MODE': ('ewma_mode', str),
        'XGB_EXECUTION_MODE': ('execution_mode', str),
    }
    for env_key, (field_name, cast) in env_map.items():
        val = _env(env_key)
        if val is not None:
            kwargs[field_name] = cast(val)
    return StrategyConfig(**kwargs)


# ── Bloomberg loader ──────────────────────────────────────────────────────────

def _bbg_series(df_yf, name):
    if df_yf is None or df_yf.empty:
        return None
    cols = df_yf.columns
    if isinstance(cols, pd.MultiIndex):
        s = df_yf['Adj Close'].iloc[:, 0] if 'Adj Close' in cols.get_level_values(0) else df_yf.iloc[:, 0]
    else:
        s = df_yf.get('Adj Close', df_yf.get('Close', df_yf.iloc[:, 0]))
    return s.rename(name)


def _load_bloomberg_inputs():
    """Load DATA PAUL.xlsx + FRED yields + ^VIX + ^IRX. Returns (raw, fred, vix, irx)."""
    import yfinance as yf
    import main as backend  # main.py owns the FRED loader + cache
    if not os.path.exists(BBG_XLSX_PATH):
        raise FileNotFoundError(
            f"Bloomberg data file not found: {BBG_XLSX_PATH}\n"
            "Place 'DATA PAUL.xlsx' under cache/ to run with --source bloomberg."
        )
    raw = pd.read_excel(BBG_XLSX_PATH, header=None, skiprows=6)
    raw.columns = ['Date', 'SPTR', 'SPTRMDCP', 'RU20INTR', 'NDDUEAFE', 'NDUEEGF',
                   'LBUSTRUU', 'IBOXHY', 'LUACTRUU', 'DJUSRET', 'DBLCDBCE',
                   'GOLDLNPM', 'LUTLTRUU']
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw = raw.set_index('Date').sort_index()

    fred = backend._fetch_fred_data().ffill().dropna()
    fetch_end = (pd.Timestamp.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    vix = _bbg_series(yf.download('^VIX', start='1987-01-01', end=fetch_end,
                                  auto_adjust=False, progress=False), 'VIX')
    irx = _bbg_series(yf.download('^IRX', start='1987-01-01', end=fetch_end,
                                  auto_adjust=False, progress=False), 'IRX')
    if vix is None or irx is None:
        raise RuntimeError("Failed to fetch ^VIX/^IRX from Yahoo (needed for Bloomberg pipeline).")
    return raw, fred, vix, irx


def _build_bloomberg_features(raw, fred, vix, irx, target_col, include_dd):
    """Reproduce benchmark_assets.fetch_etf_data feature shape from Bloomberg series."""
    cols = list(dict.fromkeys([target_col, 'SPTR', 'LBUSTRUU']))  # dedup; need SPTR+LBUSTRUU for Stock_Bond_Corr
    df = (raw[cols]
          .join(fred, how='inner')
          .join(vix, how='inner')
          .join(irx, how='inner')
          .ffill().dropna())

    f = pd.DataFrame(index=df.index)
    tr = df[target_col].pct_change().fillna(0)
    f['Target_Return'] = tr
    f['RF_Rate'] = (df['IRX'] / 100) / 252
    f['Excess_Return'] = tr - f['RF_Rate']

    dn = np.minimum(f['Excess_Return'], 0)
    if include_dd:
        for hl in [5, 21]:
            ewm_dd = np.sqrt((dn ** 2).ewm(halflife=hl).mean()).fillna(0)
            f[f'DD_log_{hl}'] = np.log(ewm_dd + 1e-8)
    for hl in [5, 10, 21]:
        f[f'Avg_Ret_{hl}'] = f['Excess_Return'].ewm(halflife=hl).mean()
    for hl in [5, 10, 21]:
        dd_r = np.maximum(np.sqrt((dn ** 2).ewm(halflife=hl).mean()).fillna(1e-8), 1e-8)
        f[f'Sortino_{hl}'] = (f[f'Avg_Ret_{hl}'] / dd_r).clip(-10, 10)

    f['Yield_2Y_EWMA_diff'] = df['DGS2'].diff().fillna(0).ewm(halflife=21).mean()
    sl = df['DGS10'] - df['DGS2']
    f['Yield_Slope_EWMA_10'] = sl.ewm(halflife=10).mean()
    f['Yield_Slope_EWMA_diff_21'] = sl.diff().fillna(0).ewm(halflife=21).mean()
    f['VIX_EWMA_log_diff'] = np.log(df['VIX'] / df['VIX'].shift(1)).fillna(0).ewm(halflife=63).mean()
    sptr_ret = df['SPTR'].pct_change().fillna(0)
    bond_ret = df['LBUSTRUU'].pct_change().fillna(0)
    f['Stock_Bond_Corr'] = sptr_ret.rolling(252).corr(bond_ret).fillna(0)
    return f.dropna()


def load_bloomberg_assets():
    """Return (asset_data, paper_hl_map, tickers) for the 12 paper-aligned BBG assets."""
    raw, fred, vix, irx = _load_bloomberg_inputs()
    asset_data = {}
    paper_hl_map = {}
    for name, info in BBG_ASSETS.items():
        feats = _build_bloomberg_features(raw, fred, vix, irx, info['col'], info['dd'])
        if len(feats) < 252 * 5:
            print(f"  [WARN] {name}: only {len(feats)} rows after feature build, skipping")
            continue
        asset_data[name] = feats
        paper_hl_map[name] = ba.PAPER_EWMA_HL.get(info['hl_proxy'], 8)
    return asset_data, paper_hl_map, list(asset_data.keys())


# ── Walk-forward backtest (single asset, two configs, full OOS window) ────────

def _tune_initial_ewma_hl(df, ticker, config, oos_start_dt, data_start, cache, paper_hl_map):
    if config.ewma_mode == "paper" and ticker in paper_hl_map:
        return paper_hl_map[ticker]

    if config.validation_window_type == 'expanding':
        init_val_start = pd.to_datetime(data_start)
    else:
        init_val_start = oos_start_dt - pd.DateOffset(years=VALIDATION_WINDOW_YRS)

    best_hl = ba.EWMA_HL_GRID[-1]
    best_metric = -np.inf
    for hl in ba.EWMA_HL_GRID:
        for lmbda in ba.LAMBDA_GRID:
            val_res = ba.simulate_strategy_fast(
                df, init_val_start, oos_start_dt, lmbda, cache, config,
                include_xgboost=True, ewma_halflife=hl,
            )
            if val_res.empty:
                continue
            _, _, sh, so, _ = ba.calculate_metrics(val_res['Strat_Return'], val_res['RF_Rate'])
            metric = so if config.tuning_metric == 'sortino' else sh
            if not np.isnan(metric) and metric > best_metric:
                best_metric = metric
                best_hl = hl
    return best_hl


def _select_lambda_subwindow_consensus(df, val_start, current_date, config, ewma_hl, cache):
    val_duration = (current_date - val_start).days
    sub_boundaries = [
        (val_start, val_start + pd.DateOffset(days=val_duration // 2)),
        (val_start + pd.DateOffset(days=val_duration // 4),
         val_start + pd.DateOffset(days=3 * val_duration // 4)),
        (val_start + pd.DateOffset(days=val_duration // 2), current_date),
    ]
    sub_scores = [[] for _ in sub_boundaries]
    for lmbda in ba.LAMBDA_GRID:
        val_res = ba.simulate_strategy_fast(
            df, val_start, current_date, lmbda, cache, config,
            include_xgboost=True, ewma_halflife=ewma_hl,
        )
        if val_res.empty:
            continue
        for sw_idx, (sw_s, sw_e) in enumerate(sub_boundaries):
            sw = val_res[(val_res.index >= sw_s) & (val_res.index < sw_e)]
            if len(sw) >= 60:
                _, _, sh, so, _ = ba.calculate_metrics(sw['Strat_Return'], sw['RF_Rate'])
                metric = so if config.tuning_metric == 'sortino' else sh
                if not np.isnan(metric):
                    sub_scores[sw_idx].append((metric, lmbda))
    consensus = []
    for sw in sub_scores:
        if sw:
            sw.sort(key=lambda x: x[0], reverse=True)
            consensus.append(sw[0][1])
    if consensus:
        return float(np.median(consensus))
    return ba.LAMBDA_GRID[len(ba.LAMBDA_GRID) // 2]


def _select_lambda_argmax(df, val_start, current_date, config, ewma_hl, cache):
    scores = []
    for lmbda in ba.LAMBDA_GRID:
        val_res = ba.simulate_strategy_fast(
            df, val_start, current_date, lmbda, cache, config,
            include_xgboost=True, ewma_halflife=ewma_hl,
        )
        if val_res.empty:
            continue
        _, _, sh, so, _ = ba.calculate_metrics(val_res['Strat_Return'], val_res['RF_Rate'])
        metric = so if config.tuning_metric == 'sortino' else sh
        if not np.isnan(metric):
            scores.append((metric, lmbda))
    scores.sort(key=lambda x: x[0], reverse=True)
    if not scores:
        return ba.LAMBDA_GRID[len(ba.LAMBDA_GRID) // 2], []
    if config.lambda_selection == 'median_positive':
        positive = [l for s, l in scores if s > 0]
        best = float(np.median(positive)) if positive else scores[0][1]
    else:
        best = scores[0][1]
    top_k = [l for _, l in scores[:max(1, config.lambda_ensemble_k)]]
    return best, top_k


def _run_single_pass(df, ticker, config, data_start, cache, paper_hl_map):
    """Run full walk-forward backtest for one (ticker, config) pair."""
    oos_start_dt = pd.to_datetime(OOS_START_DATE)
    oos_end_dt = pd.to_datetime(OOS_END_DATE)

    # Need enough history before OOS start for the JM lookback
    if df.index[0] > oos_start_dt - pd.DateOffset(years=3):
        return None, None

    best_ewma_hl = _tune_initial_ewma_hl(df, ticker, config, oos_start_dt, data_start, cache, paper_hl_map)

    current_date = oos_start_dt
    chunks = []
    lambda_history = []
    while current_date < oos_end_dt:
        chunk_end = min(current_date + pd.DateOffset(months=6), oos_end_dt)
        if config.validation_window_type == 'expanding':
            val_start = pd.to_datetime(data_start)
        else:
            val_start = current_date - pd.DateOffset(years=VALIDATION_WINDOW_YRS)

        if config.lambda_subwindow_consensus:
            best_lambda = _select_lambda_subwindow_consensus(
                df, val_start, current_date, config, best_ewma_hl, cache,
            )
            best_lambdas = [best_lambda]
        else:
            best_lambda, best_lambdas = _select_lambda_argmax(
                df, val_start, current_date, config, best_ewma_hl, cache,
            )

        if config.lambda_smoothing and lambda_history:
            best_lambda = (0.7 * best_lambda) + (0.3 * lambda_history[-1])
        lambda_history.append(best_lambda)

        if config.lambda_ensemble_k > 1 and len(best_lambdas) > 1:
            oos_chunks = []
            for l_val in best_lambdas:
                c = ba.run_period_forecast_fast(df, current_date, l_val, cache, config, include_xgboost=True)
                if c is not None:
                    oos_chunks.append(c)
            if oos_chunks:
                avg_prob = sum(c['Raw_Prob'] for c in oos_chunks) / len(oos_chunks)
                final = oos_chunks[0].copy()
                final['Raw_Prob'] = avg_prob
                chunks.append(final)
        else:
            c = ba.run_period_forecast_fast(df, current_date, best_lambda, cache, config, include_xgboost=True)
            if c is not None:
                chunks.append(c)

        current_date = chunk_end

    if not chunks:
        return None, best_ewma_hl

    full_df = pd.concat(chunks)
    if best_ewma_hl == 0:
        full_df['State_Prob'] = full_df['Raw_Prob']
    else:
        full_df['State_Prob'] = full_df['Raw_Prob'].ewm(halflife=best_ewma_hl).mean()

    if config.allocation_style == "binary":
        full_df['Forecast_State'] = (full_df['State_Prob'] > config.prob_threshold).astype(int)
        signals = full_df['Forecast_State'].shift(1).fillna(0)
        alloc = 1.0 - signals
    else:
        alloc = (1.0 - full_df['State_Prob']).shift(1).fillna(1.0)

    rets = (alloc * full_df['Target_Return']) + ((1.0 - alloc) * full_df['RF_Rate'])
    trades = alloc.diff().abs().fillna(0)
    full_df['Strat_Return'] = rets - (trades * ba.TRANSACTION_COST)

    mask = (full_df.index >= oos_start_dt) & (full_df.index < oos_end_dt)
    oos_only = full_df[mask]
    if oos_only.empty:
        return None, best_ewma_hl

    return oos_only, best_ewma_hl


def backtest_single_asset_ablation(args):
    """Worker entrypoint for the multiprocessing pool."""
    ticker, df, configs, data_start, paper_hl_map = args
    cache = {}
    rows = []
    asset_t0 = time.time()
    print(f"  [{ticker}] starting ({len(df)} rows, "
          f"{df.index[0].date()} → {df.index[-1].date()})", flush=True)
    for config in configs:
        pass_t0 = time.time()
        oos_only, ewma_hl = _run_single_pass(df, ticker, config, data_start, cache, paper_hl_map)
        pass_dt = time.time() - pass_t0
        if oos_only is None:
            print(f"  [{ticker}] pass={config.name:<12} {pass_dt:6.1f}s  → no data", flush=True)
            rows.append({
                'Ticker': ticker, 'Config': config.name,
                'Ann_Ret': np.nan, 'Ann_Vol': np.nan,
                'Sharpe': np.nan, 'Sortino': np.nan, 'Max_DD': np.nan,
                'EWMA_HL': ewma_hl,
            })
            continue
        ann_ret, ann_vol, sharpe, sortino, mdd = ba.calculate_metrics(
            oos_only['Strat_Return'], oos_only['RF_Rate'],
        )
        print(f"  [{ticker}] pass={config.name:<12} {pass_dt:6.1f}s  "
              f"Sharpe={sharpe:+.3f}  hl={ewma_hl}", flush=True)
        rows.append({
            'Ticker': ticker, 'Config': config.name,
            'Ann_Ret': ann_ret, 'Ann_Vol': ann_vol,
            'Sharpe': sharpe, 'Sortino': sortino, 'Max_DD': mdd,
            'EWMA_HL': ewma_hl,
        })
    print(f"  [{ticker}] done in {time.time() - asset_t0:.1f}s", flush=True)
    return rows


# ── Markdown report ───────────────────────────────────────────────────────────

def _fmt_pct(x, digits=2):
    return f"{x*100:.{digits}f}%" if pd.notna(x) else "N/A"


def _fmt_num(x, digits=3):
    return f"{x:.{digits}f}" if pd.notna(x) else "N/A"


def generate_markdown_report(results_df, asset_data, tickers, elapsed, timestamp,
                             configs, data_start, source_label, list_label):
    lines = []
    lines.append("# Multi-Asset Macro Ablation Report")
    lines.append("")
    lines.append(f"**Generated:** {timestamp}")
    lines.append(f"**Runtime:** {elapsed:.1f}s")
    lines.append(f"**Data source:** {source_label}")
    lines.append(f"**Asset list:** {list_label} ({len(tickers)} tickers)")
    lines.append(f"**OOS window:** {OOS_START_DATE} → {OOS_END_DATE}")
    lines.append(f"**Lambda grid:** {ba.LAMBDA_GRID}")
    lines.append(f"**Transaction cost:** {ba.TRANSACTION_COST*10000:.1f} bps")
    lines.append("")
    lines.append("**Pass A — `Full`:** `feature_ablation=\"all\"` — return + macro features.")
    lines.append("**Pass B — `Return-Only`:** `feature_ablation=\"return_only\"` — return features only "
                 "(macro features dropped from XGBoost input).")
    lines.append("")

    # Consolidated Comparison Table
    lines.append("## Consolidated Comparison")
    lines.append("")
    lines.append("| Asset | Config | Ann. Return | Vol | Sharpe | Sortino | Max DD |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for ticker in tickers:
        for cfg in configs:
            row = results_df[(results_df['Ticker'] == ticker) & (results_df['Config'] == cfg.name)]
            if row.empty:
                lines.append(f"| {ticker} | {cfg.name} | N/A | N/A | N/A | N/A | N/A |")
                continue
            r = row.iloc[0]
            lines.append(
                f"| {ticker} | {cfg.name} | {_fmt_pct(r['Ann_Ret'])} | {_fmt_pct(r['Ann_Vol'])} | "
                f"{_fmt_num(r['Sharpe'])} | {_fmt_num(r['Sortino'])} | {_fmt_pct(r['Max_DD'])} |"
            )
    lines.append("")

    # Per-config averages
    lines.append("### Aggregate Means")
    lines.append("")
    lines.append("| Config | Avg Ann. Return | Avg Vol | Avg Sharpe | Avg Sortino | Avg Max DD |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for cfg in configs:
        sub = results_df[results_df['Config'] == cfg.name].dropna(subset=['Sharpe'])
        if sub.empty:
            lines.append(f"| {cfg.name} | N/A | N/A | N/A | N/A | N/A |")
            continue
        lines.append(
            f"| {cfg.name} | {_fmt_pct(sub['Ann_Ret'].mean())} | {_fmt_pct(sub['Ann_Vol'].mean())} | "
            f"{_fmt_num(sub['Sharpe'].mean())} | {_fmt_num(sub['Sortino'].mean())} | "
            f"{_fmt_pct(sub['Max_DD'].mean())} |"
        )
    lines.append("")

    # Sharpe delta summary (Full − Return-Only)
    if len(configs) >= 2:
        full_name, ro_name = configs[0].name, configs[1].name
        lines.append(f"## Macro Feature Impact (Sharpe Delta: `{full_name}` − `{ro_name}`)")
        lines.append("")
        lines.append("| Asset | Full Sharpe | Return-Only Sharpe | Delta | Verdict |")
        lines.append("|---|---:|---:|---:|---|")
        deltas = []
        for ticker in tickers:
            f_row = results_df[(results_df['Ticker'] == ticker) & (results_df['Config'] == full_name)]
            r_row = results_df[(results_df['Ticker'] == ticker) & (results_df['Config'] == ro_name)]
            if (f_row.empty or r_row.empty
                    or pd.isna(f_row.iloc[0]['Sharpe']) or pd.isna(r_row.iloc[0]['Sharpe'])):
                lines.append(f"| {ticker} | N/A | N/A | N/A | N/A |")
                continue
            fs, rs = f_row.iloc[0]['Sharpe'], r_row.iloc[0]['Sharpe']
            delta = fs - rs
            deltas.append(delta)
            if delta > 0.02:
                verdict = "Macro helps"
            elif delta < -0.02:
                verdict = "Macro hurts"
            else:
                verdict = "Neutral"
            lines.append(f"| {ticker} | {fs:.3f} | {rs:.3f} | {delta:+.3f} | {verdict} |")
        if deltas:
            avg = np.mean(deltas)
            helps = sum(1 for d in deltas if d > 0.02)
            hurts = sum(1 for d in deltas if d < -0.02)
            neutral = len(deltas) - helps - hurts
            lines.append(
                f"| **AVG** | — | — | **{avg:+.3f}** | "
                f"**Helps {helps}, Hurts {hurts}, Neutral {neutral}** |"
            )
        lines.append("")

    # Data coverage
    lines.append("## Data Coverage")
    lines.append("")
    lines.append("| Ticker | Start | End | Rows | EWMA HL (Full) |")
    lines.append("|---|---|---|---:|---:|")
    full_name = configs[0].name
    for t in tickers:
        if t in asset_data:
            df_t = asset_data[t]
            row = results_df[(results_df['Ticker'] == t) & (results_df['Config'] == full_name)]
            hl = row.iloc[0].get('EWMA_HL') if not row.empty else None
            hl_str = f"{int(hl)}" if pd.notna(hl) else "N/A"
            lines.append(f"| {t} | {df_t.index[0].date()} | {df_t.index[-1].date()} | {len(df_t)} | {hl_str} |")
        else:
            lines.append(f"| {t} | N/A | N/A | N/A | N/A |")
    lines.append("")

    # Configuration
    cfg = configs[0]  # both configs share non-ablation parameters
    lines.append("## Strategy Configuration (shared)")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| Data source | {source_label} |")
    lines.append(f"| Asset list | {list_label} |")
    lines.append(f"| Data start | {data_start} |")
    lines.append(f"| OOS window | {OOS_START_DATE} → {OOS_END_DATE} |")
    lines.append(f"| Validation window | {cfg.validation_window_type} ({VALIDATION_WINDOW_YRS}y) |")
    lines.append(f"| Tuning metric | {cfg.tuning_metric} |")
    lines.append(f"| Allocation style | {cfg.allocation_style} |")
    lines.append(f"| Prob threshold | {cfg.prob_threshold} |")
    lines.append(f"| Lambda smoothing | {cfg.lambda_smoothing} |")
    lines.append(f"| Lambda ensemble K | {cfg.lambda_ensemble_k} |")
    lines.append(f"| Lambda selection | {cfg.lambda_selection} |")
    lines.append(f"| Sub-window consensus | {cfg.lambda_subwindow_consensus} |")
    lines.append(f"| EWMA mode | {cfg.ewma_mode} |")
    lines.append(f"| Lambda grid | {ba.LAMBDA_GRID} |")
    lines.append(f"| Transaction cost | {ba.TRANSACTION_COST:.4f} ({ba.TRANSACTION_COST*10000:.1f} bps) |")
    xgb_params = cfg.xgb_params or {}
    lines.append("")
    lines.append("### XGBoost Hyperparameters")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    for key in ('max_depth', 'n_estimators', 'learning_rate', 'subsample',
                'colsample_bytree', 'reg_alpha', 'reg_lambda'):
        lines.append(f"| {key} | {xgb_params.get(key)} |")
    lines.append("")

    return '\n'.join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Validate OOS window — pd.to_datetime('') silently returns NaT, which would
    # cause every WF loop to exit immediately and produce all-N/A results.
    oos_start_ts = pd.to_datetime(OOS_START_DATE, errors='coerce')
    oos_end_ts = pd.to_datetime(OOS_END_DATE, errors='coerce')
    if pd.isna(oos_start_ts) or pd.isna(oos_end_ts) or oos_start_ts >= oos_end_ts:
        print(f"ERROR: invalid OOS window [{OOS_START_DATE!r} → {OOS_END_DATE!r}].")
        print("If launched from the Diagnostics page, ensure 'OOS Start Date' and "
              "'End Date' are set in the sidebar (XGB_OOS_START_DATE / XGB_END_DATE).")
        sys.exit(2)

    t0 = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Resolve data source
    if DATA_SOURCE == 'bloomberg':
        source_label = "Bloomberg (DATA PAUL.xlsx)"
        list_label = "Bloomberg Paper Assets"
        data_start = DATA_START_OVERRIDE or '1987-01-01'
        print("=" * 80)
        print(f"  MULTI-ASSET MACRO ABLATION  [{list_label}]")
        print("=" * 80)
        print(f"Data source: {source_label}")
        print(f"OOS window: {OOS_START_DATE} → {OOS_END_DATE}")
        print(f"Data start: {data_start}")
        print(f"Lambda grid: {ba.LAMBDA_GRID}")
        print(f"Validation window: {VALIDATION_WINDOW_YRS} years")
        print(f"Passes: {[name for name, _ in ABLATION_PASSES]}\n")
        print(f"Loading 12 Bloomberg paper-aligned assets from {BBG_XLSX_PATH} ...")
        try:
            asset_data, paper_hl_map, tickers = load_bloomberg_assets()
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(3)
        # Trim each asset to OOS-end so backtests don't reference future bars
        for tk, df in list(asset_data.items()):
            asset_data[tk] = df[df.index <= oos_end_ts]
            print(f"  {tk:<10} OK ({len(asset_data[tk])} rows, "
                  f"{asset_data[tk].index[0].date()} → {asset_data[tk].index[-1].date()})")
    else:
        list_label = ASSET_LIST_NAME
        source_label = f"Yahoo Finance ({ASSET_LIST_NAME})"
        all_lists = ba.load_asset_lists()
        if ASSET_LIST_NAME not in all_lists:
            print(f"Error: '{ASSET_LIST_NAME}' not found in asset_lists.md")
            print("Available lists:", ', '.join(f"'{n}'" for n in all_lists))
            sys.exit(1)
        selected = all_lists[ASSET_LIST_NAME]
        tickers = selected['tickers']
        data_start = DATA_START_OVERRIDE or selected['data_start'] or '1993-01-01'
        paper_hl_map = ba.PAPER_EWMA_HL

        print("=" * 80)
        print(f"  MULTI-ASSET MACRO ABLATION  [{list_label}]")
        print("=" * 80)
        print(f"Data source: {source_label}")
        print(f"OOS window: {OOS_START_DATE} → {OOS_END_DATE}")
        print(f"Data start: {data_start}")
        print(f"Lambda grid: {ba.LAMBDA_GRID}")
        print(f"Validation window: {VALIDATION_WINDOW_YRS} years")
        print(f"Passes: {[name for name, _ in ABLATION_PASSES]}\n")
        print(f"Fetching data for {len(tickers)} assets...")
        asset_data = {}
        for ticker in tickers:
            print(f"  Fetching {ticker}...", end=" ", flush=True)
            _, df = ba.fetch_etf_data(ticker, data_start=data_start)
            if df is not None:
                asset_data[ticker] = df
                print(f"OK ({len(df)} rows, {df.index[0].date()} → {df.index[-1].date()})")
            else:
                print("FAILED")

    print(f"\nLoaded {len(asset_data)}/{len(tickers)} assets")
    if not asset_data:
        print("No data available. Exiting.")
        return

    # 2. Build configs (shared across assets, env-overridden once)
    configs = [build_config(name, ablation) for name, ablation in ABLATION_PASSES]

    # 3. Run backtests in parallel across assets
    print(f"\nRunning ablation backtests "
          f"({len(configs)} passes × {len(asset_data)} assets) ...")
    args_list = [(ticker, df, configs, data_start, paper_hl_map)
                 for ticker, df in asset_data.items()]
    n_workers = min(cpu_count(), len(args_list))
    print(f"Using {n_workers} parallel workers\n")

    all_rows = []
    if n_workers > 1:
        with Pool(n_workers) as pool:
            for i, rows in enumerate(pool.imap_unordered(backtest_single_asset_ablation, args_list)):
                ticker = rows[0]['Ticker'] if rows else '?'
                print(f"  Completed {ticker} ({i+1}/{len(args_list)})")
                all_rows.extend(rows)
    else:
        for args in args_list:
            rows = backtest_single_asset_ablation(args)
            ticker = rows[0]['Ticker'] if rows else '?'
            print(f"  Completed {ticker}")
            all_rows.extend(rows)

    elapsed = time.time() - t0
    print(f"\nBacktests completed in {elapsed:.1f}s")

    # 4. Persist results
    results_df = pd.DataFrame(all_rows)
    csv_path = os.path.join(BENCHMARKS_DIR, f'macro_ablation_results_{timestamp}.csv')
    results_df.to_csv(csv_path, index=False)

    md_content = generate_markdown_report(
        results_df, asset_data, tickers, elapsed, timestamp, configs, data_start,
        source_label, list_label,
    )
    md_path = os.path.join(BENCHMARKS_DIR, f'macro_ablation_report_{timestamp}.md')
    with open(md_path, 'w') as f:
        f.write(md_content)

    # 5. Console summary
    print("\n" + "=" * 100)
    print("  CONSOLIDATED COMPARISON")
    print("=" * 100)
    print(f"  {'Ticker':<8} │ {'Config':<14} │ {'Ann Ret':>8} │ {'Vol':>7} │ "
          f"{'Sharpe':>7} │ {'Sortino':>8} │ {'Max DD':>8} │ {'EWMA HL':>7}")
    print(f"  {'─'*8}─┼─{'─'*14}─┼─{'─'*8}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*7}")
    for ticker in tickers:
        for cfg in configs:
            row = results_df[(results_df['Ticker'] == ticker) & (results_df['Config'] == cfg.name)]
            if row.empty:
                continue
            r = row.iloc[0]
            hl = r.get('EWMA_HL')
            hl_str = f"{int(hl)}" if pd.notna(hl) else "—"
            if pd.isna(r['Sharpe']):
                print(f"  {ticker:<8} │ {cfg.name:<14} │ {'N/A':>8} │ {'N/A':>7} │ "
                      f"{'N/A':>7} │ {'N/A':>8} │ {'N/A':>8} │ {hl_str:>7}")
            else:
                print(f"  {ticker:<8} │ {cfg.name:<14} │ {r['Ann_Ret']*100:>7.1f}% │ "
                      f"{r['Ann_Vol']*100:>6.1f}% │ {r['Sharpe']:>7.2f} │ "
                      f"{r['Sortino']:>8.2f} │ {r['Max_DD']*100:>7.1f}% │ {hl_str:>7}")

    print(f"\nOutputs:")
    print(f"  CSV:    {csv_path}")
    print(f"  Report: {md_path}")
    print(f"Total runtime: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
