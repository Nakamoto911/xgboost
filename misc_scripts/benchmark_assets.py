"""
Multi-Asset Benchmark: JM-XGB Strategy vs Buy & Hold
Tests across 12 ETFs and multiple time periods.

Optimizations for speed (methodology preserved):
- Skip SHAP computation (not needed for performance evaluation)
- Reduced lambda grid (5 candidates instead of 11)
- Full EWMA halflife grid (4 candidates)
- No PDF/chart generation
- Multiprocessing across assets
"""

import sys
import os
import types

# Compatibility shim for distutils.version (Python 3.12+)
try:
    import distutils
    import distutils.version
except ImportError:
    d = types.ModuleType('distutils')
    dv = types.ModuleType('distutils.version')
    class LooseVersion:
        def __init__(self, vstring=None): self.vstring = vstring
        def __str__(self): return self.vstring
        def __lt__(self, other): return self.vstring < (other.vstring if hasattr(other, 'vstring') else other)
        def __le__(self, other): return self.vstring <= (other.vstring if hasattr(other, 'vstring') else other)
        def __gt__(self, other): return self.vstring > (other.vstring if hasattr(other, 'vstring') else other)
        def __ge__(self, other): return self.vstring >= (other.vstring if hasattr(other, 'vstring') else other)
        def __eq__(self, other): return self.vstring == (other.vstring if hasattr(other, 'vstring') else other)
    dv.LooseVersion = LooseVersion
    d.version = dv
    sys.modules['distutils'] = d
    sys.modules['distutils.version'] = dv

import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier
import pandas_datareader.data as web
import warnings
from multiprocessing import Pool, cpu_count
import time
from datetime import datetime
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import StrategyConfig, _strategy_config_from_env

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, 'cache')
BENCHMARKS_DIR = os.path.join(PROJECT_ROOT, 'benchmarks')
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(BENCHMARKS_DIR, exist_ok=True)

BBG_EXCEL_PATH = os.path.join(PROJECT_ROOT, 'cache', 'DATA PAUL.xlsx')
BBG_EXCEL_COLS = ['Date', 'SPTR', 'SPTRMDCP', 'RU20INTR', 'NDDUEAFE', 'NDUEEGF',
                  'LBUSTRUU', 'IBOXHY', 'LUACTRUU', 'DJUSRET', 'DBLCDBCE', 'GOLDLNPM', 'LUTLTRUU']

# "Bloomberg Indices" list tickers — loaded from DATA PAUL.xlsx instead of Yahoo Finance.
BBG_PRICE_COLS = {c: c for c in BBG_EXCEL_COLS[1:]}

_bbg_raw_cache: dict = {}


def _load_bbg_raw() -> pd.DataFrame:
    """Load and cache the full DATA PAUL.xlsx price table (indexed by date)."""
    if not _bbg_raw_cache:
        df = pd.read_excel(BBG_EXCEL_PATH, header=None, skiprows=6)
        df.columns = BBG_EXCEL_COLS
        df['Date'] = pd.to_datetime(df['Date'])
        _bbg_raw_cache['df'] = df.set_index('Date').sort_index()
    return _bbg_raw_cache['df']


def _load_bbg_price_series(ticker: str) -> pd.Series:
    """Return a price Series from DATA PAUL.xlsx for a Bloomberg Indices ticker."""
    s = _load_bbg_raw()[BBG_PRICE_COLS[ticker]].dropna()
    s.name = ticker
    return s


# ── BBG + Yahoo ETF Hybrid loader ─────────────────────────────────────────────
# Tickers are composite "<BBG>+<YETF>". BBG drives history up to ETF inception
# (exclusive); ETF drives from inception onward. Splice is performed in
# return-space and a synthetic continuous price series is rebuilt — so the
# JM/XGB pipeline sees one seamless series and no scale discontinuity.

def _parse_hybrid_ticker(ticker: str):
    """Return (bbg_ticker, etf_ticker) if `ticker` is a hybrid spec, else None."""
    if '+' not in ticker:
        return None
    bbg, etf = ticker.split('+', 1)
    if bbg not in BBG_PRICE_COLS:
        return None
    return bbg, etf


def _load_hybrid_price_series(bbg_ticker: str, etf_ticker: str,
                              data_start: str, fetch_end: str) -> pd.Series:
    """Splice BBG (pre-ETF) and Yahoo ETF (post-inception) into one price series.

    Splice is in return-space: BBG daily returns up to (ETF first date - 1),
    then ETF daily returns afterwards. The returned price series is the
    cumulative product re-based to BBG's starting level, so feature
    engineering downstream sees a continuous synthetic price.
    """
    bbg_series = _load_bbg_price_series(bbg_ticker)
    bbg_series = bbg_series[bbg_series.index >= pd.Timestamp(data_start)]

    etf_df = yf.download(etf_ticker, start=data_start, end=fetch_end,
                         auto_adjust=False, progress=False)
    if etf_df.empty:
        raise ValueError(f"Hybrid loader: Yahoo returned no data for {etf_ticker}")
    if isinstance(etf_df.columns, pd.MultiIndex):
        if 'Adj Close' in etf_df.columns.get_level_values(0):
            etf_series = etf_df['Adj Close'].iloc[:, 0]
        else:
            etf_series = etf_df.iloc[:, 0]
    else:
        etf_series = etf_df.get('Adj Close', etf_df.get('Close', etf_df.iloc[:, 0]))
    etf_series = etf_series.dropna().sort_index()

    # Splice: BBG returns up to AND INCLUDING the ETF's inception date (BBG has a
    # complete return on that day; ETF's pct_change on its first day is NaN).
    # ETF returns from the day AFTER inception onward.
    splice_date = etf_series.index.min()
    bbg_rets = bbg_series.pct_change()
    etf_rets = etf_series.pct_change()

    bbg_pre = bbg_rets[bbg_rets.index <= splice_date]
    etf_post = etf_rets[etf_rets.index > splice_date]
    combined_rets = pd.concat([bbg_pre, etf_post]).sort_index()
    combined_rets = combined_rets[~combined_rets.index.duplicated(keep='last')]

    base_level = float(bbg_series.iloc[0])
    growth = (1.0 + combined_rets.fillna(0.0)).cumprod()
    price = base_level * growth
    price.name = f"{bbg_ticker}+{etf_ticker}"
    return price


# ── Asset List Loading ────────────────────────────────────────────────────────

ASSET_LISTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'asset_lists.md')
DEFAULT_LIST_NAME = 'Yahoo ETFs'


def load_asset_lists(md_path=ASSET_LISTS_PATH):
    """Parse asset_lists.md -> dict[name -> {tickers, asset_classes, data_start}]."""
    lists = {}
    current_name = None
    current_tickers = []
    current_classes = {}
    current_data_start = None

    with open(md_path, 'r') as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith('## '):
                if current_name and current_tickers:
                    lists[current_name] = {
                        'tickers': current_tickers,
                        'asset_classes': current_classes,
                        'data_start': current_data_start,
                    }
                current_name = stripped[3:].strip()
                current_tickers = []
                current_classes = {}
                current_data_start = None
                continue

            if stripped.lower().startswith('data_start:'):
                current_data_start = stripped.split(':', 1)[1].strip()
                continue

            if stripped.startswith('|'):
                cells = [c.strip() for c in stripped.split('|')[1:-1]]
                if not cells or len(cells) < 2:
                    continue
                ticker = cells[0]
                if ticker.lower() == 'ticker' or ticker.startswith('---') or ticker.startswith(':---'):
                    continue
                if set(ticker.replace('-', '')) == set():
                    continue
                asset_class = cells[1]
                current_tickers.append(ticker)
                if asset_class not in current_classes:
                    current_classes[asset_class] = []
                current_classes[asset_class].append(ticker)

    if current_name and current_tickers:
        lists[current_name] = {
            'tickers': current_tickers,
            'asset_classes': current_classes,
            'data_start': current_data_start,
        }

    return lists


def parse_asset_list_selection(args, all_lists):
    """Parse CLI args to select an asset list. Returns list name or None (exit)."""
    if not args:
        return DEFAULT_LIST_NAME

    arg = ' '.join(args).strip()

    if arg.lower() in ('list', 'ls'):
        print("Available asset lists:")
        for name, info in all_lists.items():
            print(f"  {name!r}  ({len(info['tickers'])} tickers, data_start={info['data_start']})")
        return None

    if arg in ('-h', '--help', 'help'):
        print("Usage: python misc_scripts/benchmark_assets.py [LIST_NAME | list | --help]")
        print()
        print("  (no argument)       Use 'Yahoo ETFs' list")
        print("  \"Yahoo Mutual Funds\"  Use mutual fund long-history proxies (Yahoo Finance)")
        print("  \"Bloomberg Indices\"   Use paper-aligned total-return indices (DATA PAUL.xlsx)")
        print("  \"BBG+Yahoo ETF Hybrid\" BBG history pre-ETF inception + Yahoo ETF post-inception (return-space splice)")
        print("  list                Show all available asset lists")
        return None

    if arg in all_lists:
        return arg

    print(f"Error: asset list {arg!r} not found.")
    print("Available lists:", ', '.join(f"'{n}'" for n in all_lists))
    return None


# ── Configuration ─────────────────────────────────────────────────────────────

# Time periods to test (label, oos_start, oos_end)
TIME_PERIODS = [
    ('2007-2009 (GFC)',       '2007-01-01', '2010-01-01'),
    ('2010-2015 (Recovery)',  '2010-01-01', '2016-01-01'),
    ('2016-2019 (Late Cycle)','2016-01-01', '2020-01-01'),
    ('2020-2021 (COVID)',     '2020-01-01', '2022-01-01'),
    ('2022-2025 (Post-COVID)','2022-01-01', '2026-01-01'),
    ('Full (2007-2025)',      '2007-01-01', '2026-01-01'),
    ('Full (Paper 2007-2023)','2007-01-01', '2024-01-01'),  # Apples-to-apples window for paper Table 4 comparison
]

RISK_FREE_TICKER = '^IRX'
BOND_TICKER = 'VBMFX'
VIX_TICKER = '^VIX'
LARGECAP_TICKER = '^SP500TR'  # Paper Table 3: Stock-Bond Corr uses LargeCap vs AggBond for ALL assets
DATA_START = '1993-01-01'  # ETFs need less lookback than ^SP500TR

TRANSACTION_COST = 0.0005
LAMBDA_GRID = [4.64, 10.0, 15.0, 21.54, 30.0, 46.42, 70.0, 100.0]  # Dense 8-pt mid-range (Session 5: adds 15, 30, 70 to fill gaps; avoids low λ<4.6 that WF overpicks)
EWMA_HL_GRID = [0, 1, 2, 4, 8, 12, 16]  # Fixed auto-tune grid (not user-configurable)

# Allow env var overrides (from Diagnostics Launcher page 3)
import json as _json
if os.environ.get('XGB_TRANSACTION_COST'):
    TRANSACTION_COST = float(os.environ['XGB_TRANSACTION_COST'])
if os.environ.get('XGB_LAMBDA_GRID'):
    LAMBDA_GRID = _json.loads(os.environ['XGB_LAMBDA_GRID'])

# Paper-prescribed EWMA halflives (from paper Section 4.2, tuned on Bloomberg data).
# Used only when ewma_mode="paper" on StrategyConfig.
PAPER_EWMA_HL = {
    # hl=8: LargeCap, MidCap, REIT, AggBond, Treasury
    '^SP500TR': 8, 'IVV': 8,
    'IJH': 8, 'VIMSX': 8,
    'IWM': 8,
    'IYR': 8, 'FRESX': 8,
    'AGG': 8, 'VBMFX': 8,
    'SPTL': 8, 'VUSTX': 8, 'TLT': 8, 'VGLT': 8,
    # Session 5 override: NAESX (Vanguard SmallCap) needs hl=2 not hl=8
    # Diagnostic: hl=8 → Δ=-0.006 (LOSE), hl=2 → Δ=+0.142 (WIN). Mutual fund NAV
    # smoothing differs from Bloomberg Russell 2000 TR index.
    'NAESX': 2,
    # hl=4: Commodity, Gold
    'DBC': 4, '^SPGSCI': 4, 'DBLCDBCE': 4,
    'GLD': 4, 'GC=F': 4, 'IAU': 4, 'GOLDLNPM': 4,
    # hl=2: Corporate
    'SPBO': 2, 'VWESX': 2, 'LUACTRUU': 2,
    # hl=0: EM, EAFE, HighYield
    'EEM': 0, 'VEIEX': 0, 'NDUEEGF': 0,
    'EFA': 0, 'FDIVX': 0, 'NDDUEAFE': 0,
    'HYG': 0, 'VWEHX': 0, 'IBOXHY': 0,
    # Bloomberg Indices (hl=8: equity, AggBond, Treasury, REIT)
    'SPTR': 8, 'SPTRMDCP': 8, 'RU20INTR': 8,
    'LBUSTRUU': 8, 'LUTLTRUU': 8, 'DJUSRET': 8,
    # BBG+Yahoo ETF Hybrid — same hl as ETF leg (paper-prescribed for each asset class)
    'SPTR+IVV': 8, 'SPTRMDCP+IJH': 8, 'RU20INTR+IWM': 8,
    'DJUSRET+IYR': 8, 'LBUSTRUU+AGG': 8, 'LUTLTRUU+SPTL': 8,
    'DBLCDBCE+DBC': 4, 'GOLDLNPM+GLD': 4,
    'LUACTRUU+SPBO': 2,
    'NDUEEGF+EEM': 0, 'NDDUEAFE+EFA': 0, 'IBOXHY+HYG': 0,
}
VALIDATION_WINDOW_YRS = 5

# Paper Table 2: bonds and gold use only 6 return features (Avg_Return×3, Sortino×3),
# excluding Downside Deviation (DD_log_5, DD_log_21) which are equity-specific.
DD_EXCLUDE_TICKERS = {'AGG', 'VBMFX',                    # AggBond
                      'SPTL', 'VUSTX', 'IEF', 'TLT', 'VGLT',  # Treasury
                      'GLD', 'GC=F', 'IAU',               # Gold (Yahoo)
                      'LBUSTRUU', 'LUTLTRUU', 'GOLDLNPM', # Bloomberg AggBond/Treasury/Gold
                      'LBUSTRUU+AGG', 'LUTLTRUU+SPTL', 'GOLDLNPM+GLD'}  # Hybrid AggBond/Treasury/Gold

# ── Paper Table 4 reference (2007-2023, Bloomberg, gross of TC) ───────────────
# Source: arXiv 2406.09578v2, Table 4. Values are GROSS of transaction costs.
# Sharpe ratios and Max Drawdowns for B&H, JM, and JM-XGB strategies.
PAPER_TABLE4 = {
    'LargeCap':  {'bh_sharpe': 0.50, 'jm_sharpe': 0.59, 'jmxgb_sharpe': 0.79,
                  'bh_mdd': -0.5525, 'jm_mdd': -0.2478, 'jmxgb_mdd': -0.1769},
    'MidCap':    {'bh_sharpe': 0.45, 'jm_sharpe': 0.49, 'jmxgb_sharpe': 0.59,
                  'bh_mdd': -0.5515, 'jm_mdd': -0.3324, 'jmxgb_mdd': -0.2989},
    'SmallCap':  {'bh_sharpe': 0.36, 'jm_sharpe': 0.28, 'jmxgb_sharpe': 0.51,
                  'bh_mdd': -0.5889, 'jm_mdd': -0.3835, 'jmxgb_mdd': -0.3584},
    'EAFE':      {'bh_sharpe': 0.20, 'jm_sharpe': 0.28, 'jmxgb_sharpe': 0.56,
                  'bh_mdd': -0.6041, 'jm_mdd': -0.2972, 'jmxgb_mdd': -0.1993},
    'EM':        {'bh_sharpe': 0.20, 'jm_sharpe': 0.65, 'jmxgb_sharpe': 0.85,
                  'bh_mdd': -0.6525, 'jm_mdd': -0.2622, 'jmxgb_mdd': -0.2130},
    'REIT':      {'bh_sharpe': 0.27, 'jm_sharpe': 0.39, 'jmxgb_sharpe': 0.56,
                  'bh_mdd': -0.7423, 'jm_mdd': -0.5471, 'jmxgb_mdd': -0.3270},
    'AggBond':   {'bh_sharpe': 0.46, 'jm_sharpe': 0.43, 'jmxgb_sharpe': 0.67,
                  'bh_mdd': -0.1841, 'jm_mdd': -0.0609, 'jmxgb_mdd': -0.0630},
    'Treasury':  {'bh_sharpe': 0.26, 'jm_sharpe': 0.21, 'jmxgb_sharpe': 0.38,
                  'bh_mdd': -0.4691, 'jm_mdd': -0.2285, 'jmxgb_mdd': -0.1746},
    'HighYield': {'bh_sharpe': 0.67, 'jm_sharpe': 1.49, 'jmxgb_sharpe': 1.88,
                  'bh_mdd': -0.3287, 'jm_mdd': -0.1388, 'jmxgb_mdd': -0.1025},
    'Corporate': {'bh_sharpe': 0.54, 'jm_sharpe': 0.83, 'jmxgb_sharpe': 0.76,
                  'bh_mdd': -0.2204, 'jm_mdd': -0.0826, 'jmxgb_mdd': -0.0679},
    'Commodity': {'bh_sharpe': 0.03, 'jm_sharpe': 0.08, 'jmxgb_sharpe': 0.23,
                  'bh_mdd': -0.7554, 'jm_mdd': -0.5848, 'jmxgb_mdd': -0.4790},
    'Gold':      {'bh_sharpe': 0.43, 'jm_sharpe': 0.12, 'jmxgb_sharpe': 0.31,
                  'bh_mdd': -0.4462, 'jm_mdd': -0.3178, 'jmxgb_mdd': -0.2162},
}

# Mapping from ticker (all three asset lists) to paper asset name in PAPER_TABLE4
TICKER_TO_PAPER_ASSET = {
    # Yahoo Mutual Funds proxies (^SPGSCI replaces delisted PCASX; closest corr with DBLCDBCE + long history)
    '^SP500TR': 'LargeCap', 'VIMSX': 'MidCap',  'NAESX': 'SmallCap',
    'FDIVX':    'EAFE',     'VEIEX': 'EM',       'VBMFX': 'AggBond',
    'VUSTX':    'Treasury', 'VWEHX': 'HighYield','VWESX': 'Corporate',
    'FRESX':    'REIT',     '^SPGSCI':'Commodity','GC=F':  'Gold',
    # Yahoo ETFs (investable only; DBC has 11mo pre-2007 but partial-window logic handles it)
    'IVV':  'LargeCap', 'IJH':  'MidCap',  'IWM':  'SmallCap',
    'EFA':  'EAFE',     'EEM':  'EM',       'AGG':  'AggBond',
    'SPTL': 'Treasury', 'HYG':  'HighYield','SPBO': 'Corporate',
    'IYR':  'REIT',     'DBC':  'Commodity','GLD':  'Gold',
    # Bloomberg Indices
    'SPTR':     'LargeCap', 'SPTRMDCP': 'MidCap',    'RU20INTR': 'SmallCap',
    'NDDUEAFE': 'EAFE',     'NDUEEGF':  'EM',         'LBUSTRUU': 'AggBond',
    'LUTLTRUU': 'Treasury', 'IBOXHY':   'HighYield',  'LUACTRUU': 'Corporate',
    'DJUSRET':  'REIT',     'DBLCDBCE': 'Commodity',  'GOLDLNPM': 'Gold',
    # BBG+Yahoo ETF Hybrid
    'SPTR+IVV':      'LargeCap', 'SPTRMDCP+IJH':  'MidCap',    'RU20INTR+IWM':  'SmallCap',
    'NDDUEAFE+EFA':  'EAFE',     'NDUEEGF+EEM':   'EM',         'LBUSTRUU+AGG':  'AggBond',
    'LUTLTRUU+SPTL': 'Treasury', 'IBOXHY+HYG':    'HighYield',  'LUACTRUU+SPBO': 'Corporate',
    'DJUSRET+IYR':   'REIT',     'DBLCDBCE+DBC':  'Commodity',  'GOLDLNPM+GLD':  'Gold',
}

# ── Statistical Jump Model (same as main.py) ─────────────────────────────────

class StatisticalJumpModel:
    """Matches the paper's jumpmodels library: k-means++ init, n_init=10, max_iter=1000, tol=1e-8."""
    def __init__(self, n_states=2, lambda_penalty=10.0, max_iter=1000, n_init=10, tol=1e-8):
        self.n_states = n_states
        self.lambda_penalty = lambda_penalty
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.means = None
        self.val_ = np.inf

    def _viterbi_full(self, X_arr, distances):
        """Full Viterbi (forward + backtrack) for E-step during fitting."""
        n_samples = X_arr.shape[0]
        cost_matrix = np.zeros((n_samples, self.n_states))
        back_pointers = np.zeros((n_samples, self.n_states), dtype=int)
        cost_matrix[0] = distances[0]
        penalty_matrix = self.lambda_penalty * (1 - np.eye(self.n_states))

        for t in range(1, n_samples):
            c00 = cost_matrix[t-1, 0]
            c10 = cost_matrix[t-1, 1] + self.lambda_penalty
            if c00 <= c10:
                cost_matrix[t, 0] = c00 + distances[t, 0]
                back_pointers[t, 0] = 0
            else:
                cost_matrix[t, 0] = c10 + distances[t, 0]
                back_pointers[t, 0] = 1
            c01 = cost_matrix[t-1, 0] + self.lambda_penalty
            c11 = cost_matrix[t-1, 1]
            if c01 <= c11:
                cost_matrix[t, 1] = c01 + distances[t, 1]
                back_pointers[t, 1] = 0
            else:
                cost_matrix[t, 1] = c11 + distances[t, 1]
                back_pointers[t, 1] = 1

        states = np.zeros(n_samples, dtype=int)
        states[-1] = np.argmin(cost_matrix[-1])
        obj_val = cost_matrix[-1, states[-1]]
        for t in range(n_samples - 2, -1, -1):
            states[t] = back_pointers[t+1, states[t+1]]
        return states, obj_val

    def fit_predict(self, X):
        from sklearn.cluster import kmeans_plusplus
        from sklearn.utils import check_random_state

        X_arr = np.array(X)
        n_samples, n_features = X_arr.shape
        rng = check_random_state(42)
        best_states = None
        best_val = np.inf
        best_means = None

        for init_idx in range(self.n_init):
            centers, _ = kmeans_plusplus(X_arr, self.n_states, random_state=rng)
            means = centers.copy()
            states = np.zeros(n_samples, dtype=int)
            val_prev = np.inf

            for iteration in range(self.max_iter):
                distances = 0.5 * np.sum((X_arr[:, None, :] - means[None, :, :])**2, axis=2)
                new_states, obj_val = self._viterbi_full(X_arr, distances)
                for k in range(self.n_states):
                    mask = (new_states == k)
                    if np.sum(mask) > 0:
                        means[k] = np.mean(X_arr[mask], axis=0)
                if np.array_equal(states, new_states) or (val_prev - obj_val <= self.tol):
                    states = new_states
                    break
                val_prev = obj_val
                states = new_states

            distances = 0.5 * np.sum((X_arr[:, None, :] - means[None, :, :])**2, axis=2)
            _, final_val = self._viterbi_full(X_arr, distances)
            if final_val < best_val:
                best_val = final_val
                best_states = states.copy()
                best_means = means.copy()

        self.means = best_means
        self.val_ = best_val
        return best_states

    def predict_online(self, X, last_known_state=None):
        """Forward-only Viterbi (online) prediction matching the paper's jumpmodels library."""
        X_arr = np.array(X)
        n_samples = X_arr.shape[0]
        loss_mx = 0.5 * np.sum((X_arr[:, None, :] - self.means[None, :, :])**2, axis=2)
        penalty_mx = self.lambda_penalty * (1 - np.eye(self.n_states))
        values = np.empty((n_samples, self.n_states))
        values[0] = loss_mx[0]
        for t in range(1, n_samples):
            values[t] = loss_mx[t] + (values[t-1][:, np.newaxis] + penalty_mx).min(axis=0)
        return values.argmin(axis=1)

# ── Data Fetching ─────────────────────────────────────────────────────────────

def fetch_etf_data(ticker, data_start=None):
    """Fetch and prepare features for a single ETF. Returns (ticker, df) or (ticker, None)."""
    if data_start is None:
        data_start = DATA_START
    exclude_dd = ticker in DD_EXCLUDE_TICKERS
    cache_suffix = '_noDD' if exclude_dd else ''
    cache_ticker = ticker.replace('+', '_PLUS_')  # hybrid tickers contain '+', not legal in filenames on some FS
    cache_file = os.path.join(CACHE_DIR, f'data_cache_{cache_ticker}_{data_start.replace("-", "")}{cache_suffix}_v2.pkl')
    if os.path.exists(cache_file):
        cached_df = pd.read_pickle(cache_file)
        # Check staleness: re-fetch if cache is >30 days behind today
        today = pd.Timestamp.now().normalize()
        gap_days = (today - cached_df.index.max()).days
        if gap_days <= 30:
            return ticker, cached_df
        else:
            print(f"  {ticker}: cache {gap_days}d stale, re-fetching...")

    try:
        fetch_end = (pd.Timestamp.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        # Bloomberg Indices list: load price from DATA PAUL.xlsx,
        # then fetch auxiliary series (VIX, IRX, LargeCap, AggBond) from Yahoo.
        hybrid_spec = _parse_hybrid_ticker(ticker)
        if hybrid_spec is not None:
            bbg_t, etf_t = hybrid_spec
            spliced = _load_hybrid_price_series(bbg_t, etf_t, data_start, fetch_end)
            aux_tickers = dict.fromkeys([BOND_TICKER, RISK_FREE_TICKER, VIX_TICKER, LARGECAP_TICKER])
            raw = {ticker: spliced}
            for t in aux_tickers:
                df_t = yf.download(t, start=data_start, end=fetch_end, auto_adjust=False, progress=False)
                if df_t.empty:
                    continue
                if isinstance(df_t.columns, pd.MultiIndex):
                    if 'Adj Close' in df_t.columns.get_level_values(0):
                        s = df_t['Adj Close'].iloc[:, 0].rename(t)
                    else:
                        s = df_t.iloc[:, 0].rename(t)
                else:
                    s = df_t.get('Adj Close', df_t.get('Close', df_t.iloc[:, 0])).rename(t)
                raw[t] = s
        elif ticker in BBG_PRICE_COLS:
            bbg_series = _load_bbg_price_series(ticker)
            aux_tickers = dict.fromkeys([BOND_TICKER, RISK_FREE_TICKER, VIX_TICKER, LARGECAP_TICKER])
            raw = {ticker: bbg_series}
            for t in aux_tickers:
                df_t = yf.download(t, start=data_start, end=fetch_end, auto_adjust=False, progress=False)
                if df_t.empty:
                    continue
                if isinstance(df_t.columns, pd.MultiIndex):
                    if 'Adj Close' in df_t.columns.get_level_values(0):
                        s = df_t['Adj Close'].iloc[:, 0].rename(t)
                    else:
                        s = df_t.iloc[:, 0].rename(t)
                else:
                    s = df_t.get('Adj Close', df_t.get('Close', df_t.iloc[:, 0])).rename(t)
                raw[t] = s
        else:
            # Yahoo Finance (ETFs or Mutual Funds lists): download target + auxiliary tickers.
            # Always include LARGECAP_TICKER so Stock_Bond_Corr = corr(LargeCap, AggBond) for all assets.
            tickers_to_fetch = dict.fromkeys([ticker, BOND_TICKER, RISK_FREE_TICKER, VIX_TICKER, LARGECAP_TICKER])
            raw = {}
            for t in tickers_to_fetch:
                df_t = yf.download(t, start=data_start, end=fetch_end, auto_adjust=False, progress=False)
                if df_t.empty:
                    continue
                if isinstance(df_t.columns, pd.MultiIndex):
                    if 'Adj Close' in df_t.columns.get_level_values(0):
                        s = df_t['Adj Close'].iloc[:, 0].rename(t)
                    else:
                        s = df_t.iloc[:, 0].rename(t)
                else:
                    s = df_t.get('Adj Close', df_t.get('Close', df_t.iloc[:, 0])).rename(t)
                raw[t] = s

        if ticker not in raw or RISK_FREE_TICKER not in raw or VIX_TICKER not in raw:
            return ticker, None

        data = pd.concat(raw.values(), axis=1).ffill().dropna()

        # FRED macro data — use shared cache (ticker-independent)
        import time
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        fred_month = pd.Timestamp.now().strftime('%Y-%m')
        fred_cache_file = os.path.join(cache_dir, f'fred_cache_{data_start[:4]}_{fred_month}.pkl')
        fred = None
        if os.path.exists(fred_cache_file):
            fred = pd.read_pickle(fred_cache_file)
        else:
            for attempt in range(5):
                try:
                    fred = web.DataReader(['DGS2', 'DGS10'], 'fred', data_start, fetch_end)
                    fred.to_pickle(fred_cache_file)
                    break
                except Exception as e:
                    print(f"Failed to fetch FRED data (attempt {attempt+1}/5): {e}")
                    time.sleep(3 * (attempt + 1))
        if fred is None:
            raise ValueError("Failed to download FRED data after 5 attempts.")
        fred = fred.ffill().dropna()
        df = data.join(fred, how='inner')

        if len(df) < 252 * 5:
            return ticker, None

        features = pd.DataFrame(index=df.index)
        target_returns = df[ticker].pct_change().fillna(0)
        features['Target_Return'] = target_returns
        features['RF_Rate'] = (df[RISK_FREE_TICKER] / 100) / 252
        features['Excess_Return'] = target_returns - features['RF_Rate']

        downside_returns = np.minimum(features['Excess_Return'], 0)
        if not exclude_dd:
            for hl in [5, 21]:
                ewm_var = (downside_returns ** 2).ewm(halflife=hl).mean()
                features[f'DD_log_{hl}'] = np.log(np.sqrt(ewm_var).fillna(0) + 1e-8)
        for hl in [5, 10, 21]:
            features[f'Avg_Ret_{hl}'] = features['Excess_Return'].ewm(halflife=hl).mean()
        for hl in [5, 10, 21]:
            ewm_var = (downside_returns ** 2).ewm(halflife=hl).mean()
            ewm_dd_raw = np.maximum(np.sqrt(ewm_var).fillna(1e-8), 1e-8)
            features[f'Sortino_{hl}'] = (features[f'Avg_Ret_{hl}'] / ewm_dd_raw).clip(-10, 10)

        features['Yield_2Y_EWMA_diff'] = df['DGS2'].diff().fillna(0).ewm(halflife=21).mean()
        slope = df['DGS10'] - df['DGS2']
        features['Yield_Slope_EWMA_10'] = slope.ewm(halflife=10).mean()
        features['Yield_Slope_EWMA_diff_21'] = slope.diff().fillna(0).ewm(halflife=21).mean()
        features['VIX_EWMA_log_diff'] = np.log(df[VIX_TICKER] / df[VIX_TICKER].shift(1)).fillna(0).ewm(halflife=63).mean()

        # Paper Table 3: Stock-Bond Corr = rolling corr(LargeCap_ret, AggBond_ret), 252-day lookback.
        # This is a macro feature identical for all assets — NOT corr(target, bond).
        if BOND_TICKER in df.columns and LARGECAP_TICKER in df.columns:
            largecap_rets = df[LARGECAP_TICKER].pct_change().fillna(0)
            bond_rets = df[BOND_TICKER].pct_change().fillna(0)
            features['Stock_Bond_Corr'] = largecap_rets.rolling(window=252).corr(bond_rets).fillna(0)
        else:
            features['Stock_Bond_Corr'] = 0.0

        final = features.dropna()
        final.to_pickle(cache_file)
        return ticker, final

    except Exception as e:
        print(f"  [WARN] Failed to fetch {ticker}: {e}")
        return ticker, None

# ── Core Backtest Logic (fast version, no SHAP) ──────────────────────────────

def run_period_forecast_fast(df, current_date, lambda_penalty, cache, config: StrategyConfig = None, include_xgboost=True):
    if config is None:
        config = StrategyConfig()
        
    cache_key = (current_date, lambda_penalty, include_xgboost, config.name)
    if cache_key in cache:
        return cache[cache_key]

    train_start = current_date - pd.DateOffset(years=11)
    train_df = df[(df.index >= train_start) & (df.index < current_date)].copy()
    if len(train_df) < 252 * 3:  # relaxed for shorter ETF histories
        cache[cache_key] = None
        return None

    return_features = [c for c in train_df.columns if c.startswith(('DD_', 'Avg_Ret_', 'Sortino_'))]
    X_train_jm = train_df[return_features]
    std = X_train_jm.std()
    std[std == 0] = 1.0
    X_train_jm = (X_train_jm - X_train_jm.mean()) / std

    jm = StatisticalJumpModel(n_states=2, lambda_penalty=lambda_penalty)
    identified_states = jm.fit_predict(X_train_jm.values)

    cum_ret_0 = train_df['Excess_Return'][identified_states == 0].sum()
    cum_ret_1 = train_df['Excess_Return'][identified_states == 1].sum()
    if cum_ret_1 > cum_ret_0:
        identified_states = 1 - identified_states
        jm.means = jm.means[::-1].copy()

    train_df['Target_State'] = np.roll(identified_states, -1)
    train_df = train_df.iloc[:-1]

    oos_end = current_date + pd.DateOffset(months=6)
    oos_df = df[(df.index >= current_date) & (df.index < oos_end)].copy()
    if len(oos_df) == 0:
        cache[cache_key] = None
        return None

    if not include_xgboost:
        X_oos_jm = oos_df[return_features]
        train_std = train_df[return_features].std()
        train_std[train_std == 0] = 1.0
        X_oos_jm = (X_oos_jm - train_df[return_features].mean()) / train_std
        oos_states = jm.predict_online(X_oos_jm.values, last_known_state=identified_states[-1])
        oos_df['Forecast_State'] = oos_states
        result = oos_df[['Target_Return', 'RF_Rate', 'Forecast_State']]
        cache[cache_key] = result
        return result

    macro_features = ['Yield_2Y_EWMA_diff', 'Yield_Slope_EWMA_10',
                      'Yield_Slope_EWMA_diff_21', 'VIX_EWMA_log_diff', 'Stock_Bond_Corr']
    ablation = getattr(config, 'feature_ablation', 'all')
    if ablation == 'return_only':
        all_features = list(return_features)
    elif ablation == 'macro_only':
        all_features = [f for f in macro_features if f in train_df.columns]
    else:
        all_features = list(return_features) + [f for f in macro_features if f in train_df.columns]

    X_train_xgb = train_df[all_features]
    y_train_xgb = train_df['Target_State']
    X_oos_xgb = oos_df[all_features]

    # Edge case: if JM assigned all data to one state, XGB can't train a classifier
    unique_labels = y_train_xgb.unique()
    if len(unique_labels) < 2:
        # All one class — predict that class uniformly
        only_class = unique_labels[0]
        oos_probs = np.full(len(X_oos_xgb), float(only_class))
        oos_df['Raw_Prob'] = oos_probs
        result = oos_df[['Target_Return', 'RF_Rate', 'Raw_Prob']]
        cache[cache_key] = result
        return result

    xgb = XGBClassifier(
        eval_metric='logloss', random_state=42, verbosity=0, **config.xgb_params
    )
    xgb.fit(X_train_xgb, y_train_xgb)
    oos_probs = xgb.predict_proba(X_oos_xgb)[:, 1]
    oos_df['Raw_Prob'] = oos_probs
    result = oos_df[['Target_Return', 'RF_Rate', 'Raw_Prob']]
    cache[cache_key] = result
    return result


def simulate_strategy_fast(df, start_date, end_date, lambda_penalty, cache, config: StrategyConfig = None, include_xgboost=True, ewma_halflife=8):
    if config is None:
        config = StrategyConfig()
        
    results = []
    current_date = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    while current_date < end_dt:
        res = run_period_forecast_fast(df, current_date, lambda_penalty, cache, config, include_xgboost)
        if res is not None:
            results.append(res)
        current_date += pd.DateOffset(months=6)
    if not results:
        return pd.DataFrame()
    full_res = pd.concat(results)

    if include_xgboost:
        if ewma_halflife == 0:
            full_res['State_Prob'] = full_res['Raw_Prob']
        else:
            full_res['State_Prob'] = full_res['Raw_Prob'].ewm(halflife=ewma_halflife).mean()
            
        if config.allocation_style == "binary":
            full_res['Forecast_State'] = (full_res['State_Prob'] > config.prob_threshold).astype(int)
            trading_signals = full_res['Forecast_State'].shift(1).fillna(0)
            alloc_target = 1.0 - trading_signals
        elif config.allocation_style == "continuous":
            alloc_target = (1.0 - full_res['State_Prob']).shift(1).fillna(1.0)
    else:
        trading_signals = full_res['Forecast_State'].shift(1).fillna(0)
        alloc_target = 1.0 - trading_signals

    strat_returns = (alloc_target * full_res['Target_Return']) + ((1.0 - alloc_target) * full_res['RF_Rate'])
    trades = alloc_target.diff().abs().fillna(0)
    
    full_res['Strat_Return'] = strat_returns - (trades * TRANSACTION_COST)
    return full_res


def calculate_metrics(returns_series, rf_series):
    if len(returns_series) < 10:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    excess_returns = returns_series - rf_series
    ann_excess_ret = excess_returns.mean() * 252
    ann_vol = returns_series.std() * np.sqrt(252)
    sharpe = ann_excess_ret / ann_vol if ann_vol != 0 else 0.0

    downside_returns = np.minimum(excess_returns, 0)
    ann_downside_vol = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(252)
    sortino = ann_excess_ret / ann_downside_vol if ann_downside_vol != 0 else 0.0

    cum_wealth = (1 + returns_series).cumprod()
    peak = cum_wealth.cummax()
    drawdown = (cum_wealth - peak) / peak
    mdd = drawdown.min()

    n_days = len(returns_series)
    ann_geom_ret = cum_wealth.iloc[-1] ** (252 / n_days) - 1 if n_days > 0 else 0.0
    return ann_geom_ret, ann_vol, sharpe, sortino, mdd


# ── Single-Asset Full Backtest ────────────────────────────────────────────────

def backtest_single_asset(args):
    """Run full walk-forward backtest for one ETF. Returns (result_dicts, full_period_ts_or_None)."""
    ticker, df, config, data_start = args
    results = []
    cache = {}  # per-asset forecast cache
    full_period_ts = None  # timeseries for the full OOS period (State_Prob, Forecast_State, Target_Return, RF_Rate)

    for period_name, oos_start, oos_end in TIME_PERIODS:
        oos_start_dt = pd.to_datetime(oos_start)
        oos_end_dt = pd.to_datetime(oos_end)

        # Per-asset OOS start: need at least 3 years of pre-OOS history for training.
        # Short-history ETFs (e.g. SPTL, HYG, SPBO inception >2004) get a partial window
        # starting at data_start+3y rather than being dropped from the period entirely.
        effective_oos_start = max(oos_start_dt, df.index[0] + pd.DateOffset(years=3))
        if effective_oos_start >= oos_end_dt:
            results.append({'Ticker': ticker, 'Period': period_name, 'Strategy': 'JM-XGB', 'Ann_Ret': np.nan, 'Ann_Vol': np.nan, 'Sharpe': np.nan, 'Sortino': np.nan, 'Max_DD': np.nan, 'Sharpe_Gross': np.nan, 'Ann_Ret_Gross': np.nan, 'Max_DD_Gross': np.nan, 'EWMA_HL': np.nan})
            results.append({'Ticker': ticker, 'Period': period_name, 'Strategy': 'B&H', 'Ann_Ret': np.nan, 'Ann_Vol': np.nan, 'Sharpe': np.nan, 'Sortino': np.nan, 'Max_DD': np.nan, 'Sharpe_Gross': np.nan, 'Ann_Ret_Gross': np.nan, 'Max_DD_Gross': np.nan, 'EWMA_HL': np.nan})
            continue

        # Phase 1: Tune EWMA halflife on initial validation window
        if config.ewma_mode == "paper" and ticker in PAPER_EWMA_HL:
            best_ewma_hl = PAPER_EWMA_HL[ticker]
        else:
            # Auto-tune: search fixed grid on pre-OOS validation window
            init_val_start = effective_oos_start - pd.DateOffset(years=VALIDATION_WINDOW_YRS)
            if config.validation_window_type == 'expanding':
                init_val_start = pd.to_datetime(data_start)

            best_ewma_hl = EWMA_HL_GRID[-1]
            best_init_metric = -np.inf
            for hl in EWMA_HL_GRID:
                for lmbda in LAMBDA_GRID:
                    val_res = simulate_strategy_fast(df, init_val_start, effective_oos_start, lmbda, cache, config, include_xgboost=True, ewma_halflife=hl)
                    if not val_res.empty:
                        _, _, sharpe, sortino, _ = calculate_metrics(val_res['Strat_Return'], val_res['RF_Rate'])
                        metric_val = sortino if config.tuning_metric == 'sortino' else sharpe
                        if not np.isnan(metric_val) and metric_val > best_init_metric:
                            best_init_metric = metric_val
                            best_ewma_hl = hl

        # Phase 2: Walk-forward lambda tuning + OOS execution
        current_date = effective_oos_start
        jm_xgb_chunks = []
        lambda_history = []

        while current_date < oos_end_dt:
            chunk_end = min(current_date + pd.DateOffset(months=6), oos_end_dt)
            
            if config.validation_window_type == 'expanding':
                val_start = pd.to_datetime(data_start)
            else:
                val_start = current_date - pd.DateOffset(years=VALIDATION_WINDOW_YRS)

            lambda_scores = []

            if config.lambda_subwindow_consensus:
                # Run simulate_strategy ONCE per lambda for full validation window (hits cache),
                # then slice into 3 overlapping sub-windows and evaluate each independently.
                val_duration = (current_date - val_start).days
                sub_boundaries = [
                    (val_start, val_start + pd.DateOffset(days=val_duration // 2)),
                    (val_start + pd.DateOffset(days=val_duration // 4), val_start + pd.DateOffset(days=3 * val_duration // 4)),
                    (val_start + pd.DateOffset(days=val_duration // 2), current_date),
                ]
                sub_best_lambdas = [[] for _ in sub_boundaries]
                for lmbda in LAMBDA_GRID:
                    val_res = simulate_strategy_fast(df, val_start, current_date, lmbda, cache, config, include_xgboost=True, ewma_halflife=best_ewma_hl)
                    if val_res.empty:
                        continue
                    for sw_idx, (sw_start, sw_end) in enumerate(sub_boundaries):
                        sw_slice = val_res[(val_res.index >= sw_start) & (val_res.index < sw_end)]
                        if len(sw_slice) >= 60:
                            _, _, sharpe, sortino, _ = calculate_metrics(sw_slice['Strat_Return'], sw_slice['RF_Rate'])
                            metric_val = sortino if config.tuning_metric == 'sortino' else sharpe
                            if not np.isnan(metric_val):
                                sub_best_lambdas[sw_idx].append((metric_val, lmbda))
                consensus_lambdas = []
                for sw_scores in sub_best_lambdas:
                    if sw_scores:
                        sw_scores.sort(key=lambda x: x[0], reverse=True)
                        consensus_lambdas.append(sw_scores[0][1])
                if consensus_lambdas:
                    best_lambda = float(np.median(consensus_lambdas))
                    best_lambdas = [best_lambda]
                    lambda_scores = [(0.0, best_lambda)]
                else:
                    best_lambda = LAMBDA_GRID[len(LAMBDA_GRID) // 2]
                    best_lambdas = [best_lambda]
                    lambda_scores = [(0.0, best_lambda)]
            else:
                for lmbda in LAMBDA_GRID:
                    val_res = simulate_strategy_fast(df, val_start, current_date, lmbda, cache, config, include_xgboost=True, ewma_halflife=best_ewma_hl)
                    if not val_res.empty:
                        _, _, sharpe, sortino, _ = calculate_metrics(val_res['Strat_Return'], val_res['RF_Rate'])
                        metric_val = sortino if config.tuning_metric == 'sortino' else sharpe
                        if not np.isnan(metric_val):
                            lambda_scores.append((metric_val, lmbda))

                lambda_scores.sort(key=lambda x: x[0], reverse=True)
                if not lambda_scores:
                    lambda_scores = [(0.0, LAMBDA_GRID[len(LAMBDA_GRID)//2])]

                if config.lambda_selection == 'median_positive':
                    positive_lambdas = [lam for score, lam in lambda_scores if score > 0]
                    if positive_lambdas:
                        best_lambda = float(np.median(positive_lambdas))
                        best_lambdas = [best_lambda]
                    else:
                        best_lambdas = [l for _, l in lambda_scores[:max(1, config.lambda_ensemble_k)]]
                        best_lambda = best_lambdas[0]
                else:
                    best_lambdas = [l for _, l in lambda_scores[:max(1, config.lambda_ensemble_k)]]
                    best_lambda = best_lambdas[0]

            if config.lambda_smoothing and lambda_history:
                best_lambda = (0.7 * best_lambda) + (0.3 * lambda_history[-1])

            lambda_history.append(best_lambda)

            if config.lambda_ensemble_k > 1:
                oos_chunks = []
                for l_val in best_lambdas:
                    chunk = run_period_forecast_fast(df, current_date, l_val, cache, config, include_xgboost=True)
                    if chunk is not None: oos_chunks.append(chunk)
                if oos_chunks:
                    avg_prob = sum(c['Raw_Prob'] for c in oos_chunks) / len(oos_chunks)
                    final_chunk = oos_chunks[0].copy()
                    final_chunk['Raw_Prob'] = avg_prob
                    jm_xgb_chunks.append(final_chunk)
            else:
                oos_chunk = run_period_forecast_fast(df, current_date, best_lambda, cache, config, include_xgboost=True)
                if oos_chunk is not None:
                    jm_xgb_chunks.append(oos_chunk)
                    
            current_date = chunk_end

        if not jm_xgb_chunks:
            results.append({
                'Ticker': ticker, 'Period': period_name,
                'Strategy': 'JM-XGB', 'Ann_Ret': np.nan, 'Ann_Vol': np.nan,
                'Sharpe': np.nan, 'Sortino': np.nan, 'Max_DD': np.nan,
                'Sharpe_Gross': np.nan, 'Ann_Ret_Gross': np.nan, 'Max_DD_Gross': np.nan,
            })
            results.append({
                'Ticker': ticker, 'Period': period_name,
                'Strategy': 'B&H', 'Ann_Ret': np.nan, 'Ann_Vol': np.nan,
                'Sharpe': np.nan, 'Sortino': np.nan, 'Max_DD': np.nan,
                'Sharpe_Gross': np.nan, 'Ann_Ret_Gross': np.nan, 'Max_DD_Gross': np.nan,
            })
            continue

        jm_xgb_df = pd.concat(jm_xgb_chunks)

        # Apply EWMA + threshold + signal shift
        if best_ewma_hl == 0:
            jm_xgb_df['State_Prob'] = jm_xgb_df['Raw_Prob']
        else:
            jm_xgb_df['State_Prob'] = jm_xgb_df['Raw_Prob'].ewm(halflife=best_ewma_hl).mean()
            
        if config.allocation_style == "binary":
            jm_xgb_df['Forecast_State'] = (jm_xgb_df['State_Prob'] > config.prob_threshold).astype(int)
            trading_signals = jm_xgb_df['Forecast_State'].shift(1).fillna(0)
            alloc_target = 1.0 - trading_signals
        elif config.allocation_style == "continuous":
            alloc_target = (1.0 - jm_xgb_df['State_Prob']).shift(1).fillna(1.0)
            
        strat_rets = (alloc_target * jm_xgb_df['Target_Return']) + ((1.0 - alloc_target) * jm_xgb_df['RF_Rate'])
        trades = alloc_target.diff().abs().fillna(0)
        jm_xgb_df['Strat_Return_Gross'] = strat_rets  # before TC — matches paper Table 4 reporting
        jm_xgb_df['Strat_Return'] = strat_rets - (trades * TRANSACTION_COST)

        # Trim to exact OOS window (per-asset effective start handles short histories)
        mask = (jm_xgb_df.index >= effective_oos_start) & (jm_xgb_df.index < oos_end_dt)
        jm_xgb_df = jm_xgb_df[mask]
        if jm_xgb_df.empty:
            continue

        # JM-XGB metrics (net of TC; also gross for apples-to-apples paper comparison)
        ret, vol, sharpe, sortino, mdd = calculate_metrics(jm_xgb_df['Strat_Return'], jm_xgb_df['RF_Rate'])
        ret_g, vol_g, sharpe_g, sortino_g, mdd_g = calculate_metrics(jm_xgb_df['Strat_Return_Gross'], jm_xgb_df['RF_Rate'])
        results.append({
            'Ticker': ticker, 'Period': period_name,
            'Strategy': 'JM-XGB', 'Ann_Ret': ret, 'Ann_Vol': vol,
            'Sharpe': sharpe, 'Sortino': sortino, 'Max_DD': mdd,
            'Sharpe_Gross': sharpe_g, 'Ann_Ret_Gross': ret_g, 'Max_DD_Gross': mdd_g,
            'EWMA_HL': best_ewma_hl,
        })

        # B&H metrics (no rebalancing, so net == gross)
        bh_ret, bh_vol, bh_sharpe, bh_sortino, bh_mdd = calculate_metrics(
            jm_xgb_df['Target_Return'], jm_xgb_df['RF_Rate'])
        results.append({
            'Ticker': ticker, 'Period': period_name,
            'Strategy': 'B&H', 'Ann_Ret': bh_ret, 'Ann_Vol': bh_vol,
            'Sharpe': bh_sharpe, 'Sortino': bh_sortino, 'Max_DD': bh_mdd,
            'Sharpe_Gross': bh_sharpe, 'Ann_Ret_Gross': bh_ret, 'Max_DD_Gross': bh_mdd,
            'EWMA_HL': best_ewma_hl,
        })

        if period_name == 'Full (2007-2025)':
            full_period_ts = jm_xgb_df[['State_Prob', 'Forecast_State', 'Target_Return', 'RF_Rate']].copy()

    return results, full_period_ts


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_markdown_report(results_df, elapsed, timestamp, asset_data, tickers, asset_classes, list_name, config, data_start, preset_name="Custom"):
    """Generate a markdown report with parameters and full results."""
    lines = []
    lines.append(f"# JM-XGB Multi-Asset Benchmark Report — {list_name}")
    lines.append(f"")
    lines.append(f"**Generated:** {timestamp}")
    lines.append(f"**Runtime:** {elapsed:.1f}s")
    lines.append(f"**Asset List:** {list_name}")
    lines.append(f"**Strategy Preset:** {preset_name}")
    lines.append(f"")

    # Data coverage
    lines.append(f"## Data Coverage")
    lines.append(f"")
    lines.append(f"| Ticker | Start | End | Rows |")
    lines.append(f"|---|---|---|---|")
    for ticker in tickers:
        if ticker in asset_data:
            df = asset_data[ticker]
            lines.append(f"| {ticker} | {df.index[0].date()} | {df.index[-1].date()} | {len(df)} |")
        else:
            lines.append(f"| {ticker} | N/A | N/A | N/A |")
    lines.append(f"")

    # DD exclusion note
    dd_excluded = [t for t in tickers if t in DD_EXCLUDE_TICKERS]
    dd_standard = [t for t in tickers if t not in DD_EXCLUDE_TICKERS]
    lines.append(f"## Feature Set")
    lines.append(f"")
    lines.append(f"**Paper Table 2 compliance:** Bonds and gold use 6 return features (Avg_Return×3, Sortino×3); "
                 f"equities use 8 features (DD_log×2, Avg_Return×3, Sortino×3).")
    lines.append(f"")
    if dd_excluded:
        lines.append(f"- **DD excluded (6 features):** {', '.join(dd_excluded)}")
    if dd_standard:
        lines.append(f"- **DD included (8 features):** {', '.join(dd_standard)}")
    lines.append(f"")

    # Aggregate summary
    full_data = results_df[results_df['Period'] == 'Full (2007-2025)']
    if not full_data.empty:
        jm_full = full_data[full_data['Strategy'] == 'JM-XGB'].dropna(subset=['Sharpe'])
        bh_full = full_data[full_data['Strategy'] == 'B&H'].dropna(subset=['Sharpe'])

        wins, total = 0, 0
        for ticker in tickers:
            j = jm_full[jm_full['Ticker'] == ticker]
            b = bh_full[bh_full['Ticker'] == ticker]
            if not j.empty and not b.empty:
                total += 1
                if j.iloc[0]['Sharpe'] > b.iloc[0]['Sharpe']:
                    wins += 1

        lines.append(f"## Aggregate Summary (Full Period)")
        lines.append(f"")
        if total > 0:
            lines.append(f"**JM-XGB beats B&H on Sharpe: {wins}/{total} assets ({wins/total*100:.0f}%)**")
            lines.append(f"")

        lines.append(f"| Asset | Ticker | JM-XGB Ret | JM-XGB Vol | JM-XGB | B&H Ret | B&H Vol | B&H | Sharpe Delta | JM-XGB MDD | B&H MDD | Verdict |")
        lines.append(f"|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for ticker in tickers:
            j = jm_full[jm_full['Ticker'] == ticker]
            b = bh_full[bh_full['Ticker'] == ticker]
            asset = TICKER_TO_PAPER_ASSET.get(ticker, ticker)
            if j.empty or b.empty:
                lines.append(f"| {asset} | {ticker} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
                continue
            js, bs = j.iloc[0]['Sharpe'], b.iloc[0]['Sharpe']
            jr, br = j.iloc[0]['Ann_Ret'], b.iloc[0]['Ann_Ret']
            jv, bv = j.iloc[0]['Ann_Vol'], b.iloc[0]['Ann_Vol']
            delta = js - bs
            jmdd, bmdd = j.iloc[0]['Max_DD'], b.iloc[0]['Max_DD']
            verdict = 'WIN' if js > bs else 'LOSE'
            lines.append(f"| {asset} | {ticker} | {jr*100:.1f}% | {jv*100:.1f}% | {js:.2f} | {br*100:.1f}% | {bv*100:.1f}% | {bs:.2f} | {delta:+.2f} | {jmdd*100:.1f}% | {bmdd*100:.1f}% | **{verdict}** |")

        if not jm_full.empty and not bh_full.empty:
            avg_j = jm_full['Sharpe'].mean()
            avg_b = bh_full['Sharpe'].mean()
            avg_jr = jm_full['Ann_Ret'].mean()
            avg_br = bh_full['Ann_Ret'].mean()
            avg_jv = jm_full['Ann_Vol'].mean()
            avg_bv = bh_full['Ann_Vol'].mean()
            avg_jmdd = jm_full['Max_DD'].mean()
            avg_bmdd = bh_full['Max_DD'].mean()
            lines.append(f"| | **AVG** | **{avg_jr*100:.1f}%** | **{avg_jv*100:.1f}%** | **{avg_j:.2f}** | **{avg_br*100:.1f}%** | **{avg_bv*100:.1f}%** | **{avg_b:.2f}** | **{avg_j-avg_b:+.2f}** | **{avg_jmdd*100:.1f}%** | **{avg_bmdd*100:.1f}%** | |")
        lines.append(f"")

        # Asset class summary
        lines.append(f"### Asset Class Averages (Full Period Sharpe)")
        lines.append(f"")
        lines.append(f"| Asset Class | JM-XGB Avg | B&H Avg | Delta |")
        lines.append(f"|---|---:|---:|---:|")
        for cls_name, cls_tickers in asset_classes.items():
            j_sharpes, b_sharpes = [], []
            for t in cls_tickers:
                j = full_data[(full_data['Ticker'] == t) & (full_data['Strategy'] == 'JM-XGB')]
                b = full_data[(full_data['Ticker'] == t) & (full_data['Strategy'] == 'B&H')]
                if not j.empty and not b.empty and not np.isnan(j.iloc[0]['Sharpe']):
                    j_sharpes.append(j.iloc[0]['Sharpe'])
                    b_sharpes.append(b.iloc[0]['Sharpe'])
            if j_sharpes:
                lines.append(f"| {cls_name} | {np.mean(j_sharpes):.2f} | {np.mean(b_sharpes):.2f} | {np.mean(j_sharpes)-np.mean(b_sharpes):+.2f} |")
            else:
                lines.append(f"| {cls_name} | N/A | N/A | N/A |")
        lines.append(f"")

    # vs. Paper Table 4 comparison — use the paper-matched window for an apples-to-apples gap
    paper_window_data = results_df[results_df['Period'] == 'Full (Paper 2007-2023)']
    mapped_tickers = [(t, TICKER_TO_PAPER_ASSET[t]) for t in tickers if t in TICKER_TO_PAPER_ASSET]
    if mapped_tickers and not paper_window_data.empty:
        jm_full = paper_window_data[paper_window_data['Strategy'] == 'JM-XGB']
        bh_full = paper_window_data[paper_window_data['Strategy'] == 'B&H']

        lines.append(f"## vs. Paper Table 4 (2007–2023, Bloomberg, gross of TC)")
        lines.append(f"")
        lines.append(f"> Paper Table 4 reports Sharpe ratios **gross of transaction costs** (Session 19 finding). "
                     f"Both `Ours (gross)` and `Paper` are gross-of-TC and directly comparable. "
                     f"`Ours (net)` applies 5 bps per one-way trade — what an investor would actually earn.")
        lines.append(f"")

        lines.append(f"### JM-XGB Sharpe")
        lines.append(f"")
        lines.append(f"| Asset | Ticker | Paper | Ours (gross) | Gap (gross) | Ours (net) | Gap (net) |")
        lines.append(f"|---|---|---:|---:|---:|---:|---:|")
        gaps_gross, gaps_net = [], []
        for ticker, asset_name in mapped_tickers:
            paper = PAPER_TABLE4[asset_name]
            row = jm_full[jm_full['Ticker'] == ticker]
            if row.empty or np.isnan(row.iloc[0]['Sharpe']):
                lines.append(f"| {asset_name} | {ticker} | {paper['jmxgb_sharpe']:.2f} | N/A | N/A | N/A | N/A |")
            else:
                ours_g = row.iloc[0]['Sharpe_Gross']
                ours_n = row.iloc[0]['Sharpe']
                gap_g = ours_g - paper['jmxgb_sharpe']
                gap_n = ours_n - paper['jmxgb_sharpe']
                gaps_gross.append(gap_g)
                gaps_net.append(gap_n)
                sign_g = '+' if gap_g >= 0 else ''
                sign_n = '+' if gap_n >= 0 else ''
                lines.append(f"| {asset_name} | {ticker} | {paper['jmxgb_sharpe']:.2f} | {ours_g:.2f} | {sign_g}{gap_g:.2f} | {ours_n:.2f} | {sign_n}{gap_n:.2f} |")
        if gaps_gross:
            avg_g = np.mean(gaps_gross)
            avg_n = np.mean(gaps_net)
            sg = '+' if avg_g >= 0 else ''
            sn = '+' if avg_n >= 0 else ''
            lines.append(f"| **AVG** | | | | **{sg}{avg_g:.2f}** | | **{sn}{avg_n:.2f}** |")
        lines.append(f"")

        lines.append(f"### B&H Sharpe")
        lines.append(f"")
        lines.append(f"> B&H does not rebalance, so net == gross.")
        lines.append(f"")
        lines.append(f"| Asset | Ticker | Paper | Ours | Gap |")
        lines.append(f"|---|---|---:|---:|---:|")
        for ticker, asset_name in mapped_tickers:
            paper = PAPER_TABLE4[asset_name]
            row = bh_full[bh_full['Ticker'] == ticker]
            if row.empty or np.isnan(row.iloc[0]['Sharpe']):
                lines.append(f"| {asset_name} | {ticker} | {paper['bh_sharpe']:.2f} | N/A | N/A |")
            else:
                ours = row.iloc[0]['Sharpe']
                gap = ours - paper['bh_sharpe']
                sign = '+' if gap >= 0 else ''
                lines.append(f"| {asset_name} | {ticker} | {paper['bh_sharpe']:.2f} | {ours:.2f} | {sign}{gap:.2f} |")
        lines.append(f"")

        lines.append(f"### JM-XGB Max Drawdown")
        lines.append(f"")
        lines.append(f"| Asset | Ticker | Paper | Ours | Gap |")
        lines.append(f"|---|---|---:|---:|---:|")
        for ticker, asset_name in mapped_tickers:
            paper = PAPER_TABLE4[asset_name]
            row = jm_full[jm_full['Ticker'] == ticker]
            if row.empty or np.isnan(row.iloc[0]['Max_DD']):
                lines.append(f"| {asset_name} | {ticker} | {paper['jmxgb_mdd']*100:.1f}% | N/A | N/A |")
            else:
                ours = row.iloc[0]['Max_DD']
                gap = ours - paper['jmxgb_mdd']
                sign = '+' if gap >= 0 else ''
                lines.append(f"| {asset_name} | {ticker} | {paper['jmxgb_mdd']*100:.1f}% | {ours*100:.1f}% | {sign}{gap*100:.1f}pp |")
        lines.append(f"")

    # Per-period tables
    for period_name, _, _ in TIME_PERIODS:
        period_data = results_df[results_df['Period'] == period_name]
        if period_data.empty:
            continue

        lines.append(f"## {period_name}")
        lines.append(f"")
        lines.append(f"| Ticker | JM-XGB Sharpe | B&H Sharpe | Delta | JM-XGB Ret | B&H Ret | JM-XGB MDD | B&H MDD |")
        lines.append(f"|---|---:|---:|---:|---:|---:|---:|---:|")

        for ticker in tickers:
            jmxgb = period_data[(period_data['Ticker'] == ticker) & (period_data['Strategy'] == 'JM-XGB')]
            bh = period_data[(period_data['Ticker'] == ticker) & (period_data['Strategy'] == 'B&H')]
            if jmxgb.empty or bh.empty:
                continue
            j = jmxgb.iloc[0]
            b = bh.iloc[0]
            if np.isnan(j['Sharpe']):
                lines.append(f"| {ticker} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
                continue
            delta = j['Sharpe'] - b['Sharpe']
            lines.append(f"| {ticker} | {j['Sharpe']:.2f} | {b['Sharpe']:.2f} | {delta:+.2f} | {j['Ann_Ret']*100:.1f}% | {b['Ann_Ret']*100:.1f}% | {j['Max_DD']*100:.1f}% | {b['Max_DD']*100:.1f}% |")
        lines.append(f"")

    # Parameters
    lines.append(f"## Strategy Configuration (`{config.name}`)")
    lines.append(f"")
    lines.append(f"### Data")
    lines.append(f"")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Asset list | {list_name} |")
    lines.append(f"| Data start | {data_start} |")
    lines.append(f"| OOS periods | {TIME_PERIODS[0][1]} → {TIME_PERIODS[-1][2]} |")
    lines.append(f"| Transaction cost | {TRANSACTION_COST:.4f} ({TRANSACTION_COST*10000:.1f} bps) |")
    lines.append(f"")
    lines.append(f"### Jump Model / Walk-Forward")
    lines.append(f"")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Allocation style | {config.allocation_style} |")
    lines.append(f"| Tuning metric | {config.tuning_metric} |")
    lines.append(f"| Prob threshold | {config.prob_threshold} |")
    lines.append(f"| Validation window | {config.validation_window_type} ({VALIDATION_WINDOW_YRS} yrs) |")
    lines.append(f"| Lambda smoothing | {config.lambda_smoothing} |")
    lines.append(f"| Lambda ensemble K | {config.lambda_ensemble_k} |")
    lines.append(f"| Lambda selection | {config.lambda_selection} |")
    lines.append(f"| Sub-window consensus | {config.lambda_subwindow_consensus} |")
    lines.append(f"| Lambda grid | {LAMBDA_GRID} |")
    lines.append(f"| EWMA mode | {config.ewma_mode} |")
    lines.append(f"")
    lines.append(f"### XGBoost Hyperparameters")
    lines.append(f"")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|---|---|")
    xgb_params = config.xgb_params or {}
    lines.append(f"| max_depth | {xgb_params.get('max_depth', 6)} |")
    lines.append(f"| n_estimators | {xgb_params.get('n_estimators', 100)} |")
    lines.append(f"| learning_rate | {xgb_params.get('learning_rate', 0.3)} |")
    lines.append(f"| subsample | {xgb_params.get('subsample', 1.0)} |")
    lines.append(f"| colsample_bytree | {xgb_params.get('colsample_bytree', 1.0)} |")
    lines.append(f"| reg_alpha (L1) | {xgb_params.get('reg_alpha', 0.0)} |")
    lines.append(f"| reg_lambda (L2) | {xgb_params.get('reg_lambda', 1.0)} |")
    lines.append(f"")
    lines.append(f"### EWMA Halflife per Asset")
    lines.append(f"")
    # Extract EWMA HL from Full period results (most representative)
    full_jm = results_df[(results_df['Period'] == 'Full (2007-2025)') & (results_df['Strategy'] == 'JM-XGB')]
    if 'EWMA_HL' in results_df.columns and not full_jm.empty:
        lines.append(f"| Ticker | EWMA HL | Source |")
        lines.append(f"|---|---:|---|")
        for ticker in tickers:
            row = full_jm[full_jm['Ticker'] == ticker]
            if not row.empty and not pd.isna(row.iloc[0].get('EWMA_HL')):
                hl = int(row.iloc[0]['EWMA_HL'])
                source = "paper" if (config.ewma_mode == "paper" and ticker in PAPER_EWMA_HL) else "auto-tuned"
                lines.append(f"| {ticker} | {hl} | {source} |")
            else:
                lines.append(f"| {ticker} | N/A | — |")
    else:
        lines.append(f"*(EWMA HL data not available — re-run benchmark to populate)*")
    lines.append(f"")

    return '\n'.join(lines)


def main():
    # Load asset lists and parse CLI
    all_lists = load_asset_lists()
    selected_name = parse_asset_list_selection(sys.argv[1:], all_lists)
    if selected_name is None:
        sys.exit(0)

    selected = all_lists[selected_name]
    tickers = selected['tickers']
    asset_classes = selected['asset_classes']
    data_start = selected['data_start']

    t0 = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Fetch all data (sequential — yfinance doesn't parallelize well)
    print("=" * 80)
    print(f"  MULTI-ASSET JM-XGB BENCHMARK  [{selected_name}]")
    print("=" * 80)
    print(f"\nFetching data for {len(tickers)} assets (data_start={data_start})...")
    asset_data = {}
    for ticker in tickers:
        print(f"  Fetching {ticker}...", end=" ", flush=True)
        _, df = fetch_etf_data(ticker, data_start=data_start)
        if df is not None:
            asset_data[ticker] = df
            print(f"OK ({len(df)} rows, {df.index[0].date()} to {df.index[-1].date()})")
        else:
            print("FAILED")

    print(f"\nSuccessfully loaded {len(asset_data)}/{len(tickers)} assets")
    if not asset_data:
        print("No data available. Exiting.")
        return

    # 2. Run backtests (parallel across assets)
    print(f"\nRunning walk-forward backtests across {len(TIME_PERIODS)} time periods...")
    print(f"Using {min(cpu_count(), len(asset_data))} parallel workers\n")

    # Build config from env vars (honors XGB_EWMA_MODE, XGB_LAMBDA_SUBWINDOW_CONSENSUS, etc.
    # set by Diagnostics Launcher or CLI). Missing env vars fall back to StrategyConfig defaults.
    config = StrategyConfig(**_strategy_config_from_env())
    preset_name = os.environ.get('XGB_PRESET_NAME', 'Custom')
    args_list = [(ticker, df, config, data_start) for ticker, df in asset_data.items()]

    all_results = []
    all_timeseries = {}  # ticker -> full-period DataFrame (State_Prob, Forecast_State, Target_Return, RF_Rate)
    n_workers = min(cpu_count(), len(args_list))
    if n_workers > 1:
        with Pool(n_workers) as pool:
            for i, (result_list, ts_df) in enumerate(pool.imap_unordered(backtest_single_asset, args_list)):
                ticker = result_list[0]['Ticker'] if result_list else '?'
                print(f"  Completed {ticker} ({i+1}/{len(args_list)})")
                all_results.extend(result_list)
                if ts_df is not None:
                    all_timeseries[ticker] = ts_df
    else:
        for args in args_list:
            result_list, ts_df = backtest_single_asset(args)
            ticker = result_list[0]['Ticker'] if result_list else '?'
            print(f"  Completed {ticker}")
            all_results.extend(result_list)
            if ts_df is not None:
                all_timeseries[ticker] = ts_df

    elapsed = time.time() - t0
    print(f"\nBacktest completed in {elapsed:.1f}s")

    # 3. Build results DataFrame and save
    results_df = pd.DataFrame(all_results)

    csv_path = os.path.join(BENCHMARKS_DIR, f'benchmark_results_{timestamp}.csv')
    results_df.to_csv(csv_path, index=False)

    # Save per-ticker full-period regime timeseries for Diagnostics Launcher charts
    if all_timeseries:
        import pickle as _pickle
        regimes_path = os.path.join(BENCHMARKS_DIR, f'benchmark_regimes_{timestamp}.pkl')
        with open(regimes_path, 'wb') as f:
            _pickle.dump({'timeseries': all_timeseries, 'asset_names': TICKER_TO_PAPER_ASSET, 'list_name': selected_name}, f)

    # 4. Generate and save markdown report
    md_content = generate_markdown_report(results_df, elapsed, timestamp, asset_data, tickers, asset_classes, selected_name, config, data_start, preset_name)
    md_path = os.path.join(BENCHMARKS_DIR, f'benchmark_report_{timestamp}.md')
    with open(md_path, 'w') as f:
        f.write(md_content)

    # 5. Print summary to console
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)

    for period_name, _, _ in TIME_PERIODS:
        period_data = results_df[results_df['Period'] == period_name]
        if period_data.empty:
            continue

        print(f"\n{'─' * 80}")
        print(f"  {period_name}")
        print(f"{'─' * 80}")
        print(f"  {'Ticker':<8} │ {'JM-XGB Sharpe':>13} │ {'B&H Sharpe':>10} │ {'Delta':>7} │ {'JM-XGB Ret':>10} │ {'B&H Ret':>9} │ {'JM-XGB MDD':>10} │ {'B&H MDD':>9}")
        print(f"  {'─'*8}─┼─{'─'*13}─┼─{'─'*10}─┼─{'─'*7}─┼─{'─'*10}─┼─{'─'*9}─┼─{'─'*10}─┼─{'─'*9}")

        for ticker in tickers:
            jmxgb = period_data[(period_data['Ticker'] == ticker) & (period_data['Strategy'] == 'JM-XGB')]
            bh = period_data[(period_data['Ticker'] == ticker) & (period_data['Strategy'] == 'B&H')]
            if jmxgb.empty or bh.empty:
                continue
            j = jmxgb.iloc[0]
            b = bh.iloc[0]
            if np.isnan(j['Sharpe']):
                print(f"  {ticker:<8} │ {'N/A':>13} │ {'N/A':>10} │ {'N/A':>7} │ {'N/A':>10} │ {'N/A':>9} │ {'N/A':>10} │ {'N/A':>9}")
                continue
            delta = j['Sharpe'] - b['Sharpe']
            marker = '+' if delta > 0 else ''
            print(f"  {ticker:<8} │ {j['Sharpe']:>13.2f} │ {b['Sharpe']:>10.2f} │ {marker}{delta:>6.2f} │ {j['Ann_Ret']*100:>9.1f}% │ {b['Ann_Ret']*100:>8.1f}% │ {j['Max_DD']*100:>9.1f}% │ {b['Max_DD']*100:>8.1f}%")

    # Aggregate
    full_data = results_df[results_df['Period'] == 'Full (2007-2025)']
    if not full_data.empty:
        jm_full = full_data[full_data['Strategy'] == 'JM-XGB'].dropna(subset=['Sharpe'])
        bh_full = full_data[full_data['Strategy'] == 'B&H'].dropna(subset=['Sharpe'])

        wins, total = 0, 0
        for ticker in tickers:
            j = jm_full[jm_full['Ticker'] == ticker]
            b = bh_full[bh_full['Ticker'] == ticker]
            if not j.empty and not b.empty:
                total += 1
                if j.iloc[0]['Sharpe'] > b.iloc[0]['Sharpe']:
                    wins += 1

        print(f"\n{'=' * 80}")
        print("  AGGREGATE SUMMARY (Full Period: 2007-2025)")
        print(f"{'=' * 80}")
        if total > 0:
            print(f"\n  JM-XGB beats B&H on Sharpe: {wins}/{total} assets ({wins/total*100:.0f}%)")

        print(f"\n  {'Asset':<10} {'Ticker':<10} │ {'JM Ret':>7} {'JM Vol':>7} {'JM-XGB':>8} │ {'BH Ret':>7} {'BH Vol':>7} {'B&H':>8} │ {'Sharpe Δ':>9} │ {'JM MDD':>7} {'BH MDD':>7} │ {'Verdict':>8}")
        sep = f"  {'─'*10} {'─'*10}─┼─{'─'*7} {'─'*7} {'─'*8}─┼─{'─'*7} {'─'*7} {'─'*8}─┼─{'─'*9}─┼─{'─'*7} {'─'*7}─┼─{'─'*8}"
        print(sep)
        for ticker in tickers:
            j = jm_full[jm_full['Ticker'] == ticker]
            b = bh_full[bh_full['Ticker'] == ticker]
            asset = TICKER_TO_PAPER_ASSET.get(ticker, ticker)
            if j.empty or b.empty:
                print(f"  {asset:<10} {ticker:<10} │ {'N/A':>7} {'N/A':>7} {'N/A':>8} │ {'N/A':>7} {'N/A':>7} {'N/A':>8} │ {'N/A':>9} │ {'N/A':>7} {'N/A':>7} │ {'N/A':>8}")
                continue
            js, bs = j.iloc[0]['Sharpe'], b.iloc[0]['Sharpe']
            jr, br = j.iloc[0]['Ann_Ret'], b.iloc[0]['Ann_Ret']
            jv, bv = j.iloc[0]['Ann_Vol'], b.iloc[0]['Ann_Vol']
            delta = js - bs
            jmdd, bmdd = j.iloc[0]['Max_DD'], b.iloc[0]['Max_DD']
            verdict = 'WIN' if js > bs else 'LOSE'
            print(f"  {asset:<10} {ticker:<10} │ {jr*100:>6.1f}% {jv*100:>6.1f}% {js:>8.2f} │ {br*100:>6.1f}% {bv*100:>6.1f}% {bs:>8.2f} │ {delta:>+9.2f} │ {jmdd*100:>6.1f}% {bmdd*100:>6.1f}% │ {verdict:>8}")

        if not jm_full.empty and not bh_full.empty:
            print(sep)
            avg_j = jm_full['Sharpe'].mean()
            avg_b = bh_full['Sharpe'].mean()
            avg_jr = jm_full['Ann_Ret'].mean()
            avg_br = bh_full['Ann_Ret'].mean()
            avg_jv = jm_full['Ann_Vol'].mean()
            avg_bv = bh_full['Ann_Vol'].mean()
            avg_jmdd = jm_full['Max_DD'].mean()
            avg_bmdd = bh_full['Max_DD'].mean()
            print(f"  {'':10} {'AVG':<10} │ {avg_jr*100:>6.1f}% {avg_jv*100:>6.1f}% {avg_j:>8.2f} │ {avg_br*100:>6.1f}% {avg_bv*100:>6.1f}% {avg_b:>8.2f} │ {avg_j-avg_b:>+9.2f} │ {avg_jmdd*100:>6.1f}% {avg_bmdd*100:>6.1f}% │ {'':>8}")

    print(f"\nOutputs saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  Report: {md_path}")
    print(f"Total runtime: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
