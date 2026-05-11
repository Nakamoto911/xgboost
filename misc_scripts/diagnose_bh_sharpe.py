"""
B&H Sharpe gap decomposition vs paper Table 4.

Isolates two mechanical sources of the small B&H Sharpe gap reported in
benchmark_assets.py vs paper Table 4 (which is 2007–2023, FRED DTB3 RF):
  1. Period: ours = 2007-01-01 → 2026-01-01 (Full). Paper = 2007-2023.
  2. Risk-free rate: ours = Yahoo ^IRX (discount yield). Paper = FRED DTB3 (BEY).

For each Bloomberg asset, computes B&H Sharpe in 4 configurations and
shows the gap to paper Table 4. No strategy backtest — just B&H.
"""

import sys, os, types
try:
    import distutils, distutils.version
except ImportError:
    d = types.ModuleType('distutils'); dv = types.ModuleType('distutils.version')
    class LV:
        def __init__(self, v=None): self.v = v
        def __lt__(self, o): return self.v < (o.v if hasattr(o, 'v') else o)
        def __eq__(self, o): return self.v == (o.v if hasattr(o, 'v') else o)
    dv.LooseVersion = LV; d.version = dv
    sys.modules['distutils'] = d; sys.modules['distutils.version'] = dv

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

BBG_PATH = os.path.join(PROJECT_ROOT, 'cache', 'DATA PAUL.xlsx')
BBG_COLS = ['Date', 'SPTR', 'SPTRMDCP', 'RU20INTR', 'NDDUEAFE', 'NDUEEGF',
            'LBUSTRUU', 'IBOXHY', 'LUACTRUU', 'DJUSRET', 'DBLCDBCE', 'GOLDLNPM', 'LUTLTRUU']

PAPER_BH = {
    'SPTR': ('LargeCap', 0.50), 'SPTRMDCP': ('MidCap', 0.45),
    'RU20INTR': ('SmallCap', 0.36), 'NDDUEAFE': ('EAFE', 0.20),
    'NDUEEGF': ('EM', 0.20), 'LBUSTRUU': ('AggBond', 0.46),
    'LUTLTRUU': ('Treasury', 0.26), 'IBOXHY': ('HighYield', 0.67),
    'LUACTRUU': ('Corporate', 0.54), 'DJUSRET': ('REIT', 0.27),
    'DBLCDBCE': ('Commodity', 0.03), 'GOLDLNPM': ('Gold', 0.43),
}

def load_bbg():
    df = pd.read_excel(BBG_PATH, header=None, skiprows=6)
    df.columns = BBG_COLS
    df['Date'] = pd.to_datetime(df['Date'])
    return df.set_index('Date').sort_index()

def fetch_irx(start, end):
    df = yf.download('^IRX', start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        s = df['Adj Close'].iloc[:, 0]
    else:
        s = df.get('Adj Close', df.get('Close', df.iloc[:, 0]))
    s.name = 'IRX'
    return s

def fetch_dtb3(start, end):
    return web.DataReader('DTB3', 'fred', start, end)['DTB3'].rename('DTB3')

def bh_sharpe(prices, rf_annual_pct, start, end):
    """B&H Sharpe over [start, end]. rf is annual % yield; convert to daily."""
    p = prices[(prices.index >= start) & (prices.index < end)]
    ret = p.pct_change().fillna(0)
    rf = (rf_annual_pct.reindex(ret.index).ffill() / 100.0) / 252.0
    excess = ret - rf.fillna(0)
    if len(ret) < 10 or ret.std() == 0:
        return np.nan
    return (excess.mean() * 252) / (ret.std() * np.sqrt(252))

def main():
    print("Loading Bloomberg prices...")
    bbg = load_bbg()
    print(f"  Loaded {len(bbg)} rows, {bbg.index[0].date()} → {bbg.index[-1].date()}")

    print("Fetching Yahoo ^IRX...")
    irx = fetch_irx('2006-01-01', '2026-05-01')
    print(f"  Loaded {len(irx)} obs, {irx.index[0].date()} → {irx.index[-1].date()}")

    print("Fetching FRED DTB3...")
    dtb3 = fetch_dtb3('2006-01-01', '2026-05-01')
    print(f"  Loaded {len(dtb3)} obs, {dtb3.index[0].date()} → {dtb3.index[-1].date()}")

    # Compare RF series themselves
    overlap = irx.reindex(dtb3.index).dropna()
    common = pd.concat([overlap.rename('IRX'), dtb3.rename('DTB3')], axis=1).dropna()
    rf_diff_bps = (common['IRX'] - common['DTB3']).mean() * 100
    rf_corr = common['IRX'].corr(common['DTB3'])
    print(f"\n^IRX vs DTB3 (2006-2026 overlap, {len(common)} obs):")
    print(f"  Mean diff: {rf_diff_bps:+.2f} bps   |   Correlation: {rf_corr:.4f}")

    configs = [
        ('IRX  2007-2025', irx,  '2007-01-01', '2026-01-01'),
        ('IRX  2007-2023', irx,  '2007-01-01', '2024-01-01'),
        ('DTB3 2007-2025', dtb3, '2007-01-01', '2026-01-01'),
        ('DTB3 2007-2023', dtb3, '2007-01-01', '2024-01-01'),
    ]

    rows = []
    for ticker, (name, paper) in PAPER_BH.items():
        prices = bbg[ticker].dropna()
        rec = {'Asset': name, 'Ticker': ticker, 'Paper': paper}
        for label, rf, s, e in configs:
            sh = bh_sharpe(prices, rf, s, e)
            rec[label] = sh
        rows.append(rec)
    df = pd.DataFrame(rows)

    # Print table
    print("\n" + "=" * 110)
    print(f"{'Asset':<11}{'Ticker':<11}{'Paper':>7}  "
          f"{'IRX 07-25':>10}{'gap':>7}  "
          f"{'IRX 07-23':>10}{'gap':>7}  "
          f"{'DTB3 07-25':>11}{'gap':>7}  "
          f"{'DTB3 07-23':>11}{'gap':>7}")
    print("=" * 110)
    gaps = {c[0]: [] for c in configs}
    for _, r in df.iterrows():
        line = f"{r['Asset']:<11}{r['Ticker']:<11}{r['Paper']:>7.2f}  "
        for label, *_ in configs:
            v = r[label]
            g = v - r['Paper']
            gaps[label].append(g)
            spacer = '>10' if 'IRX' in label else '>11'
            line += f"{v:{spacer}.3f}{g:>+7.2f}  "
        print(line)
    print("-" * 110)
    avg_line = f"{'AVG |gap|':<22}{'':>7}  "
    for label, *_ in configs:
        mag = np.mean(np.abs(gaps[label]))
        bias = np.mean(gaps[label])
        spacer = '>10' if 'IRX' in label else '>11'
        avg_line += f"{'':{spacer}}{mag:>+7.3f}  "
    print(avg_line)
    bias_line = f"{'AVG bias':<22}{'':>7}  "
    for label, *_ in configs:
        bias = np.mean(gaps[label])
        spacer = '>10' if 'IRX' in label else '>11'
        bias_line += f"{'':{spacer}}{bias:>+7.3f}  "
    print(bias_line)
    print("=" * 110)
    print("\nInterpretation:")
    print("  - IRX 07-25 reproduces the published report (should match within rounding).")
    print("  - IRX 07-23 isolates the period fix alone.")
    print("  - DTB3 07-25 isolates the RF-source fix alone.")
    print("  - DTB3 07-23 applies both fixes — expected to be closest to paper.")

if __name__ == '__main__':
    main()
