# Asset Lists for benchmark_assets.py

## Yahoo ETFs

data_start: 1993-01-01

| Ticker | Asset Class  | Description                      |
|--------|--------------|----------------------------------|
| IVV    | US Equity    | iShares Core S&P 500             |
| IJH    | US Equity    | iShares Core S&P Mid-Cap 400     |
| IWM    | US Equity    | iShares Russell 2000             |
| EFA    | Intl Equity  | iShares MSCI EAFE                |
| EEM    | Intl Equity  | iShares MSCI Emerging Markets    |
| AGG    | Fixed Income | iShares Core US Aggregate Bond   |
| SPTL   | Fixed Income | SPDR Portfolio Long Term Treasury|
| HYG    | Fixed Income | iShares iBoxx High Yield Corp    |
| SPBO   | Fixed Income | SPDR Portfolio Corporate Bond    |
| IYR    | Real Assets  | iShares US Real Estate           |
| DBC    | Real Assets  | Invesco DB Commodity Tracking    |
| GLD    | Real Assets  | SPDR Gold Shares                 |

## Yahoo Mutual Funds

data_start: 1975-01-01

| Ticker   | Asset Class  | Description                        |
|----------|--------------|------------------------------------|
| ^SP500TR | US Equity    | S&P 500 Total Return (1988)        |
| VIMSX    | US Equity    | Vanguard Mid-Cap Index (1998)      |
| NAESX    | US Equity    | Vanguard Small-Cap Index (1980)    |
| FDIVX    | Intl Equity  | Fidelity Diversified Intl (1981)   |
| VEIEX    | Intl Equity  | Vanguard Emerging Markets (1994)   |
| VBMFX    | Fixed Income | Vanguard Total Bond Market (1986)  |
| VUSTX    | Fixed Income | Vanguard Long-Term Treasury (1986) |
| VWEHX    | Fixed Income | Vanguard High-Yield Corp (1980)    |
| VWESX    | Fixed Income | Vanguard Long-Term Corp (1980)     |
| FRESX    | Real Assets  | Fidelity Real Estate (1986)        |
| ^SPGSCI  | Real Assets  | S&P GSCI Commodity Index (1984)    |
| GC=F     | Real Assets  | Gold Futures (2000)                |

## Bloomberg Indices

data_start: 1989-01-01

| Ticker   | Asset Class  | Description                                    |
|----------|--------------|------------------------------------------------|
| SPTR     | US Equity    | S&P 500 Total Return (Bloomberg)               |
| SPTRMDCP | US Equity    | S&P MidCap 400 Total Return (Bloomberg)        |
| RU20INTR | US Equity    | Russell 2000 Total Return (Bloomberg)          |
| NDDUEAFE | Intl Equity  | MSCI EAFE Net Total Return USD (Bloomberg)     |
| NDUEEGF  | Intl Equity  | MSCI Emerging Net Total Return USD (Bloomberg) |
| LBUSTRUU | Fixed Income | Bloomberg US Aggregate Total Return (Bloomberg)|
| LUTLTRUU | Fixed Income | Bloomberg US Long Treasury Total Return (BBG)  |
| IBOXHY   | Fixed Income | iBoxx USD Liquid High Yield (Bloomberg)        |
| LUACTRUU | Fixed Income | Bloomberg US Corporate Total Return (Bloomberg)|
| DJUSRET  | Real Assets  | Dow Jones US Real Estate Total Return (BBG)    |
| DBLCDBCE | Real Assets  | Deutsche Bank DBIQ Optimum Yield Commodity (BBG)|
| GOLDLNPM | Real Assets  | LBMA Gold Price PM USD (Bloomberg)             |

## BBG+Yahoo ETF Hybrid

data_start: 1989-01-01

Splices Bloomberg total-return index history (pre-ETF inception) with the Yahoo Finance ETF return series (post-inception). Goal: paper-accurate JM lookback + investable OOS returns. Splice is in return space, so cumulative price series stays continuous. Ticker format `<BBG>+<YETF>`; the BBG leg drives history before ETF inception, the ETF leg drives history after.

| Ticker            | Asset Class  | Description                                            |
|-------------------|--------------|--------------------------------------------------------|
| SPTR+IVV          | US Equity    | SPTR pre-2000-05, then IVV (S&P 500)                   |
| SPTRMDCP+IJH      | US Equity    | SPTRMDCP pre-2000-05, then IJH (S&P MidCap 400)        |
| RU20INTR+IWM      | US Equity    | RU20INTR pre-2000-05, then IWM (Russell 2000)          |
| NDDUEAFE+EFA      | Intl Equity  | NDDUEAFE pre-2001-08, then EFA (MSCI EAFE)             |
| NDUEEGF+EEM       | Intl Equity  | NDUEEGF pre-2003-04, then EEM (MSCI EM)                |
| LBUSTRUU+AGG      | Fixed Income | LBUSTRUU pre-2003-09, then AGG (US Aggregate Bond)     |
| LUTLTRUU+SPTL     | Fixed Income | LUTLTRUU pre-2007-05, then SPTL (Long Treasury)        |
| IBOXHY+HYG        | Fixed Income | IBOXHY pre-2007-04, then HYG (HY Corporate)            |
| LUACTRUU+SPBO     | Fixed Income | LUACTRUU pre-2011-04, then SPBO (IG Corporate)         |
| DJUSRET+IYR       | Real Assets  | DJUSRET pre-2000-06, then IYR (US Real Estate)         |
| DBLCDBCE+DBC      | Real Assets  | DBLCDBCE pre-2006-02, then DBC (DB Commodity)          |
| GOLDLNPM+GLD      | Real Assets  | GOLDLNPM pre-2004-11, then GLD (SPDR Gold)             |
