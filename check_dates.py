import yfinance as yf
import pandas_datareader.data as web
import pandas as pd

try:
    yf_data = yf.download('^SP500TR', start='2025-12-01', end='2026-03-09')
    yf_date = yf_data.index.max()
    print("YF ^SP500TR max date:", yf_date)
except Exception as e:
    print("YF ^SP500TR error:", e)

try:
    import time
    fred_data = None
    for attempt in range(5):
        try:
            fred_data = web.DataReader(['DGS2', 'DGS10'], 'fred', '1990-01-01', '2026-03-09')
            break
        except Exception as e:
            print(f"Failed to fetch FRED data (attempt {attempt+1}/5): {e}")
            time.sleep(2)
    if fred_data is None:
        raise ValueError("Failed to download FRED data after 5 attempts.")
    fred_date = fred_data.index.max()
    print("FRED max date:", fred_date)
except Exception as e:
    print("FRED error:", e)

