import yfinance as yf
print("Testing single download...")
try:
    df = yf.download("^GSPC", start="2020-01-01", end="2020-01-10", auto_adjust=False)
    print("Success:")
    print(df.head())
except Exception as e:
    print("Failed:", e)
