import yfinance as yf
data = yf.download(["US2Y=X", "US10Y=X", "US2Y"], period="5d")
print(data.head())
