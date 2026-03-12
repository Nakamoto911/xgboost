import yfinance as yf
print(yf.Ticker('^TNX').history(period='5d'))
