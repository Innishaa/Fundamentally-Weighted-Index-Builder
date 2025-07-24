import yfinance as yf
import pandas as pd

def download_price_data(tickers, start, end):
    print(f"Downloading data for: {tickers}")
    data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=False)
    # Handle multi and single ticker differently
    if isinstance(data.columns, pd.MultiIndex):
        try:
            close = data.loc[:, (slice(None), 'Close')]
            close.columns = close.columns.droplevel(1)  # Drop the 'Close' level
        except KeyError:
            raise KeyError("'Close' prices not found for one or more tickers in MultiIndex DataFrame.")
    else:
        if 'Close' not in data.columns:
            raise KeyError("'Close' price not found in DataFrame.")
        close = data[['Close']]
        close.columns = tickers  # Set column name to ticker
    return close
