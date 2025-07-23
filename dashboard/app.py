import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from src.optimizer import optimize_weights
from src.backtester import backtest_portfolio, calculate_performance,calculate_turnover, plot_returns
from src.data_utils import compute_fundamental_scores, get_sector_matrix

st.set_page_config(layout="wide")
st.title("Fundamental Index Strategy (with Constraints)")

data_source = st.selectbox("Select Data Source", ["Use default CSV Data", "Use yfinance", "Use Financial Modeling Prep API", "Upload Custom CSV/Excel"])
weighing_strategy = st.selectbox("Select Weighting Strategy", ["Equal Weight", "Market Cap Weight", "Free Float Market Cap", "Score Based", "Mean-Variance Optimized"])

prices = None  # initialize
fundamentals= None  # initialize
index500 = None  # initialize
if data_source == "Use default CSV Data":
    try:
        # Load price data in wide format (Date + multiple ticker columns)
        prices_wide = pd.read_csv("data/price_data.csv", parse_dates=["Date"])
        # Convert from wide to long format: (Date, Ticker, Close)
        prices = prices_wide.melt(id_vars=["Date"], var_name="Ticker", value_name="Close")
        # Now prices looks like:
        # Date        Ticker   Close
        # 2024-07-01  AAPL     198.1
        # 2024-07-01  MSFT     342.2
        # ...
        # Get latest prices (most recent date)
        latest_date = prices["Date"].max()
        latest_prices = prices[prices["Date"] == latest_date]
        # Load fundamentals
        fundamentals = pd.read_csv("data/dummy_fundamentals.csv")
        # Merge on Ticker
        combined_data = fundamentals.merge(latest_prices, on="Ticker")
        combined_data.rename(columns={0: "Latest_Price"}, inplace=True)
        if weighing_strategy == "Equal Weight":
            combined_data["Weight"] = 1 / len(combined_data)
        elif weighing_strategy == "Score-Based":
            combined_data = compute_fundamental_scores(combined_data)  # assume PE, ROE, DE are there
            combined_data["Weight"] = combined_data["Composite_Score"] / combined_data["Composite_Score"].sum()
        elif weighing_strategy == "Mean-Variance Optimized":
            combined_data["Weight"] = optimize_weights(prices, combined_data["Ticker"].tolist())
        st.success("Default data loaded successfully!")
        st.write("Price Data:")
        st.dataframe(prices.tail())
        st.write("Fundamentals Data:")
        st.dataframe(fundamentals.tail())
        
    except Exception as e:
        st.error(f"Error loading default data: {e}")
        st.stop()

elif data_source == "Upload Custom CSV/Excel":
    uploaded_prices = st.file_uploader("Upload stock price CSV/Excel file", type=["csv", "xlsx"])
    uploaded_fundamentals = st.file_uploader("Upload stock fundamentals CSV/Excel file", type=["csv", "xlsx"])
    uploaded_index500 = st.file_uploader("Upload Index 500 CSV/Excel file", type=["csv", "xlsx"])

    if uploaded_prices and uploaded_fundamentals and uploaded_index500:

        if uploaded_prices.name.endswith('.csv'):

            prices = pd.read_csv(uploaded_prices, index_col=0, parse_dates=True)
            fundamentals = pd.read_csv(uploaded_fundamentals, index_col=0)
            index500 = pd.read_csv(uploaded_index500, index_col=0, parse_dates=True)
            st.success("Custom data loaded successfully!")
            st.write("Price Data:")
            st.dataframe(prices.tail())
            st.write("Fundamentals Data:")
            st.dataframe(fundamentals.tail())
            st.write("Index 500 Data:")
            st.dataframe(index500.tail())

        else:
            st.warning("Please upload CSV files for prices, fundamentals, and Index 500 data.")

elif data_source == "Use yfinance":

    import yfinance as yf

    tickers_input = st.text_input("Enter tickers separated by space (e.g., AAPL MSFT GOOGL)")

    start_date = st.date_input("Start Date")

    end_date = st.date_input("End Date")

    if st.button("Fetch from yfinance"):

        tickers = tickers_input.upper().split()

        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        else:
            # If only one ticker, data is single-level columns
            prices = data
            prices = data.dropna(axis=1)

        st.write("Fetched Data:")

        st.dataframe(prices.tail())

elif data_source == "Use Financial Modeling Prep API":

    import requests

    api_key = st.text_input("Enter your FMP API Key", type="password")

    tickers_input = st.text_input("Enter tickers separated by comma (e.g., AAPL,MSFT,GOOGL)")

    start_date = st.date_input("Start Date")

    end_date = st.date_input("End Date")

    if st.button("Fetch from FMP"):

        tickers = tickers_input.upper().split(',')

        dfs = []

        for ticker in tickers:

            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={start_date}&to={end_date}&apikey={api_key}"

            r = requests.get(url)

            data = r.json()

            try:

                hist = pd.DataFrame(data['historical'])

                hist['date'] = pd.to_datetime(hist['date'])

                hist.set_index('date', inplace=True)

                close_prices = hist[['close']].rename(columns={'close': ticker})

                dfs.append(close_prices)

            except:
                st.warning(f"Data not available for {ticker}")

        if dfs:
            prices = pd.concat(dfs, axis=1).sort_index()
            st.write("Fetched Data:")
            st.dataframe(prices.tail())
else:
    st.warning("Please select a data source.")

## If prices are loaded, proceed with the backtesting setup
max_weight = st.slider("Max weight per stock", 0.01, 0.3, 0.1, 0.01)
sector_cap = st.slider("Max sector cap", 0.05, 0.5, 0.25, 0.05)

if prices is not None:
    prices = prices.dropna(axis=1)  # clean missing securities

    # Pivot to wide format for backtesting
    prices_wide = prices.pivot(index="Date", columns="Ticker", values="Close")


    tickers = prices['Ticker'].unique().tolist()
    fundamentals= pd.read_csv("data/dummy_fundamentals.csv") 
    fundamentals = fundamentals.set_index('Ticker')
    if not set(tickers).issubset(set(fundamentals.index)):
        st.error("Fundamentals data is missing for some tickers.")
    
    if len(tickers) == 0:
        st.warning("No valid tickers found in the data.")
        st.stop()

    if fundamentals is not None:
        scores = fundamentals.loc[tickers].dropna()
    else:
        scores = compute_fundamental_scores(tickers)

    sector_matrix = get_sector_matrix(tickers)

    def weight_fn(scores, sector_matrix):
        return optimize_weights(scores, sector_matrix, max_weight=max_weight, sector_cap=sector_cap)


    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            
                pf_returns, weights_dict = backtest_portfolio(prices_wide, weight_fn=weight_fn, rebalance_freq='M')
                metrics = calculate_performance(pf_returns)
                turnover = calculate_turnover(weights_dict)

                index_values=backtest(prices,combined_data)
                benchmark=load_indxx_500()
                comparison_df=align_with_benchmark(index_values, benchmark)
                
                st.subheader("Backtest Performance Stats")
                st.write(metrics)
                st.metric("CAGR", f"{metrics['CAGR']*100:.2f}%")
                st.metric("Volatility", f"{metrics['Volatility']*100:.2f}%")
                st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
                st.metric("Max Drawdown", f"{metrics['Max Drawdown']*100:.2f}%")
                st.metric("Average Turnover", f"{turnover:.2f}")

                st.write("**Cumulative Return Chart**")
                fig = plot_returns(pf_returns)
                st.pyplot(fig)