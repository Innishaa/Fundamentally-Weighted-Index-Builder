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

uploaded_file = st.file_uploader("Upload stock price Excel file", type=["xlsx", "csv"])

max_weight = st.slider("Max weight per stock", 0.01, 0.3, 0.1, 0.01)
sector_cap = st.slider("Max sector cap", 0.05, 0.5, 0.25, 0.05)

if uploaded_file:

    prices = pd.read_excel(uploaded_file, index_col=0, parse_dates=True)
    prices = prices.dropna(axis=1)  # clean missing securities

    tickers= prices.columns
    scores= compute_fundamental_scores(tickers)
    sector_matrix = get_sector_matrix(tickers)

    def weight_fn(scores, sector_matrix):
        return optimize_weights(scores, sector_matrix, max_weight=max_weight, sector_cap=sector_cap)


    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            
                pf_returns, weights_dict = backtest_portfolio(prices, weight_fn=weight_fn, rebalance_freq='M')
                metrics = calculate_performance(pf_returns)
                turnover = calculate_turnover(weights_dict)

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