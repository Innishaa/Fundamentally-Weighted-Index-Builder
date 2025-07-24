import numpy as np
import pandas as pd
from src.data_loader import load_data_from_csv
from src.scoring_engine import score_stocks
from src.price_fetcher import download_price_data
from src.optimizer import optimize_weights
from src.backtester import backtest_portfolio, calculate_performance
from src.weighting_stratergies import equal_weight, market_cap_weight, ff_market_cap_weight


def main(
    csv_path="input/universe.csv",
    weight_strategy="Market Cap",     # Options: "Equal", "Market Cap", "FF Market Cap"
    start_date="2023-01-01",
    end_date="2024-01-01",
    max_weight=0.6,
    indxx_benchmark_path=None         # Optional Excel for TRI comparison
):

    # 1. Load fundamental data from CSV
    df = load_data_from_csv(csv_path)

    # 2. Score stocks (you can update logic inside score_stocks)
    scored_df = score_stocks(df)
    top_stocks = scored_df.head(50)  # Select top 50 after scoring
    print("\n Top Scored Stocks:\n", top_stocks[['Ticker', 'Score']])

    # 3. Get tickers and price data
    tickers = top_stocks['Ticker'].tolist()
    price_df = download_price_data(tickers, start=start_date, end=end_date)

    # 4. Choose weights
    n = len(top_stocks)
    if weight_strategy == "Equal":
        weights = equal_weight(n)

    elif weight_strategy == "Market Cap":
        weights = market_cap_weight(top_stocks["Mcap"].values)

    elif weight_strategy == "FF Market Cap":
        weights = ff_market_cap_weight(top_stocks["Mcap"].values, top_stocks["FF"].values)

    else:
        raise ValueError("Unknown weight strategy!")

    print(f"\n Weights ({weight_strategy}):\n", np.round(weights, 4))

    # 5. Define the weight function (for dynamic rebalancing, if needed)
    def weight_fn(scores, cov_matrix):
        return weights  # Fixed weights in this version

    # 6. Backtest
    portfolio_returns, weight_df = backtest_portfolio(price_df, weight_fn)
    stats = calculate_performance(portfolio_returns)

    print("\n Portfolio Stats:")

    for k, v in stats.items():
        print(f"{k}: {v:.4f}")

    # 7. Compare against Indxx 500 benchmark
    if indxx_benchmark_path:
        indxx_df = pd.read_excel(indxx_benchmark_path, parse_dates=["Date"])
        indxx_df.set_index("Date", inplace=True)
        merged = pd.merge(
            portfolio_returns.rename("Custom_Index"),
            indxx_df["Rebase Value"].rename("Indxx_500"),
            left_index=True,
            right_index=True,
            how="inner"
        )
        merged.plot(title="Custom Index vs Indxx 500", figsize=(10, 5))

    return top_stocks, weights, portfolio_returns


if __name__ == "__main__":
    main()