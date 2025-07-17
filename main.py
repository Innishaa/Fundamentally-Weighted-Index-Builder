from src.data_loader import load_data
from src.scoring_engine import score_stocks
from src.price_fetcher import download_price_data
from src.optimizer import optimize_weights
from src.backtester import simulate_portfolio, compute_statistics

def main():
    df = load_data()
    scored_df = score_stocks(df)
    top_stocks = scored_df.head(3)

    print("\n Top Stocks:\n", top_stocks[['Ticker', 'Score']])

    tickers = top_stocks['Ticker'].tolist()
    price_df = download_price_data(tickers, start='2023-01-01', end='2024-01-01')

    daily_returns = price_df.pct_change().dropna()
    expected_returns = daily_returns.mean()
    covariance_matrix = daily_returns.cov()

    weights = optimize_weights(expected_returns.values, covariance_matrix.values, max_weight=0.6)

    portfolio_returns = simulate_portfolio(price_df, weights)
    stats = compute_statistics(portfolio_returns)

    print("\n Portfolio Stats:")
    for k, v in stats.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
