import numpy as np
from numpy import linalg
from numpy import eye
from src.data_loader import load_data
from src.scoring_engine import score_stocks
from src.price_fetcher import download_price_data
from src.optimizer import optimize_weights
from src.backtester import backtest_portfolio, calculate_performance, calculate_turnover, plot_returns

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

    # Optional: Calculate mean-variance weights for analysis
    mv_weights = optimize_weights(expected_returns.values, covariance_matrix.values, max_weight=0.6)
    print("\nMean-Variance Weights:\n", np.round(mv_weights, 4))

    # Define the weight function for the backtester
    def weight_fn(scores, sector_matrix):
        # Ensure sector_matrix is symmetric for cvxpy
        sector_matrix = 0.5 * (sector_matrix + sector_matrix.T)
        min_eig = np.min(np.linalg.eigvals(sector_matrix))
        if min_eig < 0:
            sector_matrix += np.eye(sector_matrix.shape[0]) * (-min_eig + 1e-8)
        return optimize_weights(scores, sector_matrix, max_weight=0.6)

    portfolio_returns, weights = backtest_portfolio(price_df, weight_fn)
    stats = calculate_performance(portfolio_returns)

    print("\n Portfolio Stats:")
    for k, v in stats.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
