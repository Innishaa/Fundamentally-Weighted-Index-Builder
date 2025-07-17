import numpy as np

def calculate_daily_returns(price_df):
    return price_df.pct_change().dropna()

def compute_statistics(portfolio_returns):
    cumulative_return = (1 + portfolio_returns).prod() - 1
    annual_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252))

    return {
        "Cumulative Return": cumulative_return,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe_ratio
    }

def simulate_portfolio(price_df, weights):
    daily_returns = calculate_daily_returns(price_df)
    portfolio_returns = daily_returns @ weights
    return portfolio_returns
