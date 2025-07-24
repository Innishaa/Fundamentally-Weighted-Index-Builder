import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
from src.data_utils import compute_fundamental_scores, get_sector_matrix
def backtest_portfolio(prices, weight_fn, rebalance_freq='D'):

    rets = prices.pct_change().dropna()

    weights = {}

    portfolio_returns = []

    for date in rets.resample(rebalance_freq).first().index:

        try:

            sub_prices = prices.loc[:date].dropna()
            if len(sub_prices) < 21:
                print(f"Skipped {date}: Not enough data points for backtest.")
                continue

            latest = sub_prices.iloc[-1]

            past = sub_prices.iloc[-21:]

            scores = compute_fundamental_scores(latest)  # define in data_utils

            sector_matrix = get_sector_matrix(latest.index)  # define in data_utils

            w = weight_fn(scores, sector_matrix)

            weights[date] = w
            future_window= rets.loc[date:]
            if len(future_window) < 5:
                print(f"Skipped {date}: Not enough future data points for backtest.")
                continue
            future_returns =future_window.iloc[:30]

            pf_returns = (future_returns @ w).dropna()
            if pf_returns.empty:
                print(f"Skipped {date}: No returns data available.")
                continue

            portfolio_returns.append(pf_returns)

            

        except Exception as e:
            #if 'weights' in locals() and (weights is None or len(weights) == 0):
            print(f"Skipped {date}: No weights generated.")
            continue

    if len(portfolio_returns) == 0:
        raise ValueError("Backtest failed: No portfolio returns to concatenate.")

    # elif portfolio_returns:
    return pd.concat(portfolio_returns), weights
    # else:
    #     print("No portfolio returns to concatenate.")
    #     return pd.Series(dtype=float), weights
    

def calculate_performance(portfolio_returns):

    cumulative_return = (1 + portfolio_returns).cumprod()

    cagr = (cumulative_return.iloc[-1]) ** (252 / len(portfolio_returns)) - 1

    volatility = np.std(portfolio_returns) * np.sqrt(252)

    sharpe_ratio = cagr / volatility

    drawdown = cumulative_return / cumulative_return.cummax() - 1

    max_drawdown = drawdown.min()

    return {

        "CAGR": round(cagr, 4),

        "Volatility": round(volatility, 4),

        "Sharpe Ratio": round(sharpe_ratio, 4),

        "Max Drawdown": round(max_drawdown, 4)

    }

def calculate_turnover(weights_dict):

    dates = sorted(weights_dict.keys())

    turnover = []

    for i in range(1, len(dates)):

        w_prev = weights_dict[dates[i - 1]]

        w_curr = weights_dict[dates[i]]

        turnover.append(np.sum(np.abs(w_curr - w_prev)))

    return np.mean(turnover)

def plot_returns(portfolio_returns, benchmark_returns=None):

    cumulative = (1 + portfolio_returns).cumprod()

    plt.figure(figsize=(10, 6))

    plt.plot(cumulative, label='Fundamental Index')

    if benchmark_returns is not None:

        plt.plot((1 + benchmark_returns).cumprod(), label='Benchmark')

    plt.legend()

    plt.title("Cumulative Returns")

    plt.grid(True)

    plt.show()
 