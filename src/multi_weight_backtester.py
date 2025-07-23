import pandas as pd

import numpy as np

from src.backtester import backtest_portfolio, calculate_performance

def run_multiple_strategies(prices, scores, sector_matrix, indxx_file):

    """

    Try different weighting strategies on the same universe and compare with Indxx 500 Rebba values.

    Args:

        prices (pd.DataFrame): Price data with datetime index and ticker columns.

        scores (pd.Series): Fundamental score for each ticker.

        sector_matrix (pd.DataFrame): Ticker vs sector matrix.

        indxx_file (str or BytesIO): Path or uploaded file for Indxx 500 final analysis sheet.

    Returns:

        dict: Dictionary with output for each strategy including universe, weights, index series, performance metrics.

    """

    tickers = prices.columns.tolist()

    # Read the Indxx 500 Rebba value sheet

    indxx_df = pd.read_excel(indxx_file)

    indxx_df['Date'] = pd.to_datetime(indxx_df['Date'])

    indxx_df = indxx_df.set_index('Date')

    indxx_df = indxx_df.sort_index()

    rebba_series = indxx_df['Rebba Value']

    # Define weighting strategies

    strategies = {

        "Equal Weight": pd.Series(1 / len(tickers), index=tickers),

        "Market Cap Weight": pd.Series(np.linspace(1.0, 2.0, len(tickers)), index=tickers),  # placeholder

        "Free Float Market Cap": pd.Series(np.linspace(1.0, 2.0, len(tickers)), index=tickers) * 0.6,  # example

        "Score Based": scores

    }

    final_outputs = {}

    for strategy_name, weights in strategies.items():

        weights = weights.reindex(tickers).fillna(0)

        weights = weights / weights.sum()

        # 1. Universe (tickers included)

        universe = list(weights.index[weights > 0])

        # 2. Parameter values

        parameter_values = {

            "Scores": scores.to_dict(),

            "Sector Matrix Shape": sector_matrix.shape

        }

        # 3. Weights

        weight_dict = weights.round(6).to_dict()

        # 4. Index values via backtesting

        returns, daily_weights = backtest_portfolio(prices, lambda *_: weights, rebalance_freq='M')

        index_value = (1 + returns).cumprod()

        index_value.name = strategy_name

        # Align with Indxx 500 series

        combined = pd.DataFrame({

            "Strategy Index": index_value,

            "Rebba Value": rebba_series

        }).dropna()

        # 5. Performance analysis

        perf = calculate_performance(combined["Strategy Index"])

        perf.update({

            "Correlation with Rebba": combined["Strategy Index"].corr(combined["Rebba Value"]),

            "Tracking Error": np.sqrt(((combined["Strategy Index"] - combined["Rebba Value"]) ** 2).mean())

        })

        final_outputs[strategy_name] = {

            "Universe": universe,

            "Parameter Values": parameter_values,

            "Weights": weight_dict,

            "Index Value Series": index_value,

            "Performance Analysis": perf

        }

    return final_outputs
 