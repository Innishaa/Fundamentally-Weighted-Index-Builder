import cvxpy as cp

import numpy as np

def optimize_weights(expected_returns, covariance_matrix, max_weight=0.6):

    n = len(expected_returns)

    weights = cp.Variable(n)

    # Objective: maximize return - lambda * variance (mean-variance trade-off)

    risk_aversion = 0.1  # You can tune this

    objective = cp.Maximize(expected_returns @ weights - risk_aversion * cp.quad_form(weights, covariance_matrix))

    constraints = [

        cp.sum(weights) == 1,

        weights >= 0,

        weights <= max_weight

    ]

    problem = cp.Problem(objective, constraints)

    problem.solve()

    if weights.value is not None:
        return weights.value
    else:
        raise ValueError("Optimization failed. Check your input data.")
