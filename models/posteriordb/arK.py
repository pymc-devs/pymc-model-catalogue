"""Conditional autoregression from the posteriordb arK probability model.

The first K observations condition the likelihood and have no density factor.
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4.
"""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("arK", posterior)
    y = np.asarray(data["y"], dtype=float)
    order = int(data["K"])
    n_time = int(data["T"])
    lagged = np.column_stack(
        [y[order - lag : n_time - lag] for lag in range(1, order + 1)]
    )
    coords = {"lag": np.arange(1, order + 1), "time": np.arange(order, n_time)}
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10, dims="lag")
        sigma = pm.HalfCauchy("sigma", beta=2.5)
        history = pm.Data("history", lagged, dims=("time", "lag"))
        pm.Normal(
            "y", mu=alpha + history @ beta, sigma=sigma, observed=y[order:], dims="time"
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("arK")
