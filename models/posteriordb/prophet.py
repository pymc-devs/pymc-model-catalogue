"""Prophet trend and regression from the posteriordb prophet probability model.

The hinge basis preserves continuity for linear trends and logistic log odds.
The logistic offset is a time location, not a log-odds intercept.
Source: Taylor and Letham (2018), as specified by posteriordb.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("prophet", posterior)
    times = np.asarray(data["t"], dtype=float)
    changepoints = np.asarray(data["t_change"], dtype=float)
    hinge = np.maximum(times[:, None] - changepoints[None, :], 0)
    coords = {
        "time": times,
        "changepoint": changepoints,
        "feature": np.arange(int(data["K"])),
    }
    with pm.Model(coords=coords) as model:
        k = pm.Normal("k", mu=0, sigma=5, initval=0.1)
        m = pm.Normal("m", mu=0, sigma=5)
        delta = pm.Laplace("delta", mu=0, b=data["tau"], dims="changepoint")
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.5)
        beta = pm.Normal(
            "beta", mu=0, sigma=np.asarray(data["sigmas"], dtype=float), dims="feature"
        )
        time = pm.Data("t", times, dims="time")
        basis = pm.Data("hinge", hinge, dims=("time", "changepoint"))
        design = pm.Data(
            "X", np.asarray(data["X"], dtype=float), dims=("time", "feature")
        )
        if int(data["trend_indicator"]) == 0:
            trend = k * time + m + basis @ delta
        else:
            capacity = pm.Data("cap", np.asarray(data["cap"], dtype=float), dims="time")
            trend = capacity * pt.sigmoid(k * (time - m) + basis @ delta)
        additive = design @ (beta * np.asarray(data["s_a"], dtype=float))
        multiplicative = design @ (beta * np.asarray(data["s_m"], dtype=float))
        expectation = trend * (1 + multiplicative) + additive
        pm.Normal(
            "y",
            mu=expectation,
            sigma=sigma_obs,
            observed=np.asarray(data["y"], dtype=float),
            dims="time",
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("prophet")
