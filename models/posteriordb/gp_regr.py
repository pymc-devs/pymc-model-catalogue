"""Exact Gaussian-process regression from posteriordb.

The sigma parameter adds variance, not standard deviation, to the diagonal.
Source: https://github.com/stan-dev/stat_comp_benchmarks/tree/master/benchmarks/gp_regr
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("gp_regr", posterior)
    n = data["N"]
    coords = {"observation": np.arange(n), "input": ["x"]}
    with pm.Model(coords=coords) as model:
        x = pm.Data(
            "x",
            np.asarray(data["x"], dtype=float)[:, None],
            dims=("observation", "input"),
        )
        rho = pm.Gamma("rho", alpha=25, beta=4)
        alpha = pm.HalfNormal("alpha", sigma=2)
        sigma = pm.HalfNormal("sigma", sigma=1)
        kernel = alpha**2 * pm.gp.cov.ExpQuad(1, ls=rho)
        covariance = kernel(x) + sigma * pt.eye(n)
        pm.MvNormal(
            "y",
            mu=pt.zeros(n),
            cov=covariance,
            observed=np.asarray(data["y"]),
            dims="observation",
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("gp_regr")
