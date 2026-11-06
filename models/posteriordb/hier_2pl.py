"""Native hierarchical two-parameter logistic model from Furr (2016).

Item effects use the centered bivariate Gaussian law.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("hier_2pl", posterior)
    item = np.asarray(data["ii"], dtype="int64") - 1
    person = np.asarray(data["jj"], dtype="int64") - 1
    coords = {
        "item": np.arange(1, data["I"] + 1),
        "person": np.arange(1, data["J"] + 1),
        "item_parameter": ["log_discrimination", "difficulty"],
        "response": np.arange(data["N"]),
    }
    with pm.Model(coords=coords) as model:
        theta = pm.Normal("theta", 0, 1, dims="person")
        mu = pm.Normal("mu", 0, [1, 5], dims="item_parameter")
        tau = pm.Exponential("tau", lam=0.1, dims="item_parameter", initval=np.ones(2))
        # A Beta(4, 4) correlation coordinate gives the two-dimensional LKJ(4) law.
        correlation_unit = pm.Beta("correlation_unit", alpha=4, beta=4)
        rho = pm.Deterministic("rho", 2 * correlation_unit - 1)
        correlation_chol = pt.stack(
            [
                pt.stack([pt.ones_like(rho), pt.zeros_like(rho)]),
                pt.stack([rho, pt.sqrt(1 - rho**2)]),
            ]
        )
        chol = tau[:, None] * correlation_chol
        xi = pm.MvNormal(
            "xi",
            mu=mu,
            chol=chol,
            dims=("item", "item_parameter"),
            initval=np.zeros((data["I"], 2)),
        )
        alpha = pm.Deterministic("alpha", pt.exp(xi[:, 0]), dims="item")
        beta = pm.Deterministic("beta", xi[:, 1], dims="item")
        pm.Bernoulli(
            "y",
            logit_p=alpha[item] * (theta[person] - beta[item]),
            observed=np.asarray(data["y"], dtype="int64"),
            dims="response",
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("hier_2pl")
