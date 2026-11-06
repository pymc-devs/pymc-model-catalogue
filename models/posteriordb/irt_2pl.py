"""Native two-parameter logistic model from the posteriordb definition."""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("irt_2pl", posterior)
    coords = {
        "item": np.arange(1, data["I"] + 1),
        "person": np.arange(1, data["J"] + 1),
    }
    with pm.Model(coords=coords) as model:
        sigma_theta = pm.HalfCauchy("sigma_theta", beta=2, initval=1)
        theta = pm.Normal("theta", mu=0, sigma=sigma_theta, dims="person")
        sigma_a = pm.HalfCauchy("sigma_a", beta=2, initval=1)
        a = pm.LogNormal(
            "a", mu=0, sigma=sigma_a, dims="item", initval=np.ones(data["I"])
        )
        mu_b = pm.Normal("mu_b", mu=0, sigma=5)
        sigma_b = pm.HalfCauchy("sigma_b", beta=2, initval=1)
        b = pm.Normal("b", mu=mu_b, sigma=sigma_b, dims="item")
        pm.Bernoulli(
            "y",
            logit_p=a[:, None] * (theta[None, :] - b[:, None]),
            observed=np.asarray(data["y"], dtype="int64"),
            dims=("item", "person"),
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("irt_2pl")
