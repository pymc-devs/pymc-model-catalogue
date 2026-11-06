"""Native hospital mortality model from posteriordb surgical_model."""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    """Return the native model and its initial point for the selected pair."""
    data = load_data("surgical_model", posterior)
    with pm.Model(coords={"hospital": np.arange(1, data["N"] + 1)}) as model:
        mu = pm.Normal("mu", 0, 1000)
        sigmasq = pm.InverseGamma("sigmasq", alpha=0.001, beta=0.001, initval=1.0)
        sigma = pm.Deterministic("sigma", pm.math.sqrt(sigmasq))
        b = pm.Normal("b", mu, sigma, dims="hospital")
        pm.Binomial("r", n=data["n"], logit_p=b, observed=data["r"], dims="hospital")
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("surgical_model")
