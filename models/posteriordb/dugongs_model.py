"""Native growth curve from posteriordb dugongs_model."""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    """Return the native model and its initial point for the selected pair."""
    data = load_data("dugongs_model", posterior)
    with pm.Model(coords={"animal": np.arange(1, data["N"] + 1)}) as model:
        x = pm.Data("x", np.asarray(data["x"], dtype=float), dims="animal")
        alpha = pm.Normal("alpha", 0, 1000, initval=2.5)
        beta = pm.Normal("beta", 0, 1000, initval=1.0)
        growth = pm.Uniform("lambda", 0.5, 1)
        tau = pm.Gamma("tau", alpha=0.0001, beta=0.0001, initval=1.0)
        sigma = pm.Deterministic("sigma", 1 / pm.math.sqrt(tau))
        pm.Normal(
            "Y",
            mu=alpha - beta * growth**x,
            sigma=sigma,
            observed=data["Y"],
            dims="animal",
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("dugongs_model")
