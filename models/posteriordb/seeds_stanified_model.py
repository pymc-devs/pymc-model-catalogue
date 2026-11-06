"""Native scale-based seed effects from posteriordb seeds_stanified_model."""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    """Return the native model and its initial point for the selected pair."""
    data = load_data("seeds_stanified_model", posterior)
    with pm.Model(coords={"plate": np.arange(1, data["I"] + 1)}) as model:
        x1 = pm.Data("x1", np.asarray(data["x1"], dtype=float), dims="plate")
        x2 = pm.Data("x2", np.asarray(data["x2"], dtype=float), dims="plate")
        alpha0 = pm.Normal("alpha0", 0, 1)
        alpha1 = pm.Normal("alpha1", 0, 1)
        alpha2 = pm.Normal("alpha2", 0, 1)
        alpha12 = pm.Normal("alpha12", 0, 1)
        sigma = pm.HalfCauchy("sigma", 1)
        b = pm.Normal("b", 0, sigma, dims="plate")
        logit_p = alpha0 + alpha1 * x1 + alpha2 * x2 + alpha12 * x1 * x2 + b
        pm.Binomial("n", n=data["N"], logit_p=logit_p, observed=data["n"], dims="plate")
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("seeds_stanified_model")
