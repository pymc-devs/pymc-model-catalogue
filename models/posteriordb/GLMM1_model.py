"""Native hierarchical Poisson site model from posteriordb GLMM1_model."""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    """Return the native model and its initial point for the selected pair."""
    data = load_data("GLMM1_model", posterior)
    coords = {
        "site": np.arange(1, data["nsite"] + 1),
        "observation": np.arange(data["nobs"]),
    }
    with pm.Model(coords=coords) as model:
        site = pm.Data(
            "site_index",
            np.asarray(data["obssite"], dtype="int64") - 1,
            dims="observation",
        )
        mu_alpha = pm.Normal("mu_alpha", 0, 10)
        sd_alpha = pm.Uniform("sd_alpha", 0, 5)
        alpha = pm.Normal("alpha", mu_alpha, sd_alpha, dims="site")
        pm.Poisson(
            "obs", mu=pm.math.exp(alpha[site]), observed=data["obs"], dims="observation"
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("GLMM1_model")
