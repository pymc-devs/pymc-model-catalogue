"""Native centered child-score interaction regression from Gelman and Hill (2006), Chapter 4.

Both predictors use their sample means. The coefficients and positive noise scale have constant densities.
"""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    """Construct the native model for the selected canonical posterior pair."""
    data = load_data("kidscore_interaction_c", posterior)
    score = np.asarray(data["kid_score"], dtype=float)
    hs = np.asarray(data["mom_hs"], dtype=float)
    iq = np.asarray(data["mom_iq"], dtype=float)
    hs = hs - hs.mean()
    iq = iq - iq.mean()
    design = np.column_stack((np.ones(score.size), hs, iq, hs * iq))
    coords = {
        "child": np.arange(score.size),
        "coefficient": ["intercept", "centered_mom_hs", "centered_mom_iq", "hs_iq"],
    }
    with pm.Model(coords=coords) as model:
        predictors = pm.Data("predictors", design, dims=("child", "coefficient"))
        beta = pm.Flat("beta", dims="coefficient")
        sigma = pm.HalfFlat("sigma", initval=20.0)
        pm.Normal(
            "kid_score",
            mu=pm.math.dot(predictors, beta),
            sigma=sigma,
            observed=score,
            dims="child",
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("kidscore_interaction_c")
