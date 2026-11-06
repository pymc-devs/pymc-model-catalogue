"""Native standardized child-score interaction regression from Gelman and Hill (2006), Chapter 4.

Each predictor uses its sample mean and twice its sample standard deviation.
All parameter densities are constant on their supports.
"""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    """Construct the native model for the selected canonical posterior pair."""
    data = load_data("kidscore_interaction_z", posterior)
    score = np.asarray(data["kid_score"], dtype=float)
    hs = np.asarray(data["mom_hs"], dtype=float)
    iq = np.asarray(data["mom_iq"], dtype=float)
    hs = (hs - hs.mean()) / (2.0 * hs.std(ddof=1))
    iq = (iq - iq.mean()) / (2.0 * iq.std(ddof=1))
    design = np.column_stack((np.ones(score.size), hs, iq, hs * iq))
    coords = {
        "child": np.arange(score.size),
        "coefficient": [
            "intercept",
            "standardized_mom_hs",
            "standardized_mom_iq",
            "hs_iq",
        ],
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

    main("kidscore_interaction_z")
