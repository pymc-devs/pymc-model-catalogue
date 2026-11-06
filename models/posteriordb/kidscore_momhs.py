"""Native child-score regression from Gelman and Hill (2006), Chapter 3.

The coefficient density is constant. The positive noise scale follows a half-Cauchy law.
"""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    """Construct the native model for the selected canonical posterior pair."""
    data = load_data("kidscore_momhs", posterior)
    score = np.asarray(data["kid_score"], dtype=float)
    hs = np.asarray(data["mom_hs"], dtype=float)
    design = np.column_stack((np.ones(score.size), hs))
    coords = {
        "child": np.arange(score.size),
        "coefficient": ["intercept", "mom_hs"],
    }
    with pm.Model(coords=coords) as model:
        predictors = pm.Data("predictors", design, dims=("child", "coefficient"))
        beta = pm.Flat("beta", dims="coefficient")
        sigma = pm.HalfCauchy("sigma", beta=2.5, initval=20.0)
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

    main("kidscore_momhs")
