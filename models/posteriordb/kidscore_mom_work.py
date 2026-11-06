"""Native child-score work-category regression from Gelman and Hill (2006), Chapter 4.

Work category one is the reference. The coefficients and positive noise scale have constant densities.
"""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    """Construct the native model for the selected canonical posterior pair."""
    data = load_data("kidscore_mom_work", posterior)
    score = np.asarray(data["kid_score"], dtype=float)
    work = np.asarray(data["mom_work"])
    design = np.column_stack((np.ones(score.size), work == 2, work == 3, work == 4))
    coords = {
        "child": np.arange(score.size),
        "coefficient": ["intercept", "work2", "work3", "work4"],
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

    main("kidscore_mom_work")
