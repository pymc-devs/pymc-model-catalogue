"""Native well-switching logistic regression from Gelman and Hill (2006), Chapter 5.

All coefficient densities are constant. Distance uses meters.
"""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    """Construct the native model for the selected canonical posterior pair."""
    data = load_data("wells_dist", posterior)
    switched = np.asarray(data["switched"], dtype=np.int64)
    dist = np.asarray(data["dist"], dtype=float)
    design = np.column_stack((np.ones(switched.size), dist))
    coords = {
        "household": np.arange(switched.size),
        "coefficient": ["intercept", "dist"],
    }
    with pm.Model(coords=coords) as model:
        predictors = pm.Data("predictors", design, dims=("household", "coefficient"))
        beta = pm.Flat("beta", dims="coefficient")
        pm.Bernoulli(
            "switched",
            logit_p=pm.math.dot(predictors, beta),
            observed=switched,
            dims="household",
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("wells_dist")
