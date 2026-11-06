"""Native well-switching logistic regression from Gelman and Hill (2006), Chapter 5.

All coefficient densities are constant. All three predictors use sample-mean centering.
Education uses four-year units. The design includes each pairwise interaction.
"""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    """Construct the native model for the selected canonical posterior pair."""
    data = load_data("wells_dae_inter_model", posterior)
    switched = np.asarray(data["switched"], dtype=np.int64)
    dist = np.asarray(data["dist"], dtype=float)
    arsenic = np.asarray(data["arsenic"], dtype=float)
    educ = np.asarray(data["educ"], dtype=float)
    dist = (dist - dist.mean()) / 100.0
    arsenic = arsenic - arsenic.mean()
    educ = (educ - educ.mean()) / 4.0
    design = np.column_stack(
        (dist, arsenic, educ, dist * arsenic, dist * educ, arsenic * educ)
    )
    coords = {
        "household": np.arange(switched.size),
        "coefficient": [
            "centered_dist100",
            "centered_arsenic",
            "centered_educ4",
            "distance_arsenic",
            "distance_education",
            "arsenic_education",
        ],
    }
    with pm.Model(coords=coords) as model:
        predictors = pm.Data("predictors", design, dims=("household", "coefficient"))
        alpha = pm.Flat("alpha")
        beta = pm.Flat("beta", dims="coefficient")
        pm.Bernoulli(
            "switched",
            logit_p=alpha + pm.math.dot(predictors, beta),
            observed=switched,
            dims="household",
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("wells_dae_inter_model")
