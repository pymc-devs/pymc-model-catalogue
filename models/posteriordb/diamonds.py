"""Native Gaussian regression for diamond prices.

Source: Bales et al. (2019), with the ggplot2 diamonds data.
The intercept has its density at the centered predictor origin.
"""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("diamonds", posterior)
    response = np.asarray(data["Y"], dtype=float)
    predictors = np.asarray(data["X"], dtype=float)[:, 1:]
    predictor_means = predictors.mean(axis=0)
    centered_predictors = predictors - predictor_means
    coords = {
        "obs_id": np.arange(len(response)),
        "coefficient": np.arange(1, int(data["K"])),
    }
    with pm.Model(coords=coords) as model:
        x = pm.Data(
            "centered_predictors",
            centered_predictors,
            dims=("obs_id", "coefficient"),
        )
        b = pm.Normal("b", mu=0, sigma=1, dims="coefficient")
        intercept = pm.StudentT("Intercept", nu=3, mu=8, sigma=10)
        sigma = pm.HalfStudentT(
            "sigma", nu=3, sigma=10, initval=max(float(response.std()), 1.0)
        )
        pm.Deterministic("b_Intercept", intercept - pm.math.dot(predictor_means, b))
        if not data["prior_only"]:
            pm.Normal(
                "Y",
                mu=intercept + pm.math.dot(x, b),
                sigma=sigma,
                observed=response,
                dims="obs_id",
            )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("diamonds")
