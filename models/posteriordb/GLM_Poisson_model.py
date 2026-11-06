"""Native cubic Poisson model from posteriordb GLM_Poisson_model."""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    """Return the native model and its initial point for the selected pair."""
    data = load_data("GLM_Poisson_model", posterior)
    with pm.Model(coords={"year_id": np.arange(data["n"])}) as model:
        year = pm.Data("year", np.asarray(data["year"], dtype=float), dims="year_id")
        alpha = pm.Uniform("alpha", -20, 20)
        beta1 = pm.Uniform("beta1", -10, 10)
        beta2 = pm.Uniform("beta2", -10, 10)
        beta3 = pm.Uniform("beta3", -10, 10)
        log_lambda = alpha + year * (beta1 + year * (beta2 + year * beta3))
        pm.Poisson("C", mu=pm.math.exp(log_lambda), observed=data["C"], dims="year_id")
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("GLM_Poisson_model")
