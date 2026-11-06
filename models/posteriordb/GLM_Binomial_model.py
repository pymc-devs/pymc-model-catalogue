"""Native quadratic binomial model from posteriordb GLM_Binomial_model."""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    """Return the native model and its initial point for the selected pair."""
    data = load_data("GLM_Binomial_model", posterior)
    with pm.Model(coords={"year_id": np.arange(data["nyears"])}) as model:
        year = pm.Data("year", np.asarray(data["year"], dtype=float), dims="year_id")
        alpha = pm.Normal("alpha", 0, 100)
        beta1 = pm.Normal("beta1", 0, 100)
        beta2 = pm.Normal("beta2", 0, 100)
        pm.Binomial(
            "C",
            n=data["N"],
            logit_p=alpha + year * (beta1 + year * beta2),
            observed=data["C"],
            dims="year_id",
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("GLM_Binomial_model")
