"""Native logistic shock model from posteriordb dogs."""

import numpy as np
import pymc as pm

from ._data import load_data
from ._hierarchical_glm import _dogs_history


def build_model(posterior: str | None = None):
    """Return the native model and its initial point for the selected pair."""
    data = load_data("dogs", posterior)
    y, shock, avoid = _dogs_history(data)
    coords = {
        "dog": np.arange(1, data["n_dogs"] + 1),
        "trial": np.arange(1, data["n_trials"] + 1),
        "coefficient": ["intercept", "avoid", "shock"],
    }
    with pm.Model(coords=coords) as model:
        previous_shock = pm.Data("previous_shock", shock, dims=("dog", "trial"))
        previous_avoid = pm.Data("previous_avoid", avoid, dims=("dog", "trial"))
        beta = pm.Normal("beta", 0, 100, dims="coefficient")
        pm.Bernoulli(
            "y",
            logit_p=beta[0] + beta[1] * previous_avoid + beta[2] * previous_shock,
            observed=y,
            dims=("dog", "trial"),
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("dogs")
