"""Native constrained logistic model from posteriordb dogs_log."""

import numpy as np
import pymc as pm

from ._data import load_data
from ._hierarchical_glm import _dogs_history


def build_model(posterior: str | None = None):
    """Return the native model and its initial point for the selected pair."""
    data = load_data("dogs_log", posterior)
    y, shock, avoid = _dogs_history(data)
    coords = {
        "dog": np.arange(1, data["n_dogs"] + 1),
        "trial": np.arange(1, data["n_trials"] + 1),
        "coefficient": ["avoid", "shock"],
    }
    with pm.Model(coords=coords) as model:
        previous_shock = pm.Data("previous_shock", shock, dims=("dog", "trial"))
        previous_avoid = pm.Data("previous_avoid", avoid, dims=("dog", "trial"))
        beta = pm.Uniform(
            "beta",
            lower=[-100.0, 0.0],
            upper=[0.0, 100.0],
            initval=[-0.1, 0.1],
            dims="coefficient",
        )
        pm.Bernoulli(
            "y",
            logit_p=beta[0] * previous_avoid + beta[1] * previous_shock,
            observed=y,
            dims=("dog", "trial"),
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("dogs_log")
