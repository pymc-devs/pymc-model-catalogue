"""Native shared learning model from posteriordb dogs_hierarchical."""

import numpy as np
import pymc as pm

from ._data import load_data
from ._hierarchical_glm import _dogs_history


def build_model(posterior: str | None = None):
    """Return the native model and its initial point for the selected pair."""
    data = load_data("dogs_hierarchical", posterior)
    y, shock, avoid = _dogs_history(data)
    coords = {
        "dog": np.arange(1, data["n_dogs"] + 1),
        "learning_trial": np.arange(2, data["n_trials"] + 1),
    }
    with pm.Model(coords=coords) as model:
        previous_shock = pm.Data(
            "previous_shock", shock[:, 1:], dims=("dog", "learning_trial")
        )
        previous_avoid = pm.Data(
            "previous_avoid", avoid[:, 1:], dims=("dog", "learning_trial")
        )
        a = pm.Uniform("a", 0, 1)
        b = pm.Uniform("b", 0, 1)
        probability = pm.math.exp(
            previous_shock * pm.math.log(a) + previous_avoid * pm.math.log(b)
        )
        # The first shock has probability one and does not depend on model parameters.
        pm.Bernoulli("first_shock", p=1.0, observed=y[:, 0], dims="dog")
        pm.Bernoulli(
            "y", p=probability, observed=y[:, 1:], dims=("dog", "learning_trial")
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("dogs_hierarchical")
