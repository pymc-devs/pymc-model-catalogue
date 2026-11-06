"""Native correlated dog effects from posteriordb dogs_nonhierarchical.

The covariance is diag(sigma) @ Omega @ diag(sigma), as defined by the source Sigma_logit_ab.
The source row-factor product does not give that covariance. This model uses the declared covariance.
Evidence: posterior_database/models/stan/dogs_nonhierarchical.stan, lines 27-32, upstream revision 91027f9.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ._data import load_data
from ._hierarchical_glm import _dogs_history


def build_model(posterior: str | None = None):
    """Return the native model and its initial point for the selected pair."""
    data = load_data("dogs_nonhierarchical", posterior)
    y, shock, avoid = _dogs_history(data)
    coords = {
        "dog": np.arange(1, data["n_dogs"] + 1),
        "learning_trial": np.arange(2, data["n_trials"] + 1),
        "learning_effect": ["shock", "avoid"],
    }
    with pm.Model(coords=coords) as model:
        previous_shock = pm.Data(
            "previous_shock", shock[:, 1:], dims=("dog", "learning_trial")
        )
        previous_avoid = pm.Data(
            "previous_avoid", avoid[:, 1:], dims=("dog", "learning_trial")
        )
        mu = pm.Logistic("mu_logit_ab", 0, 1, dims="learning_effect")
        sigma = pm.HalfNormal("sigma_logit_ab", 1, dims="learning_effect")
        # For two effects, LKJ(2) gives a Beta(2, 2) density for (rho + 1) / 2.
        rho_unit = pm.Beta("rho_unit", 2, 2)
        rho = pm.Deterministic("rho", 2 * rho_unit - 1)
        z = pm.Normal("z", 0, 1, dims=("dog", "learning_effect"))
        # Use the covariance declared by the source Sigma_logit_ab.
        logit_a = mu[0] + sigma[0] * z[:, 0]
        logit_b = mu[1] + sigma[1] * (rho * z[:, 0] + pt.sqrt(1 - rho**2) * z[:, 1])
        pm.Deterministic("a", pm.math.sigmoid(logit_a), dims="dog")
        pm.Deterministic("b", pm.math.sigmoid(logit_b), dims="dog")
        log_probability = -previous_shock * pt.softplus(
            -logit_a[:, None]
        ) - previous_avoid * pt.softplus(-logit_b[:, None])
        # The first shock has probability one and does not depend on model parameters.
        pm.Bernoulli("first_shock", p=1.0, observed=y[:, 0], dims="dog")
        pm.Bernoulli(
            "y",
            p=pt.exp(log_probability),
            observed=y[:, 1:],
            dims=("dog", "learning_trial"),
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("dogs_nonhierarchical")
