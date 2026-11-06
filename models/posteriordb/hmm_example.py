"""Native two-state Gaussian example from posteriordb.

Source: https://github.com/imadmali/bball-hmm
The transition rows have uniform simplex densities.
The initial state weights equal one, as in the published target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ._data import load_data
from ._hidden_markov import _hidden_markov_loglik


def build_model(posterior: str | None = None):
    """Build the selected posterior from its published data."""
    data = load_data("hmm_example", posterior)
    n_states = data["K"]
    coords = {
        "time": np.arange(data["N"]),
        "state": np.arange(n_states),
        "next_state": np.arange(n_states),
    }
    with pm.Model(coords=coords) as model:
        y = pm.Data("y", np.asarray(data["y"], dtype=float), dims="time")
        theta1 = pm.Dirichlet("theta1", a=np.ones(n_states), dims="next_state")
        theta2 = pm.Dirichlet("theta2", a=np.ones(n_states), dims="next_state")
        theta = pm.Deterministic(
            "theta", pt.stack([theta1, theta2]), dims=("state", "next_state")
        )
        mu = pm.Normal(
            "mu",
            mu=np.array([3.0, 10.0]),
            sigma=1,
            transform=pm.distributions.transforms.Ordered(positive=True),
            initval=np.array([3.0, 10.0]),
            dims="state",
        )
        log_emissions = pm.logp(pm.Normal.dist(mu=mu, sigma=1), y[:, None])
        pm.Potential(
            "observations",
            _hidden_markov_loglik(log_emissions, theta, log_emissions[0]),
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("hmm_example")
