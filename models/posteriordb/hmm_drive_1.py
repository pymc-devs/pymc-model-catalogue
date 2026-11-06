"""Native Gaussian drive model from the posteriordb basketball example.

Source: https://github.com/imadmali/bball-hmm
The initial state weights equal one, as in the published target.
Ordered transforms preserve the joint support of the emission means.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ._data import load_data
from ._hidden_markov import _hidden_markov_loglik


def build_model(posterior: str | None = None):
    """Build the selected posterior from its published data."""
    data = load_data("hmm_drive_1", posterior)
    n_states = data["K"]
    coords = {
        "time": np.arange(data["N"]),
        "state": np.arange(n_states),
        "next_state": np.arange(n_states),
    }
    with pm.Model(coords=coords) as model:
        u = pm.Data("u", np.asarray(data["u"], dtype=float), dims="time")
        v = pm.Data("v", np.asarray(data["v"], dtype=float), dims="time")
        alpha = np.asarray(data["alpha"], dtype=float)
        theta1 = pm.Dirichlet("theta1", a=alpha[0], dims="next_state")
        theta2 = pm.Dirichlet("theta2", a=alpha[1], dims="next_state")
        theta = pm.Deterministic(
            "theta", pt.stack([theta1, theta2]), dims=("state", "next_state")
        )
        phi = pm.Normal(
            "phi",
            mu=np.array([0.0, 3.0]),
            sigma=1,
            transform=pm.distributions.transforms.Ordered(),
            initval=np.array([0.0, 3.0]),
            dims="state",
        )
        mean_v = pm.Normal(
            "lambda",
            mu=np.array([0.0, 3.0]),
            sigma=1,
            transform=pm.distributions.transforms.Ordered(),
            initval=np.array([0.0, 3.0]),
            dims="state",
        )
        log_emissions = pm.logp(
            pm.Normal.dist(mu=phi, sigma=data["tau"]), u[:, None]
        ) + pm.logp(pm.Normal.dist(mu=mean_v, sigma=data["rho"]), v[:, None])
        pm.Potential(
            "observations",
            _hidden_markov_loglik(log_emissions, theta, log_emissions[0]),
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("hmm_drive_1")
