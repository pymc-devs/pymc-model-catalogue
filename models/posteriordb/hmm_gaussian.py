"""Native Gaussian hidden-state model from the Damiano example in posteriordb.

The published model has flat ordered means and flat positive scales.
The first observation uses its state-specific emission density.
This corrects the published scalar sum across all initial emission densities.

Evidence: the local-evidence definition and Bayesian update use one density per state.
https://github.com/luisdamiano/stancon18/blob/master/main.Rmd#L239-L271
Source model: https://github.com/luisdamiano/stancon18
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ._data import load_data
from ._hidden_markov import _hidden_markov_loglik


def build_model(posterior: str | None = None):
    """Build the selected posterior from its published data."""
    data = load_data("hmm_gaussian", posterior)
    n_states = data["K"]
    observed = np.asarray(data["y"], dtype=float)
    initial_means = np.quantile(observed, np.linspace(0.1, 0.9, n_states))
    coords = {
        "time": np.arange(data["T"]),
        "state": np.arange(n_states),
        "next_state": np.arange(n_states),
    }
    with pm.Model(coords=coords) as model:
        y = pm.Data("y", observed, dims="time")
        pi1 = pm.Dirichlet("pi1", a=np.ones(n_states), dims="state")
        transition = pm.Dirichlet(
            "A", a=np.ones(n_states), dims=("state", "next_state")
        )
        mu = pm.Flat(
            "mu",
            transform=pm.distributions.transforms.Ordered(),
            initval=initial_means,
            dims="state",
        )
        sigma = pm.HalfFlat(
            "sigma", initval=np.full(n_states, np.std(observed)), dims="state"
        )
        log_emissions = pm.logp(pm.Normal.dist(mu=mu, sigma=sigma), y[:, None])
        initial_logp = pt.log(pi1) + log_emissions[0]
        pm.Potential(
            "observations",
            _hidden_markov_loglik(log_emissions, transition, initial_logp),
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("hmm_gaussian")
