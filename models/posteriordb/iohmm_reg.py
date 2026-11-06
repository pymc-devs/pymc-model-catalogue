"""Native input-output hidden-state regression from the Damiano example in posteriordb.

The original data generator selects each destination state with input-dependent softmax weights.
These weights do not depend on the previous state, so exact state elimination gives independent Gaussian mixtures.
The first observation uses the separate initial state probabilities.

The published forward recurrence indexes the source state instead of the destination state.
This implementation corrects that defect and uses the documented generative model.
Evidence: https://github.com/luisdamiano/stancon18/blob/master/main.Rmd#L840-L875

The initial probabilities have a uniform simplex density.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ._data import load_data


def build_model(posterior: str | None = None):
    """Build the selected posterior from its published data."""
    data = load_data("iohmm_reg", posterior)
    n_states = data["K"]
    coords = {
        "time": np.arange(data["T"]),
        "state": np.arange(n_states),
        "feature": np.arange(data["M"]),
    }
    with pm.Model(coords=coords) as model:
        u = pm.Data("u", np.asarray(data["u"], dtype=float), dims=("time", "feature"))
        pi1 = pm.Dirichlet("pi1", a=np.ones(n_states), dims="state")
        w = pm.Normal("w", mu=0, sigma=5, dims=("state", "feature"))
        b = pm.Normal("b", mu=0, sigma=5, dims=("state", "feature"))
        sigma = pm.HalfNormal(
            "sigma", sigma=3, initval=np.full(n_states, 3.0), dims="state"
        )
        later_weights = pm.math.softmax(pt.dot(u[1:], w.T), axis=1)
        weights = pt.concatenate([pi1[None, :], later_weights], axis=0)
        mean = pt.dot(u, b.T)
        pm.Mixture(
            "y",
            w=weights,
            comp_dists=pm.Normal.dist(mu=mean, sigma=sigma),
            observed=np.asarray(data["y"], dtype=float),
            dims="time",
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("iohmm_reg")
