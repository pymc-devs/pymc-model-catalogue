"""Native BYM2 Poisson model from the posteriordb spatial case study.

The ICAR law retains the published soft sum constraint.
Source: https://mc-stan.org/users/documentation/case-studies/icar_stan.html
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ._data import load_data


def build_model(posterior: str | None = None):
    """Build the selected posterior from its published data."""
    data = load_data("bym2_offset_only", posterior)
    n = data["N"]
    node1 = np.asarray(data["node1"], dtype="int64") - 1
    node2 = np.asarray(data["node2"], dtype="int64") - 1
    adjacency = np.zeros((n, n), dtype="int8")
    adjacency[node1, node2] = 1
    adjacency[node2, node1] = 1
    coords = {"area": np.arange(1, n + 1)}
    with pm.Model(coords=coords) as model:
        log_exposure = pm.Data(
            "log_E", np.log(np.asarray(data["E"], dtype=float)), dims="area"
        )
        beta0 = pm.Normal("beta0", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1, initval=1)
        rho = pm.Beta("rho", alpha=0.5, beta=0.5, initval=0.5)
        theta = pm.Normal("theta", mu=0, sigma=1, dims="area")
        phi = pm.ICAR(
            "phi",
            W=adjacency,
            sigma=1,
            zero_sum_stdev=0.001,
            dims="area",
            initval=np.zeros(n),
        )
        convolved_re = pm.Deterministic(
            "convolved_re",
            pt.sqrt(1 - rho) * theta + pt.sqrt(rho / data["scaling_factor"]) * phi,
            dims="area",
        )
        eta = pm.Deterministic(
            "eta", log_exposure + beta0 + sigma * convolved_re, dims="area"
        )
        mu = pm.Deterministic("mu", pt.exp(eta), dims="area")
        pm.Poisson(
            "y", mu=mu, observed=np.asarray(data["y"], dtype="int64"), dims="area"
        )
        pm.Deterministic("log_precision", -2 * pt.log(sigma))
        pm.Deterministic("logit_rho", pt.log(rho) - pt.log1p(-rho))
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("bym2_offset_only")
