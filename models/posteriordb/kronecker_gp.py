"""Kronecker Gaussian process from posteriordb and Flaxman et al. (2015).

The native Kronecker Normal uses temporal covariance and an LKJ correlation matrix.
The sigma1 parameter adds variance to the observation covariance.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("kronecker_gp", posterior)
    n_time, n_category = data["n1"], data["n2"]
    time = np.asarray(data["x1"], dtype=float)[:n_time]
    distance_sq = np.square(time[:, None] - time[None, :])
    observations = np.asarray(data["y"], dtype=float)
    coords = {
        "time": np.arange(n_time),
        "category": np.arange(n_category),
        "category_column": np.arange(n_category),
        "grid_observation": np.arange(n_time * n_category),
    }
    # Distinct initial eigenvalues avoid a repeated-eigenvalue gradient at the identity.
    category_distance = np.abs(
        np.arange(n_category)[:, None] - np.arange(n_category)[None, :]
    )
    initial_chol = np.linalg.cholesky(0.2**category_distance)
    with pm.Model(coords=coords) as model:
        variance = pm.LogNormal("var1", mu=0, sigma=1, initval=1)
        bandwidth = pm.HalfCauchy("bw1", beta=2.5, initval=20)
        noise_variance = pm.Truncated(
            "sigma1", pm.LogNormal.dist(mu=0, sigma=1), lower=1e-5, initval=1
        )
        chol_correlation = pm.LKJCorr(
            "L",
            n=n_category,
            eta=2,
            initval=initial_chol,
            dims=("category", "category_column"),
        )
        correlation = pm.Deterministic(
            "Lambda",
            chol_correlation @ chol_correlation.T,
            dims=("category", "category_column"),
        )
        time_covariance = variance * pt.exp(-bandwidth * distance_sq) + 1e-5 * pt.eye(
            n_time
        )
        # The category axis varies fastest in the original matrix order.
        pm.KroneckerNormal(
            "y",
            mu=pt.zeros(n_time * n_category),
            covs=[time_covariance, correlation],
            sigma=pt.sqrt(noise_variance),
            observed=observations.T.reshape(-1),
            dims="grid_observation",
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("kronecker_gp")
