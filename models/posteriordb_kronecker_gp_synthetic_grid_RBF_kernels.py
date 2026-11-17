"""
Model: Kronecker Gaussian Process
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/kronecker_gp.stan
Authors: Flaxman, Seth and Gelman, Andrew and Neill, Daniel and Smola, Alex and Vehtari, Aki and Wilson, Andrew Gordon (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/kronecker_gp.py
Upstream catalogue contributor: Keane Nguyen
Description: Gaussian Process using Kronecker structure

Data: synthetic_grid_RBF_kernels (upstream posterior: synthetic_grid_RBF_kernels-kronecker_gp)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/synthetic_grid_RBF_kernels.json.zip
    Archive SHA-256: a262adae2aa1fd3bcb9baec183fb2101f79bd70eec30727a2214f0d17e843dde
    Local archive: data/posteriordb_synthetic_grid_RBF_kernels.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/synthetic_grid_RBF_kernels-kronecker_gp.json
References: flaxman2015fast (see posteriordb/source/bibliography/references.bib)
Reference URL: https://sethrf.com/files/fast-hierarchical-GPs.pdf

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Kronecker Gaussian process from posteriordb and Flaxman et al. (2015).

The native Kronecker Normal uses temporal covariance and an LKJ correlation matrix.
The sigma1 parameter adds variance to the observation covariance.

Stan comparison:
- Not recorded.

Benchmark results:
- Not measured.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    with (
        ZipFile(
            Path(__file__).parent
            / "data"
            / "posteriordb_synthetic_grid_RBF_kernels.json.zip"
        ) as archive,
        archive.open("synthetic_grid_RBF_kernels.json") as stream,
    ):
        data = json.load(stream)
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
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
