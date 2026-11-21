"""
Model: Gaussian Process Poisson Regression
Source: posteriordb/posterior_database/models/stan/gp_pois_regr.stan
Description: Poisson regression with a one-dimensional latent Gaussian process using the exponential
    quadratic covariance function.

Stan comparison:
- A: Stan = -1043.0656157679641; PyMC = -1042.3724671461957.
- B: Stan = -1044.0478221953438; PyMC = -1043.3546735735754.
- Raw delta: Stan = -0.9822064273796514; PyMC = -0.9822064273796514.
- Raw comparison: MATCH; residual = 0.0.
- Tolerance: 1.0098220642737965e-06
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_gp_pois_regr.json.zip"
        ) as archive,
        archive.open("gp_pois_regr.json") as stream,
    ):
        data = json.load(stream)
    coords = {"observation": np.arange(data["N"]), "input": ["x"]}
    with pm.Model(coords=coords) as model:
        x = pm.Data(
            "x",
            np.asarray(data["x"], dtype=float)[:, None],
            dims=("observation", "input"),
        )
        rho = pm.Gamma("rho", alpha=25, beta=4)
        alpha = pm.HalfNormal("alpha", sigma=2)
        process = pm.gp.Latent(cov_func=alpha**2 * pm.gp.cov.ExpQuad(1, ls=rho))
        f = process.prior(
            "f", X=x, reparameterize=True, jitter=1e-10, dims="observation"
        )
        pm.Poisson(
            "k",
            mu=pm.math.exp(f),
            observed=np.asarray(data["k"], dtype=int),
            dims="observation",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
