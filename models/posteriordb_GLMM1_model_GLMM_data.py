"""
Model: Generalized Linear Mixed Model for Peregrine Population Size
Source: posteriordb/posterior_database/models/stan/GLMM1_model.stan
Authors: M Kery, M Schaub (cited work)
Description: A GLMM for modeling peregrine population size with a random effect being the site of
    observation.

Stan comparison:
- A: Stan = -68148.75210242545; PyMC = -68150.36154039991.
- B: Stan = -69482.35951328461; PyMC = -69483.96895125911.
- Raw delta: Stan = -1333.6074108591565; PyMC = -1333.6074108592002.
- Raw comparison: MATCH; residual = 4.3655745685100555e-11.
- Tolerance: 1.4336074108592003e-05
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_GLMM_data.json.zip"
        ) as archive,
        archive.open("GLMM_data.json") as stream,
    ):
        data = json.load(stream)
    coords = {
        "site": np.arange(1, data["nsite"] + 1),
        "observation": np.arange(data["nobs"]),
    }
    with pm.Model(coords=coords) as model:
        site = pm.Data(
            "site_index",
            np.asarray(data["obssite"], dtype="int64") - 1,
            dims="observation",
        )
        mu_alpha = pm.Normal("mu_alpha", 0, 10)
        sd_alpha = pm.Uniform("sd_alpha", 0, 5)
        alpha = pm.Normal("alpha", mu_alpha, sd_alpha, dims="site")
        pm.Poisson(
            "obs", mu=pm.math.exp(alpha[site]), observed=data["obs"], dims="observation"
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
