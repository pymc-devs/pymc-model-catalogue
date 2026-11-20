"""
Model: A county intercept model with county level covariate for the Radon dataset (non-centered)
Source: posteriordb/posterior_database/models/stan/radon_hierarchical_intercept_noncentered.stan
Authors: Andrew Gelman, Jennifer Hill
Description: A linear model with different intercepts per county and county level log_uppm
    covariate. This model is a non-centered parametrization.

Stan comparison:
- A: Stan = -22549.801863340293; PyMC = -22548.415569075092.
- B: Stan = -22501.240163090904; PyMC = -22499.85386882575.
- Raw delta: Stan = 48.561700249389105; PyMC = 48.56170024934181.
- Raw comparison: MATCH; residual = 4.729372449219227e-11.
- Tolerance: 1.485617002493891e-06
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_radon_all.json.zip"
        ) as archive,
        archive.open("radon_all.json") as stream,
    ):
        data = json.load(stream)
    coords = {"obs_id": np.arange(data["N"]), "county": np.arange(1, data["J"] + 1)}
    coords["coefficient"] = ("log_uppm", "floor_measure")
    with pm.Model(coords=coords) as model:
        county = pm.Data(
            "county_idx",
            np.asarray(data["county_idx"], dtype="int64") - 1,
            dims="obs_id",
        )
        floor = pm.Data(
            "floor_measure",
            np.asarray(data["floor_measure"], dtype=float),
            dims="obs_id",
        )
        uranium = pm.Data(
            "log_uppm", np.asarray(data["log_uppm"], dtype=float), dims="obs_id"
        )
        sigma_y = pm.HalfNormal("sigma_y", sigma=1, initval=1)
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1, initval=1)
        alpha_raw = pm.Normal("alpha_raw", mu=0, sigma=1, dims="county")
        alpha = pm.Deterministic(
            "alpha", mu_alpha + sigma_alpha * alpha_raw, dims="county"
        )
        beta = pm.Normal("beta", mu=0, sigma=10, dims="coefficient")
        pm.Normal(
            "log_radon",
            mu=alpha[county] + beta[0] * uranium + beta[1] * floor,
            sigma=sigma_y,
            observed=np.asarray(data["log_radon"], dtype=float),
            dims="obs_id",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
