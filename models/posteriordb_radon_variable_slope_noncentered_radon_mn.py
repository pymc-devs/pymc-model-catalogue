"""
Model: Variable slope hierarchical Radon model (non-centered)
Source: posteriordb/posterior_database/models/stan/radon_variable_slope_noncentered.stan
Authors: Andrew Gelman, Jennifer Hill
Description: A linear model with a global intercept and different regression coefficients using an
    hierarchical prior. This model uses a non-centered parametrization.

Stan comparison:
- A: Stan = -2328.326864104121; PyMC = -2326.9405698069518.
- B: Stan = -1947.6366663020056; PyMC = -1946.2503720048378.
- Raw delta: Stan = 380.69019780211534; PyMC = 380.69019780211397.
- Raw comparison: MATCH; residual = 1.3642420526593924e-12.
- Tolerance: 4.806901978021154e-06
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_radon_mn.json.zip"
        ) as archive,
        archive.open("radon_mn.json") as stream,
    ):
        data = json.load(stream)
    coords = {"obs_id": np.arange(data["N"]), "county": np.arange(1, data["J"] + 1)}
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
        sigma_y = pm.HalfNormal("sigma_y", sigma=1, initval=1)
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        mu_beta = pm.Normal("mu_beta", mu=0, sigma=10)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=1, initval=1)
        beta_raw = pm.Normal("beta_raw", mu=0, sigma=1, dims="county")
        beta = pm.Deterministic("beta", mu_beta + sigma_beta * beta_raw, dims="county")
        pm.Normal(
            "log_radon",
            mu=alpha + beta[county] * floor,
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
