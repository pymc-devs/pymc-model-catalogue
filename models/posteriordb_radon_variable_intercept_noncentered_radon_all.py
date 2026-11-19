"""
Model: Variable intercept hierarchical model (non-centered)
Source: posteriordb/posterior_database/models/stan/radon_variable_intercept_noncentered.stan
Authors: Andrew Gelman, Jennifer Hill
Description: A linear model with different intercepts per county, a global regression coefficient,
    and a hierarchical prior. This model uses a non-centered parametrization.

Stan comparison:
- A: Stan = -24677.87908584188; PyMC = -24676.49279154477.
- B: Stan = -26830.194559009662; PyMC = -26828.808264712476.
- Raw delta: Stan = -2152.3154731677823; PyMC = -2152.315473167706.
- Raw comparison: MATCH; residual = -7.639755494892597e-11.
- Tolerance: 2.2523154731677823e-05
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
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1, initval=1)
        alpha_raw = pm.Normal("alpha_raw", mu=0, sigma=1, dims="county")
        alpha = pm.Deterministic(
            "alpha", mu_alpha + sigma_alpha * alpha_raw, dims="county"
        )
        beta = pm.Normal("beta", mu=0, sigma=10)
        pm.Normal(
            "log_radon",
            mu=alpha[county] + beta * floor,
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
