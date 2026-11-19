"""
Model: Hierarchical intercept model for the Radon dataset (non-centered)
Source: posteriordb/posterior_database/models/stan/radon_partially_pooled_noncentered.stan
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
Description: A linear model with different intercepts per county and a hierarchical prior. This
    model is a non-centered parametrization.

Stan comparison:
- A: Stan = -23774.80873280956; PyMC = -23773.42243848041.
- B: Stan = -22643.01928683654; PyMC = -22641.63299250735.
- Raw delta: Stan = 1131.7894459730196; PyMC = 1131.7894459730596.
- Raw comparison: MATCH; residual = -4.001776687800884e-11.
- Tolerance: 1.2317894459730598e-05
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
        sigma_y = pm.HalfNormal("sigma_y", sigma=1, initval=1)
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1, initval=1)
        alpha_raw = pm.Normal("alpha_raw", mu=0, sigma=1, dims="county")
        alpha = pm.Deterministic(
            "alpha", mu_alpha + sigma_alpha * alpha_raw, dims="county"
        )
        pm.Normal(
            "log_radon",
            mu=alpha[county],
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
