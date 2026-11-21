"""
Model: Hierarchical intercept model for the Radon dataset (non-centered)
Source: posteriordb/posterior_database/models/stan/radon_partially_pooled_noncentered.stan
Authors: Andrew Gelman, Jennifer Hill
Description: A linear model with different intercepts per county and a hierarchical prior. This
    model is a non-centered parametrization.

Stan comparison:
- A: Stan = -1804.8092520268951; PyMC = -1803.4229576977502.
- B: Stan = -2618.686128768048; PyMC = -2617.2998344389034.
- Raw delta: Stan = -813.876876741153; PyMC = -813.8768767411532.
- Raw comparison: MATCH; residual = 2.2737367544323206e-13.
- Tolerance: 9.138768767411532e-06
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
