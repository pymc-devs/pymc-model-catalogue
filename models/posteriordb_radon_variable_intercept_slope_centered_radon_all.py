"""
Model: Variable intercept and slope hierarchical Radon model (centered)
Source: posteriordb/posterior_database/models/stan/radon_variable_intercept_slope_centered.stan
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
Description: A linear model with hierarchical intercept and slope coefficients using an hierarchical
    prior. This model uses a centered parametrization.

Stan comparison:
- A: Stan = -27192.766700840995; PyMC = -27190.68725936325.
- B: Stan = -32482.191951292312; PyMC = -32480.11250981455.
- Raw delta: Stan = -5289.425250451317; PyMC = -5289.425250451299.
- Raw comparison: MATCH; residual = -1.8189894035458565e-11.
- Tolerance: 5.389425250451317e-05
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
        alpha = pm.Normal(
            "alpha",
            mu=mu_alpha,
            sigma=sigma_alpha,
            dims="county",
            initval=np.zeros(data["J"]),
        )
        mu_beta = pm.Normal("mu_beta", mu=0, sigma=10)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=1, initval=1)
        beta = pm.Normal(
            "beta",
            mu=mu_beta,
            sigma=sigma_beta,
            dims="county",
            initval=np.zeros(data["J"]),
        )
        pm.Normal(
            "log_radon",
            mu=alpha[county] + beta[county] * floor,
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
