"""
Model: Hierarchical intercept model for the Radon dataset (centered)
Source: posteriordb/posterior_database/models/stan/radon_partially_pooled_centered.stan
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
Description: A linear model with different intercepts per county and a hierarchical prior. This
    model uses a centered parametrization.

Stan comparison:
- A: Stan = -23132.18531423766; PyMC = -23130.799019908536.
- B: Stan = -26812.21417938787; PyMC = -26810.8278850587.
- Raw delta: Stan = -3680.0288651502087; PyMC = -3680.028865150165.
- Raw comparison: MATCH; residual = -4.3655745685100555e-11.
- Tolerance: 3.7800288651502085e-05
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
        alpha = pm.Normal(
            "alpha",
            mu=mu_alpha,
            sigma=sigma_alpha,
            dims="county",
            initval=np.zeros(data["J"]),
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
