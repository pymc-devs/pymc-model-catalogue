"""
Model: One Predictor Log-linear Model
Source: posteriordb/posterior_database/models/stan/logearn_height.stan
Authors: Andrew Gelman, Jennifer Hill
Description: log(earn) ~ height

Stan comparison:
- A: Stan = -1804.8224042939457; PyMC = -1804.8224042939423.
- B: Stan = -6752.272852830232; PyMC = -6752.272852830225.
- Raw delta: Stan = -4947.450448536287; PyMC = -4947.450448536283.
- Raw comparison: MATCH; residual = -3.637978807091713e-12.
- Tolerance: 5.0474504485362866e-05
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_earnings.json.zip"
        ) as archive,
        archive.open("earnings.json") as stream,
    ):
        data = json.load(stream)
    response = np.log(np.asarray(data["earn"], dtype=float))
    height = np.asarray(data["height"], dtype=float)
    design = np.column_stack(
        (
            np.ones(len(response)),
            height,
        )
    )
    coords = {
        "obs_id": np.arange(len(response)),
        "coefficient": ["intercept", "height"],
    }
    initial_beta = np.zeros(design.shape[1])
    initial_beta[0] = response.mean()
    with pm.Model(coords=coords) as model:
        x = pm.Data("design", design, dims=("obs_id", "coefficient"))
        beta = pm.Flat("beta", dims="coefficient", initval=initial_beta)
        sigma = pm.HalfFlat("sigma", initval=max(float(response.std()), 1.0))
        pm.Normal(
            "log_earn",
            mu=pm.math.dot(x, beta),
            sigma=sigma,
            observed=response,
            dims="obs_id",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
