"""
Model: Linear model for the effect of encouragement to watch on actually watching Sesame Street
Source: posteriordb/posterior_database/models/stan/sesame_one_pred_a.stan
Authors: Andrew Gelman, Jennifer Hill
Description: A linear model with one predictor.

Stan comparison:
- A: Stan = -335.9748676806489; PyMC = -335.97486768064965.
- B: Stan = -337.3023539131388; PyMC = -337.30235391313954.
- Raw delta: Stan = -1.327486232489889; PyMC = -1.327486232489889.
- Raw comparison: MATCH; residual = 0.0.
- Tolerance: 1.013274862324899e-06
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_sesame_data.json.zip"
        ) as archive,
        archive.open("sesame_data.json") as stream,
    ):
        data = json.load(stream)
    response = np.asarray(data["watched"], dtype=float)
    encouraged = np.asarray(data["encouraged"], dtype=float)
    design = np.column_stack(
        (
            np.ones(len(response)),
            encouraged,
        )
    )
    coords = {
        "obs_id": np.arange(len(response)),
        "coefficient": ["intercept", "encouraged"],
    }
    initial_beta = np.zeros(design.shape[1])
    initial_beta[0] = response.mean()
    with pm.Model(coords=coords) as model:
        x = pm.Data("design", design, dims=("obs_id", "coefficient"))
        beta = pm.Flat("beta", dims="coefficient", initval=initial_beta)
        sigma = pm.HalfFlat("sigma", initval=max(float(response.std()), 1.0))
        pm.Normal(
            "watched",
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
