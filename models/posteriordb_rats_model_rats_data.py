"""
Model: Normal Hierarchical Model to Model Rats' Weight Over Time
Source: posteriordb/posterior_database/models/stan/rats_model.stan
Authors: Alan E. Gelfand and Susan E. Hills and Amy Racine-Poon and Adrian F. M. Smith (cited work)
Description: The model is essentially a random effects linear growth curve.

Stan comparison:
- A: Stan = -5468930.328995618; PyMC = -5468930.328995745.
- B: Stan = -4398362.244870762; PyMC = -4398362.244870891.
- Raw delta: Stan = 1070568.0841248557; PyMC = 1070568.0841248538.
- Raw comparison: MATCH; residual = 1.862645149230957e-09.
- Tolerance: 0.010706680841248557
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_rats_data.json.zip"
        ) as archive,
        archive.open("rats_data.json") as stream,
    ):
        data = json.load(stream)
    coords = {
        "rat_id": np.arange(1, data["N"] + 1),
        "observation": np.arange(data["Npts"]),
    }
    with pm.Model(coords=coords) as model:
        rat = pm.Data(
            "rat_index", np.asarray(data["rat"], dtype="int64") - 1, dims="observation"
        )
        centered_x = pm.Data(
            "centered_x",
            np.asarray(data["x"], dtype=float) - data["xbar"],
            dims="observation",
        )
        mu_alpha = pm.Normal("mu_alpha", 0, 100, initval=200.0)
        mu_beta = pm.Normal("mu_beta", 0, 100, initval=6.0)
        # These improper scale densities stay flat on positive scales.
        sigma_y = pm.HalfFlat("sigma_y", initval=10.0)
        sigma_alpha = pm.HalfFlat("sigma_alpha", initval=20.0)
        sigma_beta = pm.HalfFlat("sigma_beta", initval=1.0)
        alpha = pm.Normal(
            "alpha",
            mu_alpha,
            sigma_alpha,
            dims="rat_id",
            initval=np.full(data["N"], 200.0),
        )
        beta = pm.Normal(
            "beta", mu_beta, sigma_beta, dims="rat_id", initval=np.full(data["N"], 6.0)
        )
        pm.Normal(
            "y",
            mu=alpha[rat] + beta[rat] * centered_x,
            sigma=sigma_y,
            observed=data["y"],
            dims="observation",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
