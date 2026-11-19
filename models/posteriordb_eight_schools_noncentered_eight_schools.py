"""
Model: A non-centered hierarchical model for 8 schools
Source: posteriordb/posterior_database/models/stan/eight_schools_noncentered.stan
Authors: Donald B Rubin, Andrew Gelman, John B Carlin, Hal S Stern, David B Dunson, Aki Vehtari
Description: A non-centered hierarchical model for the 8 schools example of Rubin (1981)

Stan comparison:
- A: Stan = -44.15124349654224; PyMC = -43.458096374219195.
- B: Stan = -44.22979687321376; PyMC = -43.53664975089073.
- Raw delta: Stan = -0.07855337667151474; PyMC = -0.07855337667153606.
- Raw comparison: MATCH; residual = 2.1316282072803006e-14.
- Tolerance: 1.0007855337667153e-06
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_eight_schools.json.zip"
        ) as archive,
        archive.open("eight_schools.json") as stream,
    ):
        data = json.load(stream)
    with pm.Model(coords={"school": np.arange(1, data["J"] + 1)}) as model:
        mu = pm.Normal("mu", 0, 5)
        tau = pm.HalfCauchy("tau", 5)
        theta_trans = pm.Normal("theta_trans", 0, 1, dims="school")
        theta = pm.Deterministic("theta", mu + tau * theta_trans, dims="school")
        pm.Normal(
            "y",
            theta,
            sigma=np.asarray(data["sigma"], dtype=float),
            observed=data["y"],
            dims="school",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
