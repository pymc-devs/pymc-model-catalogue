"""
Model: A centered hierarchical model for 8 schools
Source: posteriordb/posterior_database/models/stan/eight_schools_centered.stan
Authors: Rubin, Donald B; Gelman, Andrew and Carlin, John B and Stern, Hal S and Dunson, David B and
    Vehtari, Aki and Rubin, Donald B (cited work)
Description: A centered hierarchical model for the 8 schools example of Rubin (1981)

Stan comparison:
- A: Stan = -46.57206565751112; PyMC = -45.87891853518809.
- B: Stan = -43.81356876639853; PyMC = -43.12042164407549.
- Raw delta: Stan = 2.7584968911125856; PyMC = 2.7584968911125998.
- Raw comparison: MATCH; residual = -1.4210854715202004e-14.
- Tolerance: 1.027584968911126e-06
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
        theta = pm.Normal("theta", mu, tau, dims="school")
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
