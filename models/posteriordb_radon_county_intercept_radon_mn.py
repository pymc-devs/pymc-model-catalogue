"""A county intercept model (no pooling) for the Radon dataset.

Stan comparison:
- A: Stan = -2260.925627204436; PyMC = -2260.232482773764.
- B: Stan = -2023.9572563283862; PyMC = -2023.2641118977142.
- Raw delta: Stan = 236.96837087604968; PyMC = 236.96837087604968.
- Raw comparison: MATCH; residual = 0.0.
- Tolerance: 3.3696837087604966e-06
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
        floor = pm.Data(
            "floor_measure",
            np.asarray(data["floor_measure"], dtype=float),
            dims="obs_id",
        )
        alpha = pm.Normal("alpha", mu=0, sigma=10, dims="county")
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma_y = pm.HalfNormal("sigma_y", sigma=1, initval=1)
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
