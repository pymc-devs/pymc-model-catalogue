"""
Model: A pooled linear model for the Radon dataset
Source: posteriordb/posterior_database/models/stan/radon_pooled.stan
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
Description: A simple pooled linear model.

Stan comparison:
- A: Stan = -24706.489744389342; PyMC = -24705.796597272732.
- B: Stan = -22623.248817100746; PyMC = -22622.555669984133.
- Raw delta: Stan = 2083.240927288596; PyMC = 2083.2409272886.
- Raw comparison: MATCH; residual = -3.637978807091713e-12.
- Tolerance: 2.1832409272885998e-05
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
    coords = {"obs_id": np.arange(data["N"])}
    with pm.Model(coords=coords) as model:
        floor = pm.Data(
            "floor_measure",
            np.asarray(data["floor_measure"], dtype=float),
            dims="obs_id",
        )
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma_y = pm.HalfNormal("sigma_y", sigma=1, initval=1)
        pm.Normal(
            "log_radon",
            mu=alpha + beta * floor,
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
