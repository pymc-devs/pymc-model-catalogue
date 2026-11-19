"""
Model: Random Effects (Rasch) Model for True Difficulty of LSAT Questions
Source: posteriordb/posterior_database/models/stan/lsat_model.stan
Authors: MRC Biostatistics Unit (cited work)
Description: Rasch model for correct LSAT responses, with item difficulty and latent student
    ability.

Stan comparison:
- A: Stan = -4649.540355460236; PyMC = -4648.847208663369.
- B: Stan = -4287.590745592309; PyMC = -4286.897598795441.
- Raw delta: Stan = 361.9496098679274; PyMC = 361.9496098679283.
- Raw comparison: MATCH; residual = -9.094947017729282e-13.
- Tolerance: 4.619496098679283e-06
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_lsat_data.json.zip"
        ) as archive,
        archive.open("lsat_data.json") as stream,
    ):
        data = json.load(stream)
    cumulative = np.asarray(data["culm"], dtype="int64")
    counts = np.diff(np.concatenate(([0], cumulative)))
    responses = np.repeat(np.asarray(data["response"], dtype="int64"), counts, axis=0)
    coords = {
        "item": np.arange(1, data["T"] + 1),
        "person": np.arange(1, data["N"] + 1),
    }
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", mu=0, sigma=100, dims="item")
        theta = pm.Normal("theta", mu=0, sigma=1, dims="person")
        beta = pm.HalfNormal("beta", sigma=100, initval=1)
        pm.Bernoulli(
            "response",
            logit_p=beta * theta[:, None] - alpha[None, :],
            observed=responses,
            dims=("person", "item"),
        )
        pm.Deterministic("a", alpha - pt.mean(alpha), dims="item")
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
