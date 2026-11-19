"""
Model: Multiple Highly Correlated Predictors Linear Model
Source: posteriordb/posterior_database/models/stan/kilpisjarvi.stan
Authors: Bales, Ben and Pourzanjani, Arya and Vehtari, Aki and Petzold, Linda (cited work)
Description: y ~ x

Stan comparison:
- A: Stan = -381768.952124134; PyMC = -381768.95212419797.
- B: Stan = -45519254.86296274; PyMC = -45519254.86296279.
- Raw delta: Stan = -45137485.910838604; PyMC = -45137485.91083859.
- Raw comparison: MATCH; residual = -1.4901161193847656e-08.
- Tolerance: 0.451375859108386
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_kilpisjarvi_mod.json.zip"
        ) as archive,
        archive.open("kilpisjarvi_mod.json") as stream,
    ):
        data = json.load(stream)
    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    with pm.Model(coords={"year": np.arange(int(data["N"]))}) as model:
        alpha = pm.Normal("alpha", mu=data["pmualpha"], sigma=data["psalpha"])
        beta = pm.Normal("beta", mu=data["pmubeta"], sigma=data["psbeta"])
        sigma = pm.HalfFlat("sigma", initval=1)
        time = pm.Data("x", x, dims="year")
        pm.Normal("y", mu=alpha + beta * time, sigma=sigma, observed=y, dims="year")
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
