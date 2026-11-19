"""
Model: Autoregressive-5 model
Source: posteriordb/posterior_database/models/stan/arK.stan
Description: An AR(5) time series model.

Stan comparison:
- A: Stan = -233.33159578143048; PyMC = -232.63844881898444.
- B: Stan = -213.5590502472479; PyMC = -212.86590328480204.
- Raw delta: Stan = 19.77254553418257; PyMC = 19.7725455341824.
- Raw comparison: MATCH; residual = 1.7053025658242404e-13.
- Tolerance: 1.1977254553418258e-06
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(Path(__file__).parent / "data" / "posteriordb_arK.json.zip") as archive,
        archive.open("arK.json") as stream,
    ):
        data = json.load(stream)
    y = np.asarray(data["y"], dtype=float)
    order = int(data["K"])
    n_time = int(data["T"])
    lagged = np.column_stack(
        [y[order - lag : n_time - lag] for lag in range(1, order + 1)]
    )
    coords = {"lag": np.arange(1, order + 1), "time": np.arange(order, n_time)}
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10, dims="lag")
        sigma = pm.HalfCauchy("sigma", beta=2.5)
        history = pm.Data("history", lagged, dims=("time", "lag"))
        pm.Normal(
            "y", mu=alpha + history @ beta, sigma=sigma, observed=y[order:], dims="time"
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
