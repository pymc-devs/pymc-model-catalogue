"""Autoregressive-5 model.

Stan comparison: MATCH.
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
