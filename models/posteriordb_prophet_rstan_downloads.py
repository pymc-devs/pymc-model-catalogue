"""
Model: Structural Time Series Model
Source: posteriordb/posterior_database/models/stan/prophet.stan
Authors: Taylor, Sean J and Letham, Benjamin (cited work)
Description: Structural Time Series Model

Stan comparison:
- A: Stan = -1526.0074637737189; PyMC = -1525.314316651395.
- B: Stan = -1323.1678056449293; PyMC = -1322.4746585226062.
- Raw delta: Stan = 202.83965812878955; PyMC = 202.83965812878887.
- Raw comparison: MATCH; residual = 6.821210263296962e-13.
- Tolerance: 3.0283965812878954e-06
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
            Path(__file__).parent / "data" / "posteriordb_rstan_downloads.json.zip"
        ) as archive,
        archive.open("rstan_downloads.json") as stream,
    ):
        data = json.load(stream)
    times = np.asarray(data["t"], dtype=float)
    changepoints = np.asarray(data["t_change"], dtype=float)
    hinge = np.maximum(times[:, None] - changepoints[None, :], 0)
    coords = {
        "time": times,
        "changepoint": changepoints,
        "feature": np.arange(int(data["K"])),
    }
    with pm.Model(coords=coords) as model:
        k = pm.Normal("k", mu=0, sigma=5, initval=0.1)
        m = pm.Normal("m", mu=0, sigma=5)
        delta = pm.Laplace("delta", mu=0, b=data["tau"], dims="changepoint")
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.5)
        beta = pm.Normal(
            "beta", mu=0, sigma=np.asarray(data["sigmas"], dtype=float), dims="feature"
        )
        time = pm.Data("t", times, dims="time")
        basis = pm.Data("hinge", hinge, dims=("time", "changepoint"))
        design = pm.Data(
            "X", np.asarray(data["X"], dtype=float), dims=("time", "feature")
        )
        if int(data["trend_indicator"]) == 0:
            trend = k * time + m + basis @ delta
        else:
            capacity = pm.Data("cap", np.asarray(data["cap"], dtype=float), dims="time")
            trend = capacity * pt.sigmoid(k * (time - m) + basis @ delta)
        additive = design @ (beta * np.asarray(data["s_a"], dtype=float))
        multiplicative = design @ (beta * np.asarray(data["s_m"], dtype=float))
        expectation = trend * (1 + multiplicative) + additive
        pm.Normal(
            "y",
            mu=expectation,
            sigma=sigma_obs,
            observed=np.asarray(data["y"], dtype=float),
            dims="time",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
