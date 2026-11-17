"""
Model: Structural Time Series Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/prophet.stan
Authors: Taylor, Sean J and Letham, Benjamin (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/prophet.py
Upstream catalogue contributor: Oliver Järnefelt
Description: Structural Time Series Model

Data: rstan_downloads (upstream posterior: rstan_downloads-prophet)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/rstan_downloads.json.zip
    Archive SHA-256: a5403e3347731ec0c978dcfe1d81d5f7b62d3f2cad9b5f6be64e89260d298371
    Local archive: data/posteriordb_rstan_downloads.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/rstan_downloads-prophet.json
References: taylor2018forecasting (see posteriordb/source/bibliography/references.bib)
Reference URL: https://raw.githubusercontent.com/facebook/prophet/master/R/inst/stan/prophet.stan

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Prophet trend and regression from the posteriordb prophet probability model.

The hinge basis preserves continuity for linear trends and logistic log odds.
The logistic offset is a time location, not a log-odds intercept.
Source: Taylor and Letham (2018), as specified by posteriordb.

Benchmark results:
- Not measured.
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
