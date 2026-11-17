"""
Model: Autoregressive-5 model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/arK.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/arK.py
Authors: Not specified in upstream model metadata.
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/arK.py
Upstream catalogue contributor: Mans Magnusson
Description: An AR(5) time series model.

Data: arK (upstream posterior: arK-arK)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/arK.json.zip
    Archive SHA-256: 6f2acbacb04655927bfc9173c42e51fc330bf08c6184242a62b7e39c22abe410
    Local archive: data/posteriordb_arK.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/arK-arK.json
Reference URL: https://github.com/stan-dev/stat_comp_benchmarks/tree/master/benchmarks/arK

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Conditional autoregression from the posteriordb arK probability model.

The first K observations condition the likelihood and have no density factor.
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4.

Benchmark results:
- Not measured.
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
