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

Implementation notes:
Conditional autoregression from the posteriordb arK probability model.

The first K observations condition the likelihood and have no density factor.
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -233.33159578143048; PyMC = -232.63844881898444.
- B: Stan = -213.5590502472479; PyMC = -212.86590328480204.
- Raw delta: Stan = 19.77254553418257; PyMC = 19.7725455341824.
- Raw comparison: MATCH; residual = 1.7053025658242404e-13.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 1.1977254553418258e-06
- HalfCauchy versus a Cauchy density on positive sigma contributes only log(2).
- Original Stan SHA-256: fd5c71a891f648de3b9b65e2a65885c974fd9478b0a137b6b229f76ff4e01de6
- Saved points: models/posteriordb_logp/points/arK-arK.npz
- Two-point evidence only; no global equivalence, support or inference claim.

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
