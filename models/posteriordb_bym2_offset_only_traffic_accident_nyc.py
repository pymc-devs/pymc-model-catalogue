"""
Model: BYM2 model
Source: posteriordb/posterior_database/models/stan/bym2_offset_only.stan
Description: Spatial count model combining structured and unstructured random effects with a shared
    scale and a spatial mixing parameter.

Stan comparison:
- A: Stan = -6434575.605012348; PyMC = -6434574.9118651245.
- B: Stan = -7250370.332527916; PyMC = -7250369.639380687.
- Raw delta: Stan = -815794.727515568; PyMC = -815794.7275155624.
- Raw comparison: MATCH; residual = -5.587935447692871e-09.
- Tolerance: 0.00815894727515568
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
            Path(__file__).parent / "data" / "posteriordb_traffic_accident_nyc.json.zip"
        ) as archive,
        archive.open("traffic_accident_nyc.json") as stream,
    ):
        data = json.load(stream)
    n = data["N"]
    node1 = np.asarray(data["node1"], dtype="int64") - 1
    node2 = np.asarray(data["node2"], dtype="int64") - 1
    adjacency = np.zeros((n, n), dtype="int8")
    adjacency[node1, node2] = 1
    adjacency[node2, node1] = 1
    coords = {"area": np.arange(1, n + 1)}
    with pm.Model(coords=coords) as model:
        log_exposure = pm.Data(
            "log_E", np.log(np.asarray(data["E"], dtype=float)), dims="area"
        )
        beta0 = pm.Normal("beta0", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1, initval=1)
        rho = pm.Beta("rho", alpha=0.5, beta=0.5, initval=0.5)
        theta = pm.Normal("theta", mu=0, sigma=1, dims="area")
        phi = pm.ICAR(
            "phi",
            W=adjacency,
            sigma=1,
            zero_sum_stdev=0.001,
            dims="area",
            initval=np.zeros(n),
        )
        convolved_re = pm.Deterministic(
            "convolved_re",
            pt.sqrt(1 - rho) * theta + pt.sqrt(rho / data["scaling_factor"]) * phi,
            dims="area",
        )
        eta = pm.Deterministic(
            "eta", log_exposure + beta0 + sigma * convolved_re, dims="area"
        )
        mu = pm.Deterministic("mu", pt.exp(eta), dims="area")
        pm.Poisson(
            "y", mu=mu, observed=np.asarray(data["y"], dtype="int64"), dims="area"
        )
        pm.Deterministic("log_precision", -2 * pt.log(sigma))
        pm.Deterministic("logit_rho", pt.log(rho) - pt.log1p(-rho))
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
