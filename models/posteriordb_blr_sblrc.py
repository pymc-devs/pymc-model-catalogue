"""
Model: A Bayesian linear regression model with vague priors
Source: posteriordb/posterior_database/models/stan/blr.stan
Description: A Bayesian linear regression model with vague priors.

Stan comparison:
- A: Stan = -792509.0486571618; PyMC = -792508.3555101727.
- B: Stan = -959988.8302412318; PyMC = -959988.1370942437.
- Raw delta: Stan = -167479.78158407006; PyMC = -167479.781584071.
- Raw comparison: MATCH; residual = 9.313225746154785e-10.
- Tolerance: 0.00167579781584071

Benchmark results:
- Original:  logp = -6873.0122, grad norm = 14663.6951, 10.5 us/call (5 evals).
- Frozen:    logp = -6873.0122, grad norm = 14663.6951, 13.1 us/call (5 evals).
- Local NUMBA validation; short timings are indicative.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_sblrc.json.zip"
        ) as archive,
        archive.open("sblrc.json") as stream,
    ):
        data = json.load(stream)
    coords = {"observation": np.arange(data["N"]), "predictor": np.arange(data["D"])}
    with pm.Model(coords=coords) as model:
        x = pm.Data(
            "X", np.asarray(data["X"], dtype=float), dims=("observation", "predictor")
        )
        beta = pm.Normal("beta", 0, 10, dims="predictor")
        sigma = pm.HalfNormal("sigma", 10)
        pm.Normal("y", mu=x @ beta, sigma=sigma, observed=data["y"], dims="observation")
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
