"""
Model: A Bayesian linear regression model with vague priors
Source: posteriordb/posterior_database/models/stan/blr.stan
Description: A Bayesian linear regression model with vague priors.

Stan comparison:
- A: Stan = -3244257.0094579956; PyMC = -3244256.3163110074.
- B: Stan = -6068280.409912163; PyMC = -6068279.716765175.
- Raw delta: Stan = -2824023.4004541677; PyMC = -2824023.4004541673.
- Raw comparison: MATCH; residual = -4.656612873077393e-10.
- Tolerance: 0.02824123400454168

Benchmark results:
- Original:  logp = -26110.0910, grad norm = 56791.0468, 9.6 us/call (5 evals).
- Frozen:    logp = -26110.0910, grad norm = 56791.0468, 9.1 us/call (5 evals).
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
            Path(__file__).parent / "data" / "posteriordb_sblri.json.zip"
        ) as archive,
        archive.open("sblri.json") as stream,
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
