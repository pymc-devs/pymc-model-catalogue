"""
Model: Logistic Mixed Effects Model
Source: posteriordb/posterior_database/models/stan/dogs.stan
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
Description: y ~ logistic(n_avoid + n_shock)

Stan comparison:
- A: Stan = -628.7830399113662; PyMC = -628.783040103219.
- B: Stan = -903.8250716435396; PyMC = -903.8250718353919.
- Raw delta: Stan = -275.0420317321733; PyMC = -275.04203173217286.
- Raw comparison: MATCH; residual = -4.547473508864641e-13.
- Tolerance: 3.750420317321733e-06
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    def _dogs_history(data):
        """Compute observed shock and avoidance histories before each trial."""
        y = np.asarray(data["y"], dtype="int64")
        shock = np.cumsum(y, axis=1) - y
        avoid = np.arange(y.shape[1])[None, :] - shock
        return y, shock, avoid

    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_dogs.json.zip"
        ) as archive,
        archive.open("dogs.json") as stream,
    ):
        data = json.load(stream)
    y, shock, avoid = _dogs_history(data)
    coords = {
        "dog": np.arange(1, data["n_dogs"] + 1),
        "trial": np.arange(1, data["n_trials"] + 1),
        "coefficient": ["intercept", "avoid", "shock"],
    }
    with pm.Model(coords=coords) as model:
        previous_shock = pm.Data("previous_shock", shock, dims=("dog", "trial"))
        previous_avoid = pm.Data("previous_avoid", avoid, dims=("dog", "trial"))
        beta = pm.Normal("beta", 0, 100, dims="coefficient")
        pm.Bernoulli(
            "y",
            logit_p=beta[0] + beta[1] * previous_avoid + beta[2] * previous_shock,
            observed=y,
            dims=("dog", "trial"),
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
