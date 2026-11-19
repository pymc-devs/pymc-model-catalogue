"""Logarithmic Mixed Effects Model.

Stan comparison: MATCH.
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
        "coefficient": ["avoid", "shock"],
    }
    with pm.Model(coords=coords) as model:
        previous_shock = pm.Data("previous_shock", shock, dims=("dog", "trial"))
        previous_avoid = pm.Data("previous_avoid", avoid, dims=("dog", "trial"))
        beta = pm.Uniform(
            "beta",
            lower=[-100.0, 0.0],
            upper=[0.0, 100.0],
            initval=[-0.1, 0.1],
            dims="coefficient",
        )
        pm.Bernoulli(
            "y",
            logit_p=beta[0] * previous_avoid + beta[1] * previous_shock,
            observed=y,
            dims=("dog", "trial"),
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
