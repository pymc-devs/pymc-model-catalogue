"""Logistic regression model for decision to switch wells.

Stan comparison:
- A: Stan = -2116.4961435323917; PyMC = -2116.4961435323917.
- B: Stan = -2069.1450016549925; PyMC = -2069.145001654993.
- Raw delta: Stan = 47.351141877399186; PyMC = 47.35114187739873.
- Raw comparison: MATCH; residual = 4.547473508864641e-13.
- Tolerance: 1.4735114187739919e-06
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_wells_data.json.zip"
        ) as archive,
        archive.open("wells_data.json") as stream,
    ):
        data = json.load(stream)
    switched = np.asarray(data["switched"], dtype=np.int64)
    dist = np.asarray(data["dist"], dtype=float) / 100.0
    design = np.column_stack((dist,))
    coords = {
        "household": np.arange(switched.size),
        "coefficient": ["dist100"],
    }
    with pm.Model(coords=coords) as model:
        predictors = pm.Data("predictors", design, dims=("household", "coefficient"))
        alpha = pm.Flat("alpha")
        beta = pm.Flat("beta", dims="coefficient")
        pm.Bernoulli(
            "switched",
            logit_p=alpha + pm.math.dot(predictors, beta),
            observed=switched,
            dims="household",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
