"""
Model: Logistic regression model for decision to switch wells
Source: posteriordb/posterior_database/models/stan/wells_dist.stan
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
Description: Performs logistic regression using one predictor (distance from nearest safe well).

Stan comparison:
- A: Stan = -3351.6919122498107; PyMC = -3351.6919122498107.
- B: Stan = -8565.625928281908; PyMC = -8565.625928283498.
- Raw delta: Stan = -5213.934016032097; PyMC = -5213.934016033687.
- Raw comparison: MATCH; residual = 1.5897967386990786e-09.
- Tolerance: 5.313934016033687e-05
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
    dist = np.asarray(data["dist"], dtype=float)
    design = np.column_stack((np.ones(switched.size), dist))
    coords = {
        "household": np.arange(switched.size),
        "coefficient": ["intercept", "dist"],
    }
    with pm.Model(coords=coords) as model:
        predictors = pm.Data("predictors", design, dims=("household", "coefficient"))
        beta = pm.Flat("beta", dims="coefficient")
        pm.Bernoulli(
            "switched",
            logit_p=pm.math.dot(predictors, beta),
            observed=switched,
            dims="household",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
