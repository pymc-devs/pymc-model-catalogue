"""
Model: 2-Predictor logistic regression model with interactions and centered inputs for decision to
    switch wells
Source: posteriordb/posterior_database/models/stan/wells_interaction_c_model.stan
Authors: Andrew Gelman, Jennifer Hill
Description: Performs logistic regression using two predictors (distance from nearest safe well and
    arsenic levels) and their interaction with centered input variables.

Stan comparison:
- A: Stan = -2175.192592127743; PyMC = -2175.1925921277434.
- B: Stan = -2055.182483293224; PyMC = -2055.182483293224.
- Raw delta: Stan = 120.01010883451909; PyMC = 120.01010883451954.
- Raw comparison: MATCH; residual = -4.547473508864641e-13.
- Tolerance: 2.2001010883451956e-06
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
    arsenic = np.asarray(data["arsenic"], dtype=float)
    dist = (dist - dist.mean()) / 100.0
    arsenic = arsenic - arsenic.mean()
    design = np.column_stack((dist, arsenic, dist * arsenic))
    coords = {
        "household": np.arange(switched.size),
        "coefficient": ["centered_dist100", "centered_arsenic", "distance_arsenic"],
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
