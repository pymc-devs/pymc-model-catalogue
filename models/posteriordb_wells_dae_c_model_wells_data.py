"""
Model: 3-Predictor logistic regression model with centered inputs for decision to switch wells
Source: posteriordb/posterior_database/models/stan/wells_dae_c_model.stan
Authors: Andrew Gelman, Jennifer Hill
Description: Performs logistic regression using three centered predictors (distance from nearest
    safe well, arsenic levels, and years in education).

Stan comparison:
- A: Stan = -2043.3908041032953; PyMC = -2043.3908041032953.
- B: Stan = -2173.8142278637506; PyMC = -2173.8142278637506.
- Raw delta: Stan = -130.4234237604553; PyMC = -130.4234237604553.
- Raw comparison: MATCH; residual = 0.0.
- Tolerance: 2.304234237604553e-06
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
    educ = np.asarray(data["educ"], dtype=float) / 4.0
    design = np.column_stack((dist, arsenic, dist * arsenic, educ))
    coords = {
        "household": np.arange(switched.size),
        "coefficient": [
            "centered_dist100",
            "centered_arsenic",
            "distance_arsenic",
            "educ4",
        ],
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
