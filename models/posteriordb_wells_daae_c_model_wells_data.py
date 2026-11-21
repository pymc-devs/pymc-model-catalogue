"""
Model: 4-Predictor logistic regression model with centered inputs for decision to switch wells
Source: posteriordb/posterior_database/models/stan/wells_daae_c_model.stan
Authors: Andrew Gelman, Jennifer Hill
Description: Logistic regression for well switching using centered distance, arsenic, education and
    community-association predictors.

Stan comparison:
- A: Stan = -2039.2934246892526; PyMC = -2039.2934246892526.
- B: Stan = -1994.319657791627; PyMC = -1994.3196577916267.
- Raw delta: Stan = 44.97376689762564; PyMC = 44.973766897625865.
- Raw comparison: MATCH; residual = -2.2737367544323206e-13.
- Tolerance: 1.4497376689762585e-06
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
    assoc = np.asarray(data["assoc"], dtype=float)
    educ = np.asarray(data["educ"], dtype=float) / 4.0
    design = np.column_stack((dist, arsenic, dist * arsenic, assoc, educ))
    coords = {
        "household": np.arange(switched.size),
        "coefficient": [
            "centered_dist100",
            "centered_arsenic",
            "distance_arsenic",
            "assoc",
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
