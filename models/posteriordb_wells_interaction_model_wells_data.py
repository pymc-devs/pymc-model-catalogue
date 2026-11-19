"""2-Predictor Logistic regression model with interactions for decision to switch wells.

Stan comparison:
- A: Stan = -2032.6199086607278; PyMC = -2032.619908660728.
- B: Stan = -2309.673060142438; PyMC = -2309.673060142438.
- Raw delta: Stan = -277.0531514817101; PyMC = -277.05315148170985.
- Raw comparison: MATCH; residual = -2.2737367544323206e-13.
- Tolerance: 3.770531514817101e-06
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
    arsenic = np.asarray(data["arsenic"], dtype=float)
    design = np.column_stack((dist, arsenic, dist * arsenic))
    coords = {
        "household": np.arange(switched.size),
        "coefficient": ["dist100", "arsenic", "distance_arsenic"],
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
