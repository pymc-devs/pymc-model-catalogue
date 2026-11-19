"""3-Input logistic regression model with interactions and centered inputs for decision to
switch wells.

Stan comparison: MATCH.
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
    educ = np.asarray(data["educ"], dtype=float)
    dist = (dist - dist.mean()) / 100.0
    arsenic = arsenic - arsenic.mean()
    educ = (educ - educ.mean()) / 4.0
    design = np.column_stack(
        (dist, arsenic, educ, dist * arsenic, dist * educ, arsenic * educ)
    )
    coords = {
        "household": np.arange(switched.size),
        "coefficient": [
            "centered_dist100",
            "centered_arsenic",
            "centered_educ4",
            "distance_arsenic",
            "distance_education",
            "arsenic_education",
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
