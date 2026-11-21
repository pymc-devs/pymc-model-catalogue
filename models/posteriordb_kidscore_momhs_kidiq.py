"""
Model: One Predictor Linear Model
Source: posteriordb/posterior_database/models/stan/kidscore_momhs.stan
Authors: Andrew Gelman, Jennifer Hill
Description: kid_score ~ mom_hs

Stan comparison:
- A: Stan = -2089221.6174608155; PyMC = -2089220.9243136612.
- B: Stan = -1326230.7365812259; PyMC = -1326230.0434340718.
- Raw delta: Stan = 762990.8808795896; PyMC = 762990.8808795894.
- Raw comparison: MATCH; residual = 2.3283064365386963e-10.
- Tolerance: 0.007630908808795896
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_kidiq.json.zip"
        ) as archive,
        archive.open("kidiq.json") as stream,
    ):
        data = json.load(stream)
    score = np.asarray(data["kid_score"], dtype=float)
    hs = np.asarray(data["mom_hs"], dtype=float)
    design = np.column_stack((np.ones(score.size), hs))
    coords = {
        "child": np.arange(score.size),
        "coefficient": ["intercept", "mom_hs"],
    }
    with pm.Model(coords=coords) as model:
        predictors = pm.Data("predictors", design, dims=("child", "coefficient"))
        beta = pm.Flat("beta", dims="coefficient")
        sigma = pm.HalfCauchy("sigma", beta=2.5, initval=20.0)
        pm.Normal(
            "kid_score",
            mu=pm.math.dot(predictors, beta),
            sigma=sigma,
            observed=score,
            dims="child",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
