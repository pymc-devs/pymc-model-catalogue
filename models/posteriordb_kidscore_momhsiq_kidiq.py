"""
Model: Multiple Predictors Linear Model
Source: posteriordb/posterior_database/models/stan/kidscore_momhsiq.stan
Authors: Andrew Gelman, Jennifer Hill
Description: kid_score ~ mom_hs + mom_iq

Stan comparison:
- A: Stan = -1001555.0012968204; PyMC = -1001554.3081496654.
- B: Stan = -4268980.8297550175; PyMC = -4268980.136607862.
- Raw delta: Stan = -3267425.8284581974; PyMC = -3267425.8284581965.
- Raw comparison: MATCH; residual = -9.313225746154785e-10.
- Tolerance: 0.032675258284581975
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
    iq = np.asarray(data["mom_iq"], dtype=float)
    design = np.column_stack((np.ones(score.size), hs, iq))
    coords = {
        "child": np.arange(score.size),
        "coefficient": ["intercept", "mom_hs", "mom_iq"],
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
