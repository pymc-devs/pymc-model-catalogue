"""
Model: Multiple Factor Level Predictors Linear Model
Source: posteriordb/posterior_database/models/stan/kidscore_mom_work.stan
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
Description: kid_score ~ as.factor(mom_work)

Stan comparison:
- A: Stan = -3489148.4822633765; PyMC = -3489148.4822633765.
- B: Stan = -1357896.0004124728; PyMC = -1357896.0004124732.
- Raw delta: Stan = 2131252.4818509035; PyMC = 2131252.4818509035.
- Raw comparison: MATCH; residual = 0.0.
- Tolerance: 0.021313524818509035
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_kidiq_with_mom_work.json.zip"
        ) as archive,
        archive.open("kidiq_with_mom_work.json") as stream,
    ):
        data = json.load(stream)
    score = np.asarray(data["kid_score"], dtype=float)
    work = np.asarray(data["mom_work"])
    design = np.column_stack((np.ones(score.size), work == 2, work == 3, work == 4))
    coords = {
        "child": np.arange(score.size),
        "coefficient": ["intercept", "work2", "work3", "work4"],
    }
    with pm.Model(coords=coords) as model:
        predictors = pm.Data("predictors", design, dims=("child", "coefficient"))
        beta = pm.Flat("beta", dims="coefficient")
        sigma = pm.HalfFlat("sigma", initval=20.0)
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
