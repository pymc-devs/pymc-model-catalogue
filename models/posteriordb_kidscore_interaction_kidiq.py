"""
Model: Interacting Linear Model
Source: posteriordb/posterior_database/models/stan/kidscore_interaction.stan
Authors: Andrew Gelman, Jennifer Hill
Description: kid_score ~ mom_hs + mom_iq + mom_hs:mom_iq

Stan comparison:
- A: Stan = -1265544.0744717761; PyMC = -1265543.381324622.
- B: Stan = -1672581.928865948; PyMC = -1672581.235718796.
- Raw delta: Stan = -407037.85439417185; PyMC = -407037.85439417395.
- Raw comparison: MATCH; residual = 2.0954757928848267e-09.
- Tolerance: 0.004071378543941739
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
    design = np.column_stack((np.ones(score.size), hs, iq, hs * iq))
    coords = {
        "child": np.arange(score.size),
        "coefficient": ["intercept", "mom_hs", "mom_iq", "hs_iq"],
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
