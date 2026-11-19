"""Multiple Interacting Predictors Centered Linear Model.

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
            Path(__file__).parent / "data" / "posteriordb_kidiq_with_mom_work.json.zip"
        ) as archive,
        archive.open("kidiq_with_mom_work.json") as stream,
    ):
        data = json.load(stream)
    score = np.asarray(data["kid_score"], dtype=float)
    hs = np.asarray(data["mom_hs"], dtype=float)
    iq = np.asarray(data["mom_iq"], dtype=float)
    hs = hs - hs.mean()
    iq = iq - iq.mean()
    design = np.column_stack((np.ones(score.size), hs, iq, hs * iq))
    coords = {
        "child": np.arange(score.size),
        "coefficient": ["intercept", "centered_mom_hs", "centered_mom_iq", "hs_iq"],
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
