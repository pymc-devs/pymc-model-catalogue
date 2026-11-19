"""Multiple Interacting Predictors Conventionally Centered Linear Model.

Stan comparison:
- A: Stan = -866198.4706636631; PyMC = -866198.4706636635.
- B: Stan = -1876522.7716739054; PyMC = -1876522.7716739047.
- Raw delta: Stan = -1010324.3010102423; PyMC = -1010324.3010102413.
- Raw comparison: MATCH; residual = -1.0477378964424133e-09.
- Tolerance: 0.010104243010102423
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
    hs = np.asarray(data["mom_hs"], dtype=float) - 0.5
    iq = np.asarray(data["mom_iq"], dtype=float) - 100.0
    design = np.column_stack((np.ones(score.size), hs, iq, hs * iq))
    coords = {
        "child": np.arange(score.size),
        "coefficient": ["intercept", "reference_mom_hs", "reference_mom_iq", "hs_iq"],
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
