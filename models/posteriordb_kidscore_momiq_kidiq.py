"""One Predictor Linear Model.

Stan comparison:
- A: Stan = -1898952.5453926628; PyMC = -1898951.8522455075.
- B: Stan = -1759958.4851688948; PyMC = -1759957.7920217412.
- Raw delta: Stan = 138994.060223768; PyMC = 138994.06022376637.
- Raw comparison: MATCH; residual = 1.6298145055770874e-09.
- Tolerance: 0.0013909406022376799
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
    iq = np.asarray(data["mom_iq"], dtype=float)
    design = np.column_stack((np.ones(score.size), iq))
    coords = {
        "child": np.arange(score.size),
        "coefficient": ["intercept", "mom_iq"],
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
