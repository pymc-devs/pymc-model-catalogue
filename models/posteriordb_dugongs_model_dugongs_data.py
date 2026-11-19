"""Dugong Age and Length.

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
            Path(__file__).parent / "data" / "posteriordb_dugongs_data.json.zip"
        ) as archive,
        archive.open("dugongs_data.json") as stream,
    ):
        data = json.load(stream)
    with pm.Model(coords={"animal": np.arange(1, data["N"] + 1)}) as model:
        x = pm.Data("x", np.asarray(data["x"], dtype=float), dims="animal")
        alpha = pm.Normal("alpha", 0, 1000, initval=2.5)
        beta = pm.Normal("beta", 0, 1000, initval=1.0)
        growth = pm.Uniform("lambda", 0.5, 1)
        tau = pm.Gamma("tau", alpha=0.0001, beta=0.0001, initval=1.0)
        sigma = pm.Deterministic("sigma", 1 / pm.math.sqrt(tau))
        pm.Normal(
            "Y",
            mu=alpha - beta * growth**x,
            sigma=sigma,
            observed=data["Y"],
            dims="animal",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
