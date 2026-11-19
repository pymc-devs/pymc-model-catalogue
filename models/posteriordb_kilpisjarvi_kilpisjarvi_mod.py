"""Multiple Highly Correlated Predictors Linear Model.

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
            Path(__file__).parent / "data" / "posteriordb_kilpisjarvi_mod.json.zip"
        ) as archive,
        archive.open("kilpisjarvi_mod.json") as stream,
    ):
        data = json.load(stream)
    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    with pm.Model(coords={"year": np.arange(int(data["N"]))}) as model:
        alpha = pm.Normal("alpha", mu=data["pmualpha"], sigma=data["psalpha"])
        beta = pm.Normal("beta", mu=data["pmubeta"], sigma=data["psbeta"])
        sigma = pm.HalfFlat("sigma", initval=1)
        time = pm.Data("x", x, dims="year")
        pm.Normal("y", mu=alpha + beta * time, sigma=sigma, observed=y, dims="year")
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
