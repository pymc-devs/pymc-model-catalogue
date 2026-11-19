"""Gaussian Process regression.

Stan comparison: MATCH.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_gp_pois_regr.json.zip"
        ) as archive,
        archive.open("gp_pois_regr.json") as stream,
    ):
        data = json.load(stream)
    n = data["N"]
    coords = {"observation": np.arange(n), "input": ["x"]}
    with pm.Model(coords=coords) as model:
        x = pm.Data(
            "x",
            np.asarray(data["x"], dtype=float)[:, None],
            dims=("observation", "input"),
        )
        rho = pm.Gamma("rho", alpha=25, beta=4)
        alpha = pm.HalfNormal("alpha", sigma=2)
        sigma = pm.HalfNormal("sigma", sigma=1)
        kernel = alpha**2 * pm.gp.cov.ExpQuad(1, ls=rho)
        covariance = kernel(x) + sigma * pt.eye(n)
        pm.MvNormal(
            "y",
            mu=pt.zeros(n),
            cov=covariance,
            observed=np.asarray(data["y"]),
            dims="observation",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
