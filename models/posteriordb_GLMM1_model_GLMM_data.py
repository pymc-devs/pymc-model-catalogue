"""Generalized Linear Mixed Model for Peregrine Population Size.

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
            Path(__file__).parent / "data" / "posteriordb_GLMM_data.json.zip"
        ) as archive,
        archive.open("GLMM_data.json") as stream,
    ):
        data = json.load(stream)
    coords = {
        "site": np.arange(1, data["nsite"] + 1),
        "observation": np.arange(data["nobs"]),
    }
    with pm.Model(coords=coords) as model:
        site = pm.Data(
            "site_index",
            np.asarray(data["obssite"], dtype="int64") - 1,
            dims="observation",
        )
        mu_alpha = pm.Normal("mu_alpha", 0, 10)
        sd_alpha = pm.Uniform("sd_alpha", 0, 5)
        alpha = pm.Normal("alpha", mu_alpha, sd_alpha, dims="site")
        pm.Poisson(
            "obs", mu=pm.math.exp(alpha[site]), observed=data["obs"], dims="observation"
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
