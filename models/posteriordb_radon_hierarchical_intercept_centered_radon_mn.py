"""A county intercept model with county level covariate for the Radon dataset.

Stan comparison:
- A: Stan = -1802.5181112703349; PyMC = -1801.1318170051404.
- B: Stan = -2069.328868008498; PyMC = -2067.942573743304.
- Raw delta: Stan = -266.81075673816304; PyMC = -266.8107567381635.
- Raw comparison: MATCH; residual = 4.547473508864641e-13.
- Tolerance: 3.6681075673816346e-06
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_radon_mn.json.zip"
        ) as archive,
        archive.open("radon_mn.json") as stream,
    ):
        data = json.load(stream)
    coords = {"obs_id": np.arange(data["N"]), "county": np.arange(1, data["J"] + 1)}
    coords["coefficient"] = ("log_uppm", "floor_measure")
    with pm.Model(coords=coords) as model:
        county = pm.Data(
            "county_idx",
            np.asarray(data["county_idx"], dtype="int64") - 1,
            dims="obs_id",
        )
        floor = pm.Data(
            "floor_measure",
            np.asarray(data["floor_measure"], dtype=float),
            dims="obs_id",
        )
        uranium = pm.Data(
            "log_uppm", np.asarray(data["log_uppm"], dtype=float), dims="obs_id"
        )
        sigma_y = pm.HalfNormal("sigma_y", sigma=1, initval=1)
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1, initval=1)
        alpha = pm.Normal(
            "alpha",
            mu=mu_alpha,
            sigma=sigma_alpha,
            dims="county",
            initval=np.zeros(data["J"]),
        )
        beta = pm.Normal("beta", mu=0, sigma=10, dims="coefficient")
        pm.Normal(
            "log_radon",
            mu=alpha[county] + beta[0] * uranium + beta[1] * floor,
            sigma=sigma_y,
            observed=np.asarray(data["log_radon"], dtype=float),
            dims="obs_id",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
