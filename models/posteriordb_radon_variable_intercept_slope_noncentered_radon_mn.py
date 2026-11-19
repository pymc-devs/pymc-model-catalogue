"""Variable intercept and slope hierarchical Radon model (noncentered).

Stan comparison:
- A: Stan = -2250.622466411584; PyMC = -2248.543024933852.
- B: Stan = -2766.6835154946953; PyMC = -2764.6040740169633.
- Raw delta: Stan = -516.0610490831114; PyMC = -516.0610490831114.
- Raw comparison: MATCH; residual = 0.0.
- Tolerance: 6.1606104908311135e-06
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
        sigma_y = pm.HalfNormal("sigma_y", sigma=1, initval=1)
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1, initval=1)
        alpha_raw = pm.Normal("alpha_raw", mu=0, sigma=1, dims="county")
        alpha = pm.Deterministic(
            "alpha", mu_alpha + sigma_alpha * alpha_raw, dims="county"
        )
        mu_beta = pm.Normal("mu_beta", mu=0, sigma=10)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=1, initval=1)
        beta_raw = pm.Normal("beta_raw", mu=0, sigma=1, dims="county")
        beta = pm.Deterministic("beta", mu_beta + sigma_beta * beta_raw, dims="county")
        pm.Normal(
            "log_radon",
            mu=alpha[county] + beta[county] * floor,
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
