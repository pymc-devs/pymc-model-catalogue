"""Hierarchical intercept model for the Radon dataset (centered).

Stan comparison:
- A: Stan = -2179.855340272635; PyMC = -2178.4690459434896.
- B: Stan = -2120.5815623520916; PyMC = -2119.1952680229465.
- Raw delta: Stan = 59.27377792054358; PyMC = 59.273777920543125.
- Raw comparison: MATCH; residual = 4.547473508864641e-13.
- Tolerance: 1.5927377792054357e-06
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
        pm.Normal(
            "log_radon",
            mu=alpha[county],
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
