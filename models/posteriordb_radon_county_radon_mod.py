"""Hierarchical Model.

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
            Path(__file__).parent / "data" / "posteriordb_radon_mod.json.zip"
        ) as archive,
        archive.open("radon_mod.json") as stream,
    ):
        data = json.load(stream)
    coords = {"obs_id": np.arange(data["N"]), "county": np.arange(1, data["J"] + 1)}
    with pm.Model(coords=coords) as model:
        county = pm.Data(
            "county_idx", np.asarray(data["county"], dtype="int64") - 1, dims="obs_id"
        )
        mu_a = pm.Normal("mu_a", mu=0, sigma=1)
        sigma_a = pm.Uniform("sigma_a", lower=0, upper=100, initval=1)
        sigma_y = pm.Uniform("sigma_y", lower=0, upper=100, initval=1)
        a = pm.Normal(
            "a", mu=mu_a, sigma=sigma_a, dims="county", initval=np.zeros(data["J"])
        )
        pm.Normal(
            "y",
            mu=a[county],
            sigma=sigma_y,
            observed=np.asarray(data["y"], dtype=float),
            dims="obs_id",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
