"""
Model: Two Parameter Logistic Item Response Theory Model
Source: posteriordb/posterior_database/models/stan/irt_2pl.stan
Description: Hierarchical item response theory model.

Stan comparison:
- A: Stan = -1548.950210050591; PyMC = -1546.870768538983.
- B: Stan = -1533.3844413914348; PyMC = -1531.3049998798278.
- Raw delta: Stan = 15.56576865915622; PyMC = 15.565768659155083.
- Raw comparison: MATCH; residual = 1.1368683772161603e-12.
- Tolerance: 1.1556576865915621e-06
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_irt_2pl.json.zip"
        ) as archive,
        archive.open("irt_2pl.json") as stream,
    ):
        data = json.load(stream)
    coords = {
        "item": np.arange(1, data["I"] + 1),
        "person": np.arange(1, data["J"] + 1),
    }
    with pm.Model(coords=coords) as model:
        sigma_theta = pm.HalfCauchy("sigma_theta", beta=2, initval=1)
        theta = pm.Normal("theta", mu=0, sigma=sigma_theta, dims="person")
        sigma_a = pm.HalfCauchy("sigma_a", beta=2, initval=1)
        a = pm.LogNormal(
            "a", mu=0, sigma=sigma_a, dims="item", initval=np.ones(data["I"])
        )
        mu_b = pm.Normal("mu_b", mu=0, sigma=5)
        sigma_b = pm.HalfCauchy("sigma_b", beta=2, initval=1)
        b = pm.Normal("b", mu=mu_b, sigma=sigma_b, dims="item")
        pm.Bernoulli(
            "y",
            logit_p=a[:, None] * (theta[None, :] - b[:, None]),
            observed=np.asarray(data["y"], dtype="int64"),
            dims=("item", "person"),
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
