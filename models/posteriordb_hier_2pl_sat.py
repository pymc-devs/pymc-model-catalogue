"""
Model: Hierarchical Two-Parameter Logistic Item Response Model
Source: posteriordb/posterior_database/models/stan/hier_2pl.stan
Authors: Furr, Daniel (cited work)
Description: Hierachical item response theory model that includes parameters for both the difficulty
    and discrimination of items. Item parameter pairs are modeled as correlated draws from a
    bivariate normal distribution

Stan comparison:
- A: Stan = -14329.097424843068; PyMC = -14328.404277323887.
- B: Stan = -14105.817678299081; PyMC = -14105.1245307799.
- Raw delta: Stan = 223.27974654398713; PyMC = 223.27974654398713.
- Raw comparison: MATCH; residual = 0.0.
- Tolerance: 3.232797465439871e-06
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    with (
        ZipFile(Path(__file__).parent / "data" / "posteriordb_sat.json.zip") as archive,
        archive.open("sat.json") as stream,
    ):
        data = json.load(stream)
    item = np.asarray(data["ii"], dtype="int64") - 1
    person = np.asarray(data["jj"], dtype="int64") - 1
    coords = {
        "item": np.arange(1, data["I"] + 1),
        "person": np.arange(1, data["J"] + 1),
        "item_parameter": ["log_discrimination", "difficulty"],
        "response": np.arange(data["N"]),
    }
    with pm.Model(coords=coords) as model:
        theta = pm.Normal("theta", 0, 1, dims="person")
        mu = pm.Normal("mu", 0, [1, 5], dims="item_parameter")
        tau = pm.Exponential("tau", lam=0.1, dims="item_parameter", initval=np.ones(2))
        # A Beta(4, 4) correlation coordinate gives the two-dimensional LKJ(4) law.
        correlation_unit = pm.Beta("correlation_unit", alpha=4, beta=4)
        rho = pm.Deterministic("rho", 2 * correlation_unit - 1)
        correlation_chol = pt.stack(
            [
                pt.stack([pt.ones_like(rho), pt.zeros_like(rho)]),
                pt.stack([rho, pt.sqrt(1 - rho**2)]),
            ]
        )
        chol = tau[:, None] * correlation_chol
        xi = pm.MvNormal(
            "xi",
            mu=mu,
            chol=chol,
            dims=("item", "item_parameter"),
            initval=np.zeros((data["I"], 2)),
        )
        alpha = pm.Deterministic("alpha", pt.exp(xi[:, 0]), dims="item")
        beta = pm.Deterministic("beta", xi[:, 1], dims="item")
        pm.Bernoulli(
            "y",
            logit_p=alpha[item] * (theta[person] - beta[item]),
            observed=np.asarray(data["y"], dtype="int64"),
            dims="response",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
