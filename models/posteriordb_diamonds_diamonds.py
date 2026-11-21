"""
Model: Multiple Highly Correlated Predictors Log-Log Model
Source: posteriordb/posterior_database/models/stan/diamonds.stan
Authors: Ben Bales, Arya Pourzanjani, Aki Vehtari, Linda Petzold
Description: log(price) ~ carat * (log(x) + log(y) + log(z)) + cut + color + clarity

Stan comparison:
- A: Stan = -217519.159394572; PyMC = -217519.15939455116.
- B: Stan = -182886.15666732893; PyMC = -182886.15666730888.
- Raw delta: Stan = 34633.00272724306; PyMC = 34633.002727242274.
- Raw comparison: MATCH; residual = 7.8580342233181e-10.
- Tolerance: 0.00034733002727243065
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_diamonds.json.zip"
        ) as archive,
        archive.open("diamonds.json") as stream,
    ):
        data = json.load(stream)
    response = np.asarray(data["Y"], dtype=float)
    predictors = np.asarray(data["X"], dtype=float)[:, 1:]
    predictor_means = predictors.mean(axis=0)
    centered_predictors = predictors - predictor_means
    coords = {
        "obs_id": np.arange(len(response)),
        "coefficient": np.arange(1, int(data["K"])),
    }
    with pm.Model(coords=coords) as model:
        x = pm.Data(
            "centered_predictors",
            centered_predictors,
            dims=("obs_id", "coefficient"),
        )
        b = pm.Normal("b", mu=0, sigma=1, dims="coefficient")
        intercept = pm.StudentT("Intercept", nu=3, mu=8, sigma=10)
        sigma = pm.HalfStudentT(
            "sigma", nu=3, sigma=10, initval=max(float(response.std()), 1.0)
        )
        pm.Deterministic("b_Intercept", intercept - pm.math.dot(predictor_means, b))
        if not data["prior_only"]:
            pm.Normal(
                "Y",
                mu=intercept + pm.math.dot(x, b),
                sigma=sigma,
                observed=response,
                dims="obs_id",
            )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
