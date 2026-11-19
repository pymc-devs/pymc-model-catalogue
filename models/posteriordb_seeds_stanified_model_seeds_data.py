"""
Model: Random Effect Logistic Regression for Seed Germination Proportion
Source: posteriordb/posterior_database/models/stan/seeds_stanified_model.stan
Authors: Martin J. Crowder (cited work)
Description: Logistic random-effects model for seed germination with overdispersion and a direct
    scale prior instead of precision.

Stan comparison:
- A: Stan = -108.67160481789708; PyMC = -107.97845763543252.
- B: Stan = -125.66468137272972; PyMC = -124.97153419026515.
- Raw delta: Stan = -16.99307655483264; PyMC = -16.993076554832626.
- Raw comparison: MATCH; residual = -1.4210854715202004e-14.
- Tolerance: 1.1699307655483265e-06
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_seeds_data.json.zip"
        ) as archive,
        archive.open("seeds_data.json") as stream,
    ):
        data = json.load(stream)
    with pm.Model(coords={"plate": np.arange(1, data["I"] + 1)}) as model:
        x1 = pm.Data("x1", np.asarray(data["x1"], dtype=float), dims="plate")
        x2 = pm.Data("x2", np.asarray(data["x2"], dtype=float), dims="plate")
        alpha0 = pm.Normal("alpha0", 0, 1)
        alpha1 = pm.Normal("alpha1", 0, 1)
        alpha2 = pm.Normal("alpha2", 0, 1)
        alpha12 = pm.Normal("alpha12", 0, 1)
        sigma = pm.HalfCauchy("sigma", 1)
        b = pm.Normal("b", 0, sigma, dims="plate")
        logit_p = alpha0 + alpha1 * x1 + alpha2 * x2 + alpha12 * x1 * x2 + b
        pm.Binomial("n", n=data["N"], logit_p=logit_p, observed=data["n"], dims="plate")
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
