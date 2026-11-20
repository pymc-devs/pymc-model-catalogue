"""
Model: Random Effect Logistic Regression for Seed Germination Proportion
Source: posteriordb/posterior_database/models/stan/seeds_centered_model.stan
Authors: Martin J. Crowder
Description: Logistic random-effects model for seed germination with overdispersion, centered
    coefficients and a direct scale prior.

Stan comparison:
- A: Stan = -128.6663088255901; PyMC = -127.97316164312568.
- B: Stan = -123.39810550522995; PyMC = -122.70495832276539.
- Raw delta: Stan = 5.268203320360158; PyMC = 5.268203320360286.
- Raw comparison: MATCH; residual = -1.2789769243681803e-13.
- Tolerance: 1.0526820332036028e-06
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
        c = pm.Normal("c", 0, sigma, dims="plate")
        # The source model subtracts the realized mean without removing the latent mean.
        b = pm.Deterministic("b", c - c.mean(), dims="plate")
        logit_p = alpha0 + alpha1 * x1 + alpha2 * x2 + alpha12 * x1 * x2 + b
        pm.Binomial("n", n=data["N"], logit_p=logit_p, observed=data["n"], dims="plate")
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
