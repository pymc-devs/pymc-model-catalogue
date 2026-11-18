"""
Model: Random Effect Logistic Regression for Seed Germination Proportion
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/seeds_centered_model.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/seeds_centered_model.py
Authors: Martin J. Crowder (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/seeds_centered_model.py
Upstream catalogue contributor: Kane Lindsay
Description: The model is essentially a random effects logistic, allowing for over-dispersion.
    Models the proportion of germination of different seeds in different root extracts. 'Stanified'
    & centered: tau replaced by sigma, direct estimation using narrower semi-informative priors,
    coefficients are centered

Data: seeds_data (upstream posterior: seeds_data-seeds_centered_model)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/seeds_data.json.zip
    Archive SHA-256: 06769b823f428e2713ecc67f9ebf0b670b19917038f8c4616e934cd647bb244c
    Local archive: data/posteriordb_seeds_data.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/seeds_data-seeds_centered_model.json
References: crowder1978beta (see posteriordb/source/bibliography/references.bib)
Reference URL: http://www.mrc-bsu.cam.ac.uk/wp-content/uploads/WinBUGS_Vol1.pdf

Implementation notes:
Native centered seed effects from posteriordb seeds_centered_model.

Stan comparison:
- A: Stan = -128.6663088255901; PyMC = -127.97316164312568.
- B: Stan = -123.39810550522995; PyMC = -122.70495832276539.
- Raw delta: Stan = 5.268203320360158; PyMC = 5.268203320360286.
- Raw comparison: MATCH; residual = -1.2789769243681803e-13.
- Tolerance: 1.0526820332036028e-06
- PyMC HalfCauchy(1) differs from Stan's positive Cauchy only by +log(2).
- Original Stan SHA-256: 37512dcebf0e156a8c186637dbdbfd1fb9fee18bcda2a9c5a47be2d0fc2867eb

Benchmark results:
- Not measured.
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
