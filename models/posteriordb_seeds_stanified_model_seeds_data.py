"""
Model: Random Effect Logistic Regression for Seed Germination Proportion
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/seeds_stanified_model.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/seeds_stanified_model.py
Authors: Martin J. Crowder (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/seeds_stanified_model.py
Upstream catalogue contributor: Kane Lindsay
Description: The model is essentially a random effects logistic, allowing for over-dispersion.
    Models the proportion of germination of different seeds in different root extracts. 'Stanified':
    tau replaced by sigma, direct estimation using narrower semi-informative priors

Data: seeds_data (upstream posterior: seeds_data-seeds_stanified_model)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/seeds_data.json.zip
    Archive SHA-256: 06769b823f428e2713ecc67f9ebf0b670b19917038f8c4616e934cd647bb244c
    Local archive: data/posteriordb_seeds_data.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/seeds_data-seeds_stanified_model.json
References: crowder1978beta (see posteriordb/source/bibliography/references.bib)
Reference URL: http://www.mrc-bsu.cam.ac.uk/wp-content/uploads/WinBUGS_Vol1.pdf

Implementation notes:
Native scale-based seed effects from posteriordb seeds_stanified_model.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -108.67160481789708; PyMC = -107.97845763543252.
- B: Stan = -125.66468137272972; PyMC = -124.97153419026515.
- Raw delta: Stan = -16.99307655483264; PyMC = -16.993076554832626.
- Raw comparison: MATCH; residual = -1.4210854715202004e-14.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 1.1699307655483265e-06
- Positive Cauchy versus HalfCauchy(1) changes only the constant +log(2), not the sigma-
  dependent kernel.
- Original Stan SHA-256: a57822f654e024467fb7522ae013de10f0a4ba14797255b8c172c4418e57ecc4
- Saved points: models/posteriordb_logp/points/seeds_data-seeds_stanified_model.npz
- Two-point evidence only; no global equivalence, support or inference claim.

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
        b = pm.Normal("b", 0, sigma, dims="plate")
        logit_p = alpha0 + alpha1 * x1 + alpha2 * x2 + alpha12 * x1 * x2 + b
        pm.Binomial("n", n=data["N"], logit_p=logit_p, observed=data["n"], dims="plate")
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
