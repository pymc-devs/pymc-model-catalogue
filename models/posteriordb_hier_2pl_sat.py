"""
Model: Hierarchical Two-Parameter Logistic Item Response Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/hier_2pl.stan
Authors: Furr, Daniel (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/hier_2pl.py
Upstream catalogue contributor: Oliver Järnefelt
Description: Hierachical item response theory model that includes parameters for both the difficulty
    and discrimination of items. Item parameter pairs are modeled as correlated draws from a
    bivariate normal distribution

Data: sat (upstream posterior: sat-hier_2pl)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/sat.json.zip
    Archive SHA-256: 8a11aec0e71f152de6ca8b4bacf09c21d74d6c6d6c24423fdf1647950ec23a31
    Local archive: data/posteriordb_sat.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/sat-hier_2pl.json
References: furr2016hier (see posteriordb/source/bibliography/references.bib)
Reference URL: https://mc-stan.org/users/documentation/case-studies/hierarchical_2pl.html

Implementation notes:
Native hierarchical two-parameter logistic model from Furr (2016).

Item effects use the centered bivariate Gaussian law.

Stan comparison:
- A: Stan = -14329.097424843068; PyMC = -14328.404277323887.
- B: Stan = -14105.817678299081; PyMC = -14105.1245307799.
- Raw delta: Stan = 223.27974654398713; PyMC = 223.27974654398713.
- Raw comparison: MATCH; residual = 0.0.
- Tolerance: 3.232797465439871e-06
- Set xi[i,:]=[xi1[i],xi2[i]] and correlation_unit=(L_Omega[1,0]+1)/2.
- In dimension two LKJ(eta=4) has density proportional to (1-rho^2)^3; Beta(4,4) on
  u=(rho+1)/2 is its exact coordinate rewrite. log|d rho/d u|=log(2) is constant and
  omitted from delta adjustments; stacking the item vectors is a permutation with unit
  absolute determinant.
- Original Stan SHA-256: 86e17a66416ff8ffefa0c2cee1ccf07af46090e2c6177577eb7e4bd0e7cdf636

Benchmark results:
- Not measured.
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
