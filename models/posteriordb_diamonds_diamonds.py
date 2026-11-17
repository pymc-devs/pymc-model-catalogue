"""
Model: Multiple Highly Correlated Predictors Log-Log Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/diamonds.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/diamonds.py
Authors: Bales, Ben and Pourzanjani, Arya and Vehtari, Aki and Petzold, Linda (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/diamonds.py
Upstream catalogue contributor: Oliver Järnefelt
Description: log(price) ~ carat * (log(x) + log(y) + log(z)) + cut + color + clarity

Data: diamonds (upstream posterior: diamonds-diamonds)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/diamonds.json.zip
    Archive SHA-256: 127d44a642f954b707bec2d82725e168d57a412d59487708794d4a17097f26ba
    Local archive: data/posteriordb_diamonds.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/diamonds-diamonds.json
References: bales2019selecting (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/bbbales2/cmdstan-warmup/blob/develop/examples/diamonds

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native Gaussian regression for diamond prices.

Source: Bales et al. (2019), with the ggplot2 diamonds data.
The intercept has its density at the centered predictor origin.

Stan comparison:
- Not recorded.

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
