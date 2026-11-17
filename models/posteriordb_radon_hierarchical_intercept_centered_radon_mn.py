"""
Model: A county intercept model with county level covariate for the Radon dataset
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/radon_hierarchical_intercept_centered.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/radon_hierarchical_intercept_centered.py
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/radon_hierarchical_intercept_centered.py
Upstream catalogue contributor: Mans Magnusson
Description: A linear model with different intercepts per county and county level log_uppm
    covariate. This model is a centered parametrization.

Data: radon_mn (upstream posterior: radon_mn-radon_hierarchical_intercept_centered)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/radon_mn.json.zip
    Archive SHA-256: 70e5bacffb71955f1fc327f7c491b6bc59b09f40673830da306f1f9f9eb02bc0
    Local archive: data/posteriordb_radon_mn.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/radon_mn-radon_hierarchical_intercept_centered.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://mc-stan.org/users/documentation/case-studies/radon.html

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native radon model from the posteriordb radon case study.

The county hierarchy uses centered Normal effects.
Source: https://mc-stan.org/users/documentation/case-studies/radon.html

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
            Path(__file__).parent / "data" / "posteriordb_radon_mn.json.zip"
        ) as archive,
        archive.open("radon_mn.json") as stream,
    ):
        data = json.load(stream)
    coords = {"obs_id": np.arange(data["N"]), "county": np.arange(1, data["J"] + 1)}
    coords["coefficient"] = ("log_uppm", "floor_measure")
    with pm.Model(coords=coords) as model:
        county = pm.Data(
            "county_idx",
            np.asarray(data["county_idx"], dtype="int64") - 1,
            dims="obs_id",
        )
        floor = pm.Data(
            "floor_measure",
            np.asarray(data["floor_measure"], dtype=float),
            dims="obs_id",
        )
        uranium = pm.Data(
            "log_uppm", np.asarray(data["log_uppm"], dtype=float), dims="obs_id"
        )
        sigma_y = pm.HalfNormal("sigma_y", sigma=1, initval=1)
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1, initval=1)
        alpha = pm.Normal(
            "alpha",
            mu=mu_alpha,
            sigma=sigma_alpha,
            dims="county",
            initval=np.zeros(data["J"]),
        )
        beta = pm.Normal("beta", mu=0, sigma=10, dims="coefficient")
        pm.Normal(
            "log_radon",
            mu=alpha[county] + beta[0] * uranium + beta[1] * floor,
            sigma=sigma_y,
            observed=np.asarray(data["log_radon"], dtype=float),
            dims="obs_id",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
