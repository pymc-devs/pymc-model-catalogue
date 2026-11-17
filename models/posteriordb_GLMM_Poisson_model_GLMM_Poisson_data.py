"""
Model: Mixed Model to Predict Population Size with Random Site and Year Effects
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/GLMM_Poisson_model.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/GLMM_Poisson_model.py
Authors: M Kery, M Schaub (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/GLMM_Poisson_model.py
Upstream catalogue contributor: Kane Lindsay
Description: A GLMM for modeling peregrine population size with a random effect being the site and
    year of observation using the Poisson distribution to model counts.

Data: GLMM_Poisson_data (upstream posterior: GLMM_Poisson_data-GLMM_Poisson_model)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/GLMM_Poisson_data.json.zip
    Archive SHA-256: d44a8283544156f540a3fa0240748977fa9ac7d8b6bdf895155da315f8508cd3
    Local archive: data/posteriordb_GLMM_Poisson_data.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/GLMM_Poisson_data-GLMM_Poisson_model.json
References: kery2011population (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/example-models/blob/master/BPA/Ch.04

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native cubic Poisson mixed model from posteriordb GLMM_Poisson_model.

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
            Path(__file__).parent / "data" / "posteriordb_GLMM_Poisson_data.json.zip"
        ) as archive,
        archive.open("GLMM_Poisson_data.json") as stream,
    ):
        data = json.load(stream)
    with pm.Model(coords={"year_id": np.arange(data["n"])}) as model:
        year = pm.Data("year", np.asarray(data["year"], dtype=float), dims="year_id")
        alpha = pm.Uniform("alpha", -20, 20)
        beta1 = pm.Uniform("beta1", -10, 10)
        beta2 = pm.Uniform("beta2", -10, 10)
        beta3 = pm.Uniform("beta3", -10, 10)
        sigma = pm.Uniform("sigma", 0, 5)
        eps = pm.Normal("eps", 0, sigma, dims="year_id")
        log_lambda = alpha + year * (beta1 + year * (beta2 + year * beta3)) + eps
        pm.Poisson("C", mu=pm.math.exp(log_lambda), observed=data["C"], dims="year_id")
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
