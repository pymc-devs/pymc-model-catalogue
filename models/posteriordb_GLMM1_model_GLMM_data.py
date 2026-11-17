"""
Model: Generalized Linear Mixed Model for Peregrine Population Size
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/GLMM1_model.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/GLMM1_model.py
Authors: M Kery, M Schaub (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/GLMM1_model.py
Upstream catalogue contributor: Kane Lindsay
Description: A GLMM for modeling peregrine population size with a random effect being the site of
    observation.

Data: GLMM_data (upstream posterior: GLMM_data-GLMM1_model)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/GLMM_data.json.zip
    Archive SHA-256: f091c822557c0666234da07cb290ef130abbe86f7ca2fa030976fb95b6f509e1
    Local archive: data/posteriordb_GLMM_data.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/GLMM_data-GLMM1_model.json
References: kery2011population (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/example-models/blob/master/BPA/Ch.04

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native hierarchical Poisson site model from posteriordb GLMM1_model.

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
            Path(__file__).parent / "data" / "posteriordb_GLMM_data.json.zip"
        ) as archive,
        archive.open("GLMM_data.json") as stream,
    ):
        data = json.load(stream)
    coords = {
        "site": np.arange(1, data["nsite"] + 1),
        "observation": np.arange(data["nobs"]),
    }
    with pm.Model(coords=coords) as model:
        site = pm.Data(
            "site_index",
            np.asarray(data["obssite"], dtype="int64") - 1,
            dims="observation",
        )
        mu_alpha = pm.Normal("mu_alpha", 0, 10)
        sd_alpha = pm.Uniform("sd_alpha", 0, 5)
        alpha = pm.Normal("alpha", mu_alpha, sd_alpha, dims="site")
        pm.Poisson(
            "obs", mu=pm.math.exp(alpha[site]), observed=data["obs"], dims="observation"
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
