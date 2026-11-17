"""
Model: Poisson GLM for modeling a population of Peregrines
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/GLM_Poisson_model.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/GLM_Poisson_model.py
Authors: M Kery, M Schaub (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/GLM_Poisson_model.py
Upstream catalogue contributor: Kane Lindsay
Description: Poisson Generalised Linear Model where the linear predictor is a cubic polynomial
    function of time.

Data: GLM_Poisson_Data (upstream posterior: GLM_Poisson_Data-GLM_Poisson_model)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/GLM_Poisson_Data.json.zip
    Archive SHA-256: 8d4e396989e85a80b3294a72ff6654328e530f1e9f50de967dba281804bd124e
    Local archive: data/posteriordb_GLM_Poisson_Data.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/GLM_Poisson_Data-GLM_Poisson_model.json
References: kery2011population (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/example-models/tree/master/BPA/Ch.03

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native cubic Poisson model from posteriordb GLM_Poisson_model.

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
            Path(__file__).parent / "data" / "posteriordb_GLM_Poisson_Data.json.zip"
        ) as archive,
        archive.open("GLM_Poisson_Data.json") as stream,
    ):
        data = json.load(stream)
    with pm.Model(coords={"year_id": np.arange(data["n"])}) as model:
        year = pm.Data("year", np.asarray(data["year"], dtype=float), dims="year_id")
        alpha = pm.Uniform("alpha", -20, 20)
        beta1 = pm.Uniform("beta1", -10, 10)
        beta2 = pm.Uniform("beta2", -10, 10)
        beta3 = pm.Uniform("beta3", -10, 10)
        log_lambda = alpha + year * (beta1 + year * (beta2 + year * beta3))
        pm.Poisson("C", mu=pm.math.exp(log_lambda), observed=data["C"], dims="year_id")
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
