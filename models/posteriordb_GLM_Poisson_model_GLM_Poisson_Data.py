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

Implementation notes:
Native cubic Poisson model from posteriordb GLM_Poisson_model.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -19247.726566493642; PyMC = -19260.402643382677.
- B: Stan = -14928.78196329504; PyMC = -14941.458040184076.
- Raw delta: Stan = 4318.944603198603; PyMC = 4318.944603198601.
- Raw comparison: MATCH; residual = 1.8189894035458565e-12.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 4.4189446031986024e-05
- Stan implicit bounded coefficients versus PyMC Uniform priors add only
  -log(40)-3*log(20).
- Original Stan SHA-256: 84186ad779e2add1b934a00e211d9e642ae29cba0e02caf3fedb5c1b29e54f85
- Saved points: models/posteriordb_logp/points/GLM_Poisson_Data-GLM_Poisson_model.npz
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
