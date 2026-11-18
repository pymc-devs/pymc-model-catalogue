"""
Model: Success Rate of Peregrine broods
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/GLM_Binomial_model.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/GLM_Binomial_model.py
Authors: M Kery, M Schaub (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/GLM_Binomial_model.py
Upstream catalogue contributor: Kane Lindsay
Description: Binomial GLM for modeling the success rate of Peregrine broods in the French Jura.

Data: GLM_Binomial_data (upstream posterior: GLM_Binomial_data-GLM_Binomial_model)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/GLM_Binomial_data.json.zip
    Archive SHA-256: 09fc115d0825f3bed893d3836ad91146fffa131c7c3621218a95e72ba39a14ac
    Local archive: data/posteriordb_GLM_Binomial_data.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/GLM_Binomial_data-GLM_Binomial_model.json
References: kery2011population (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/example-models/tree/master/BPA/Ch.03

Implementation notes:
Native quadratic binomial model from posteriordb GLM_Binomial_model.

Stan comparison:
- A: Stan = -254.6454361190116; PyMC = -254.64543631086403.
- B: Stan = -327.0304780966763; PyMC = -327.03047828852914.
- Raw delta: Stan = -72.38504197766468; PyMC = -72.38504197766511.
- Raw comparison: MATCH; residual = 4.263256414560601e-13.
- Tolerance: 1.723850419776651e-06
- Original Stan SHA-256: 1cd08265b53a2faea2087055fd5c24eff965c6581c6ce78479a0925ddd3a1144

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
            Path(__file__).parent / "data" / "posteriordb_GLM_Binomial_data.json.zip"
        ) as archive,
        archive.open("GLM_Binomial_data.json") as stream,
    ):
        data = json.load(stream)
    with pm.Model(coords={"year_id": np.arange(data["nyears"])}) as model:
        year = pm.Data("year", np.asarray(data["year"], dtype=float), dims="year_id")
        alpha = pm.Normal("alpha", 0, 100)
        beta1 = pm.Normal("beta1", 0, 100)
        beta2 = pm.Normal("beta2", 0, 100)
        pm.Binomial(
            "C",
            n=data["N"],
            logit_p=alpha + year * (beta1 + year * beta2),
            observed=data["C"],
            dims="year_id",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
