"""
Model: Success Rate of Peregrine broods
Source: posteriordb/posterior_database/models/stan/GLM_Binomial_model.stan
Authors: M Kery, M Schaub (cited work)
Description: Binomial GLM for modeling the success rate of Peregrine broods in the French Jura.

Stan comparison:
- A: Stan = -254.6454361190116; PyMC = -254.64543631086403.
- B: Stan = -327.0304780966763; PyMC = -327.03047828852914.
- Raw delta: Stan = -72.38504197766468; PyMC = -72.38504197766511.
- Raw comparison: MATCH; residual = 4.263256414560601e-13.
- Tolerance: 1.723850419776651e-06
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
