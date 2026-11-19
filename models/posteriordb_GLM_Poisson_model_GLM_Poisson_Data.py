"""Poisson GLM for modeling a population of Peregrines.

Stan comparison:
- A: Stan = -19247.726566493642; PyMC = -19260.402643382677.
- B: Stan = -14928.78196329504; PyMC = -14941.458040184076.
- Raw delta: Stan = 4318.944603198603; PyMC = 4318.944603198601.
- Raw comparison: MATCH; residual = 1.8189894035458565e-12.
- Tolerance: 4.4189446031986024e-05
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
