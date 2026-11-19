"""Mixed Model to Predict Population Size with Random Site and Year Effects.

Stan comparison:
- A: Stan = -26260.430777006368; PyMC = -26260.430777173864.
- B: Stan = -22924.862764090765; PyMC = -22924.86276425826.
- Raw delta: Stan = 3335.5680129156026; PyMC = 3335.5680129156026.
- Raw comparison: MATCH; residual = 0.0.
- Tolerance: 3.435568012915602e-05
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
