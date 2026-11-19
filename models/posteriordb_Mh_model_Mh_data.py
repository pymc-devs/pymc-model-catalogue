"""
Model: Logistic-Normal Heterogeneity Model
Source: posteriordb/posterior_database/models/stan/Mh_model.stan
Authors: M Kery, M Schaub (cited work)
Description: Heterogeneity is inherent difference in the behaviour of individuals. Individual
    heterogeneity is modeled as random noise around some mean on a logit-transformed scale, and the
    model for the noise is the normal distribution.

Stan comparison:
- A: Stan = -753.2032385618709; PyMC = -754.8126765043747.
- B: Stan = -747.3552538441143; PyMC = -748.9646917866185.
- Raw delta: Stan = 5.847984717756617; PyMC = 5.847984717756276.
- Raw comparison: MATCH; residual = 3.410605131648481e-13.
- Tolerance: 1.058479847177566e-06
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_Mh_data.json.zip"
        ) as archive,
        archive.open("Mh_data.json") as stream,
    ):
        data = json.load(stream)
    counts = np.asarray(data["y"], dtype="int64")
    with pm.Model(coords={"individual": np.arange(data["M"])}) as model:
        omega = pm.Uniform("omega", 0, 1)
        mean_p = pm.Uniform("mean_p", 0, 1)
        sigma = pm.Uniform("sigma", 0, 5)
        eps_raw = pm.Normal("eps_raw", 0, 1, dims="individual")
        eps = pm.Deterministic(
            "eps", pm.math.logit(mean_p) + sigma * eps_raw, dims="individual"
        )
        pm.ZeroInflatedBinomial(
            "captures",
            psi=omega,
            n=data["T"],
            p=pm.math.sigmoid(eps),
            observed=counts,
            dims="individual",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
