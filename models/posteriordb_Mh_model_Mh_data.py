"""Logistic-Normal Heterogeneity Model.

Stan comparison: MATCH.
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
