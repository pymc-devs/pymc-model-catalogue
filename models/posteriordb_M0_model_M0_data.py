"""Inferring population size with constant detection probability.

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
            Path(__file__).parent / "data" / "posteriordb_M0_data.json.zip"
        ) as archive,
        archive.open("M0_data.json") as stream,
    ):
        data = json.load(stream)
    counts = np.asarray(data["y"], dtype="int64").sum(axis=1)
    occasions = int(data["T"])
    with pm.Model(coords={"individual": np.arange(data["M"])}) as model:
        omega = pm.Uniform("omega", 0, 1)
        p = pm.Uniform("p", 0, 1)
        pm.ZeroInflatedBinomial(
            "captures",
            psi=omega,
            n=occasions,
            p=p,
            observed=counts,
            dims="individual",
        )
        no_capture = (1 - p) ** occasions
        pm.Deterministic(
            "omega_nd", omega * no_capture / (omega * no_capture + 1 - omega)
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
