"""A centered hierarchical model for 8 schools.

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
            Path(__file__).parent / "data" / "posteriordb_eight_schools.json.zip"
        ) as archive,
        archive.open("eight_schools.json") as stream,
    ):
        data = json.load(stream)
    with pm.Model(coords={"school": np.arange(1, data["J"] + 1)}) as model:
        mu = pm.Normal("mu", 0, 5)
        tau = pm.HalfCauchy("tau", 5)
        theta = pm.Normal("theta", mu, tau, dims="school")
        pm.Normal(
            "y",
            theta,
            sigma=np.asarray(data["sigma"], dtype=float),
            observed=data["y"],
            dims="school",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
