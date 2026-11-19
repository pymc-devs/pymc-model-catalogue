"""Random Effects Model to Rank Hospitals on True Failure Probability.

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
            Path(__file__).parent / "data" / "posteriordb_surgical_data.json.zip"
        ) as archive,
        archive.open("surgical_data.json") as stream,
    ):
        data = json.load(stream)
    with pm.Model(coords={"hospital": np.arange(1, data["N"] + 1)}) as model:
        mu = pm.Normal("mu", 0, 1000)
        sigmasq = pm.InverseGamma("sigmasq", alpha=0.001, beta=0.001, initval=1.0)
        sigma = pm.Deterministic("sigma", pm.math.sqrt(sigmasq))
        b = pm.Normal("b", mu, sigma, dims="hospital")
        pm.Binomial("r", n=data["n"], logit_p=b, observed=data["r"], dims="hospital")
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
