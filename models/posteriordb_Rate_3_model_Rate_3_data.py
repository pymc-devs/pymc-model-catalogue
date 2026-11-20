"""
Model: Common rate of success from two trials
Source: posteriordb/posterior_database/models/stan/Rate_3_model.stan
Authors: M. D. Lee, E. Wagenmakers
Description: Inference of a common rate of success from two trials

Stan comparison:
- A: Stan = -3.756980106360462; PyMC = -3.756980106360465.
- B: Stan = -3.556700839794817; PyMC = -3.5567008397948205.
- Raw delta: Stan = 0.2002792665656452; PyMC = 0.20027926656564432.
- Raw comparison: MATCH; residual = 8.881784197001252e-16.
- Tolerance: 1.0020027926656564e-06
"""

import json
from pathlib import Path
from zipfile import ZipFile

import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_Rate_3_data.json.zip"
        ) as archive,
        archive.open("Rate_3_data.json") as stream,
    ):
        data = json.load(stream)
    with pm.Model() as model:
        theta = pm.Beta("theta", alpha=1, beta=1)
        pm.Binomial("k1", n=data["n1"], p=theta, observed=data["k1"])
        pm.Binomial("k2", n=data["n2"], p=theta, observed=data["k2"])
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
