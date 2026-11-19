"""
Model: Success rate of a trial with prior and posterior inference
Source: posteriordb/posterior_database/models/stan/Rate_4_model.stan
Authors: M. D. Lee, E. Wagenmakers
Description: Inference of the success rate of a trial with prior and posterior inference

Stan comparison:
- A: Stan = -7.016003499815746; PyMC = -7.016003499815746.
- B: Stan = -7.579023772270681; PyMC = -7.57902377227068.
- Raw delta: Stan = -0.5630202724549349; PyMC = -0.563020272454934.
- Raw comparison: MATCH; residual = -8.881784197001252e-16.
- Tolerance: 1.0056302027245493e-06
"""

import json
from pathlib import Path
from zipfile import ZipFile

import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_Rate_4_data.json.zip"
        ) as archive,
        archive.open("Rate_4_data.json") as stream,
    ):
        data = json.load(stream)
    with pm.Model() as model:
        theta = pm.Beta("theta", alpha=1, beta=1)
        pm.Beta("thetaprior", alpha=1, beta=1)
        pm.Binomial("k", n=data["n"], p=theta, observed=data["k"])
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
