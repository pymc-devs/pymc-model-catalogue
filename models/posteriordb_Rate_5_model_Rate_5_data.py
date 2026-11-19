"""
Model: Common Rate of Success From Two Trials with Posterior Predictives
Source: posteriordb/posterior_database/models/stan/Rate_5_model.stan
Authors: M. D. Lee, E. Wagenmakers
Description: Inference of a common rate of success from two trials with posterior predictive for
    each trial for checking the adequacy of the model.

Stan comparison:
- A: Stan = -13.865169993120485; PyMC = -13.865169993120485.
- B: Stan = -13.87246475492653; PyMC = -13.87246475492653.
- Raw delta: Stan = -0.007294761806043937; PyMC = -0.007294761806043937.
- Raw comparison: MATCH; residual = 0.0.
- Tolerance: 1.0000729476180604e-06
"""

import json
from pathlib import Path
from zipfile import ZipFile

import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_Rate_5_data.json.zip"
        ) as archive,
        archive.open("Rate_5_data.json") as stream,
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
