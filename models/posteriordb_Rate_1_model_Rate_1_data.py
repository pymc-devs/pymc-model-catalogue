"""
Model: Predicting the rate of correct test question answers
Source: posteriordb/posterior_database/models/stan/Rate_1_model.stan
Authors: M. D. Lee, E. Wagenmakers
Description: Predicting the rate of correct answers for one subject over questions with the same
    difficulty.

Stan comparison:
- A: Stan = -1.5191739432523104; PyMC = -1.519173943252311.
- B: Stan = -1.4755165919307762; PyMC = -1.4755165919307773.
- Raw delta: Stan = 0.043657351321534144; PyMC = 0.0436573513215337.
- Raw comparison: MATCH; residual = 4.440892098500626e-16.
- Tolerance: 1.0004365735132153e-06
"""

import json
from pathlib import Path
from zipfile import ZipFile

import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_Rate_1_data.json.zip"
        ) as archive,
        archive.open("Rate_1_data.json") as stream,
    ):
        data = json.load(stream)
    with pm.Model() as model:
        theta = pm.Beta("theta", alpha=1, beta=1)
        pm.Binomial("k", n=data["n"], p=theta, observed=data["k"])
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
