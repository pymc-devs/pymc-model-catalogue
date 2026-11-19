"""Logistic Regression Model for Voting Preference based on Income.

Stan comparison:
- A: Stan = -823.8426202699114; PyMC = -823.8426202699114.
- B: Stan = -816.3904366646025; PyMC = -816.3904366646025.
- Raw delta: Stan = 7.45218360530896; PyMC = 7.45218360530896.
- Raw comparison: MATCH; residual = 0.0.
- Tolerance: 1.0745218360530895e-06
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_nes_logit_data.json.zip"
        ) as archive,
        archive.open("nes_logit_data.json") as stream,
    ):
        data = json.load(stream)
    income = np.asarray(data["income"], dtype=float)
    vote = np.asarray(data["vote"], dtype=int)
    coords = {"obs_id": np.arange(len(vote)), "coefficient": ["income"]}
    with pm.Model(coords=coords) as model:
        x = pm.Data("income", income, dims="obs_id")
        alpha = pm.Flat("alpha")
        beta = pm.Flat("beta", dims="coefficient")
        pm.Bernoulli("vote", logit_p=alpha + beta[0] * x, observed=vote, dims="obs_id")
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
