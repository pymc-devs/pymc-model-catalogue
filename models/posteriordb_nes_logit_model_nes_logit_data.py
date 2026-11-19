"""Logistic Regression Model for Voting Preference based on Income.

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
