"""
Model: Generalized Linear Mixed Effects Model
Source: posteriordb/posterior_database/models/stan/election88_full.stan
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
Description: y ~ logistic(black + female + v_prev_full + (1 | age) + (1 | age_edu) + (1 | state) +
    (1 | region_full))

Stan comparison:
- A: Stan = -8556.772621101387; PyMC = -8579.798472670835.
- B: Stan = -8628.412684196437; PyMC = -8651.438535765885.
- Raw delta: Stan = -71.64006309505021; PyMC = -71.64006309505021.
- Raw comparison: MATCH; residual = 0.0.
- Tolerance: 1.716400630950502e-06
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_election88.json.zip"
        ) as archive,
        archive.open("election88.json") as stream,
    ):
        data = json.load(stream)
    groups = (
        ("age", "a"),
        ("edu", "b"),
        ("age_edu", "c"),
        ("state", "d"),
        ("region_full", "e"),
    )
    coords = {name: np.arange(1, data[f"n_{name}"] + 1) for name, _ in groups}
    coords.update(
        observation=np.arange(data["N"]),
        coefficient=["intercept", "black", "female", "previous_vote", "female_black"],
    )
    black = np.asarray(data["black"], dtype=float)
    female = np.asarray(data["female"], dtype=float)
    predictors = np.column_stack(
        (np.ones(data["N"]), black, female, data["v_prev_full"], female * black)
    )
    with pm.Model(coords=coords) as model:
        x = pm.Data("X", predictors, dims=("observation", "coefficient"))
        beta = pm.Normal("beta", 0, 100, dims="coefficient")
        logit_p = x @ beta
        for group, effect_name in groups:
            group_index = pm.Data(
                f"{group}_index",
                np.asarray(data[group], dtype="int64") - 1,
                dims="observation",
            )
            sigma = pm.Uniform(f"sigma_{effect_name}", 0, 100)
            effect = pm.Normal(effect_name, 0, sigma, dims=group)
            logit_p = logit_p + effect[group_index]
        pm.Bernoulli("y", logit_p=logit_p, observed=data["y"], dims="observation")
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
