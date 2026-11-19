"""
Model: Linear Mixed Effects Model
Source: posteriordb/posterior_database/models/stan/pilots.stan
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
Description: y ~ 1 + (1 | group) + (1 | scenario))

Stan comparison:
- A: Stan = -255.80167956170123; PyMC = -269.6171903115181.
- B: Stan = -259.8135650165997; PyMC = -273.6290757664166.
- Raw delta: Stan = -4.011885454898447; PyMC = -4.011885454898504.
- Raw comparison: MATCH; residual = 5.684341886080802e-14.
- Tolerance: 1.040118854548985e-06
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_pilots.json.zip"
        ) as archive,
        archive.open("pilots.json") as stream,
    ):
        data = json.load(stream)
    response = np.asarray(data["y"], dtype=float)
    group_id = np.asarray(data["group_id"], dtype=int) - 1
    scenario_id = np.asarray(data["scenario_id"], dtype=int) - 1
    coords = {
        "obs_id": np.arange(len(response)),
        "group": np.arange(1, int(data["n_groups"]) + 1),
        "scenario": np.arange(1, int(data["n_scenarios"]) + 1),
    }
    with pm.Model(coords=coords) as model:
        group_index = pm.Data("group_id", group_id, dims="obs_id")
        scenario_index = pm.Data("scenario_id", scenario_id, dims="obs_id")
        mu_a = pm.Normal("mu_a", mu=0, sigma=1)
        mu_b = pm.Normal("mu_b", mu=0, sigma=1)
        sigma_a = pm.Uniform("sigma_a", lower=0, upper=100)
        sigma_b = pm.Uniform("sigma_b", lower=0, upper=100)
        sigma_y = pm.Uniform("sigma_y", lower=0, upper=100)
        a = pm.Normal("a", mu=10 * mu_a, sigma=sigma_a, dims="group")
        b = pm.Normal("b", mu=10 * mu_b, sigma=sigma_b, dims="scenario")
        pm.Normal(
            "y",
            mu=a[group_index] + b[scenario_index],
            sigma=sigma_y,
            observed=response,
            dims="obs_id",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
