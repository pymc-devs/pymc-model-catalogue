"""
Model: Linear Mixed Effects Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/pilots.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/pilots.py
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/pilots.py
Upstream catalogue contributor: Oliver Järnefelt
Description: y ~ 1 + (1 | group) + (1 | scenario))

Data: pilots (upstream posterior: pilots-pilots)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/pilots.json.zip
    Archive SHA-256: b596d205c5393b61e923bb6d920c3a39f1c22b3968f7e86925a931c9006cc6f9
    Local archive: data/posteriordb_pilots.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/pilots-pilots.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://raw.githubusercontent.com/stan-dev/example-models/master/ARM/Ch.14/pilots.stan

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native crossed Gaussian effects for pilot groups and scenarios.

Source: Gelman and Hill (2006).
The group and scenario effects retain their centered parameterization.

Stan comparison:
- Not recorded.

Benchmark results:
- Not measured.
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
