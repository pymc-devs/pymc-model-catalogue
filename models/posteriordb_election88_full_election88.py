"""
Model: Generalized Linear Mixed Effects Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/election88_full.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/election88_full.py
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/election88_full.py
Upstream catalogue contributor: Oliver Järnefelt
Description: y ~ logistic(black + female + v_prev_full + (1 | age) + (1 | age_edu) + (1 | state) +
    (1 | region_full))

Data: election88 (upstream posterior: election88-election88_full)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/election88.json.zip
    Archive SHA-256: a25fb13eb7f8db80bfc519b5baecc5ecfa3c7efec9ba6912f156bf472aaa1c7c
    Local archive: data/posteriordb_election88.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/election88-election88_full.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://raw.githubusercontent.com/stan-dev/example-models/master/ARM/Ch.14/election88_full.stan

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native crossed logistic effects from posteriordb election88_full.

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
