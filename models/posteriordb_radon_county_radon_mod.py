"""
Model: Hierarchical Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/radon_county.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/radon_county.py
Authors: Bales, Ben and Pourzanjani, Arya and Vehtari, Aki and Petzold, Linda (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/radon_county.py
Upstream catalogue contributor: Oliver Järnefelt
Description: y ~ (1|county)

Data: radon_mod (upstream posterior: radon_mod-radon_county)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/radon_mod.json.zip
    Archive SHA-256: 341aa024485b6a7b1d7bf1ec1ae50124a59b38387e47a88db620ae37848aac66
    Local archive: data/posteriordb_radon_mod.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/radon_mod-radon_county.json
References: bales2019selecting (see posteriordb/source/bibliography/references.bib)
Reference URL: https://raw.githubusercontent.com/bbbales2/cmdstan-warmup/develop/examples/radon/radon.stan

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native county radon hierarchy from posteriordb, Bales et al. (2019).

The county model uses bounded uniform scale distributions.

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
            Path(__file__).parent / "data" / "posteriordb_radon_mod.json.zip"
        ) as archive,
        archive.open("radon_mod.json") as stream,
    ):
        data = json.load(stream)
    coords = {"obs_id": np.arange(data["N"]), "county": np.arange(1, data["J"] + 1)}
    with pm.Model(coords=coords) as model:
        county = pm.Data(
            "county_idx", np.asarray(data["county"], dtype="int64") - 1, dims="obs_id"
        )
        mu_a = pm.Normal("mu_a", mu=0, sigma=1)
        sigma_a = pm.Uniform("sigma_a", lower=0, upper=100, initval=1)
        sigma_y = pm.Uniform("sigma_y", lower=0, upper=100, initval=1)
        a = pm.Normal(
            "a", mu=mu_a, sigma=sigma_a, dims="county", initval=np.zeros(data["J"])
        )
        pm.Normal(
            "y",
            mu=a[county],
            sigma=sigma_y,
            observed=np.asarray(data["y"], dtype=float),
            dims="obs_id",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
