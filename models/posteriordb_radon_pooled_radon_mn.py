"""
Model: A pooled linear model for the Radon dataset
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/radon_pooled.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/radon_pooled.py
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/radon_pooled.py
Upstream catalogue contributor: Mans Magnusson
Description: A simple pooled linear model.

Data: radon_mn (upstream posterior: radon_mn-radon_pooled)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/radon_mn.json.zip
    Archive SHA-256: 70e5bacffb71955f1fc327f7c491b6bc59b09f40673830da306f1f9f9eb02bc0
    Local archive: data/posteriordb_radon_mn.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/radon_mn-radon_pooled.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://mc-stan.org/users/documentation/case-studies/radon.html

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native radon model from the posteriordb radon case study.

Source: https://mc-stan.org/users/documentation/case-studies/radon.html

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
            Path(__file__).parent / "data" / "posteriordb_radon_mn.json.zip"
        ) as archive,
        archive.open("radon_mn.json") as stream,
    ):
        data = json.load(stream)
    coords = {"obs_id": np.arange(data["N"])}
    with pm.Model(coords=coords) as model:
        floor = pm.Data(
            "floor_measure",
            np.asarray(data["floor_measure"], dtype=float),
            dims="obs_id",
        )
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma_y = pm.HalfNormal("sigma_y", sigma=1, initval=1)
        pm.Normal(
            "log_radon",
            mu=alpha + beta * floor,
            sigma=sigma_y,
            observed=np.asarray(data["log_radon"], dtype=float),
            dims="obs_id",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
