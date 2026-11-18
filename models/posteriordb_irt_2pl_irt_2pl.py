"""
Model: Two Parameter Logistic Item Response Theory Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/irt_2pl.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/irt_2pl.py
Authors: Not specified in upstream model metadata.
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/irt_2pl.py
Upstream catalogue contributor: Mans Magnusson
Description: Hierarchical item response theory model.

Data: irt_2pl (upstream posterior: irt_2pl-irt_2pl)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/irt_2pl.json.zip
    Archive SHA-256: f29332b7aa0f0de2ea935867669fee3a03d7de51949ef5f7b6185d4f8cb6c617
    Local archive: data/posteriordb_irt_2pl.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/irt_2pl-irt_2pl.json
Reference URL: https://github.com/stan-dev/stat_comp_benchmarks/tree/master/benchmarks/irt_2pl

Implementation notes:
Native two-parameter logistic model from the posteriordb definition.

Stan comparison:
- A: Stan = -1548.950210050591; PyMC = -1546.870768538983.
- B: Stan = -1533.3844413914348; PyMC = -1531.3049998798278.
- Raw delta: Stan = 15.56576865915622; PyMC = 15.565768659155083.
- Raw comparison: MATCH; residual = 1.1368683772161603e-12.
- Tolerance: 1.1556576865915621e-06
- The three HalfCauchy versus positive Cauchy normalizers are constants.
- Original Stan SHA-256: 27947f5afa47a63c8587213c7d5598e701aa11209dff0b117d635578d028c922

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
            Path(__file__).parent / "data" / "posteriordb_irt_2pl.json.zip"
        ) as archive,
        archive.open("irt_2pl.json") as stream,
    ):
        data = json.load(stream)
    coords = {
        "item": np.arange(1, data["I"] + 1),
        "person": np.arange(1, data["J"] + 1),
    }
    with pm.Model(coords=coords) as model:
        sigma_theta = pm.HalfCauchy("sigma_theta", beta=2, initval=1)
        theta = pm.Normal("theta", mu=0, sigma=sigma_theta, dims="person")
        sigma_a = pm.HalfCauchy("sigma_a", beta=2, initval=1)
        a = pm.LogNormal(
            "a", mu=0, sigma=sigma_a, dims="item", initval=np.ones(data["I"])
        )
        mu_b = pm.Normal("mu_b", mu=0, sigma=5)
        sigma_b = pm.HalfCauchy("sigma_b", beta=2, initval=1)
        b = pm.Normal("b", mu=mu_b, sigma=sigma_b, dims="item")
        pm.Bernoulli(
            "y",
            logit_p=a[:, None] * (theta[None, :] - b[:, None]),
            observed=np.asarray(data["y"], dtype="int64"),
            dims=("item", "person"),
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
