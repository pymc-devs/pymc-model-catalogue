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

Data: radon_all (upstream posterior: radon_all-radon_pooled)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/radon_all.json.zip
    Archive SHA-256: 3f30c7909d530be01e70ab9e98f9f5d5e83371bb15c6dd168696aefd805b5672
    Local archive: data/posteriordb_radon_all.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/radon_all-radon_pooled.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://mc-stan.org/users/documentation/case-studies/radon.html

Implementation notes:
Native radon model from the posteriordb radon case study.

Source: https://mc-stan.org/users/documentation/case-studies/radon.html

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -24706.489744389342; PyMC = -24705.796597272732.
- B: Stan = -22623.248817100746; PyMC = -22622.555669984133.
- Raw delta: Stan = 2083.240927288596; PyMC = 2083.2409272886.
- Raw comparison: MATCH; residual = -3.637978807091713e-12.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 2.1832409272885998e-05
- County labels are converted from Stan one-based to NumPy zero-based indices.
- Each Stan positive-constrained Normal(0,1) SD becomes HalfNormal(1); each contributes
  the constant log(2).
- Original Stan SHA-256: 680277f397ca17ab3603fa8acb0042fe547cdedfd19d4bc22aedd40e3ae7d0de
- Saved points: models/posteriordb_logp/points/radon_all-radon_pooled.npz
- Two-point evidence only; no global equivalence, support or inference claim.

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
            Path(__file__).parent / "data" / "posteriordb_radon_all.json.zip"
        ) as archive,
        archive.open("radon_all.json") as stream,
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
