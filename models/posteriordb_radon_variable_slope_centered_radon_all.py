"""
Model: Variable slope hiearchical Radon model (centered)
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/radon_variable_slope_centered.stan
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/radon_variable_slope_centered.py
Upstream catalogue contributor: Mans Magnusson
Description: A linear model with a global intercept and different regression coefficients using an
    hiearchical prior. This model uses a centered parametrization.

Data: radon_all (upstream posterior: radon_all-radon_variable_slope_centered)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/radon_all.json.zip
    Archive SHA-256: 3f30c7909d530be01e70ab9e98f9f5d5e83371bb15c6dd168696aefd805b5672
    Local archive: data/posteriordb_radon_all.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/radon_all-radon_variable_slope_centered.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://mc-stan.org/users/documentation/case-studies/radon.html

Implementation notes:
Native radon model from the posteriordb radon case study.

The county hierarchy uses centered Normal effects.
Source: https://mc-stan.org/users/documentation/case-studies/radon.html

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -24633.010587621415; PyMC = -24631.62429332425.
- B: Stan = -25790.900926084814; PyMC = -25789.514631787664.
- Raw delta: Stan = -1157.8903384633995; PyMC = -1157.890338463414.
- Raw comparison: MATCH; residual = 1.4551915228366852e-11.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 1.2578903384634142e-05
- County labels are converted from Stan one-based to NumPy zero-based indices.
- Each Stan positive-constrained Normal(0,1) SD becomes HalfNormal(1); each contributes
  the constant log(2).
- Original Stan SHA-256: 67b12c10b03d91e0e487ab7b072bfb2f9a6539bda62db0329e6e8220965506dc
- Saved points: models/posteriordb_logp/points/radon_all-radon_variable_slope_centered.npz
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
    coords = {"obs_id": np.arange(data["N"]), "county": np.arange(1, data["J"] + 1)}
    with pm.Model(coords=coords) as model:
        county = pm.Data(
            "county_idx",
            np.asarray(data["county_idx"], dtype="int64") - 1,
            dims="obs_id",
        )
        floor = pm.Data(
            "floor_measure",
            np.asarray(data["floor_measure"], dtype=float),
            dims="obs_id",
        )
        sigma_y = pm.HalfNormal("sigma_y", sigma=1, initval=1)
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        mu_beta = pm.Normal("mu_beta", mu=0, sigma=10)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=1, initval=1)
        beta = pm.Normal(
            "beta",
            mu=mu_beta,
            sigma=sigma_beta,
            dims="county",
            initval=np.zeros(data["J"]),
        )
        pm.Normal(
            "log_radon",
            mu=alpha + beta[county] * floor,
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
