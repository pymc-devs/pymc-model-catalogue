"""
Model: A county intercept model with county level covariate for the Radon dataset
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/radon_hierarchical_intercept_centered.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/radon_hierarchical_intercept_centered.py
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/radon_hierarchical_intercept_centered.py
Upstream catalogue contributor: Mans Magnusson
Description: A linear model with different intercepts per county and county level log_uppm
    covariate. This model is a centered parametrization.

Data: radon_mn (upstream posterior: radon_mn-radon_hierarchical_intercept_centered)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/radon_mn.json.zip
    Archive SHA-256: 70e5bacffb71955f1fc327f7c491b6bc59b09f40673830da306f1f9f9eb02bc0
    Local archive: data/posteriordb_radon_mn.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/radon_mn-radon_hierarchical_intercept_centered.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://mc-stan.org/users/documentation/case-studies/radon.html

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native radon model from the posteriordb radon case study.

The county hierarchy uses centered Normal effects.
Source: https://mc-stan.org/users/documentation/case-studies/radon.html

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -1802.5181112703349; PyMC = -1801.1318170051404.
- B: Stan = -2069.328868008498; PyMC = -2067.942573743304.
- Raw delta: Stan = -266.81075673816304; PyMC = -266.8107567381635.
- Raw comparison: MATCH; residual = 4.547473508864641e-13.
- No parameter-dependent adjustment; only constant offsets may cancel.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 3.6681075673816346e-06
  using adjusted delta_Stan when an adjustment is present.
- Identity on all constrained free parameters, including original array order. No
  coordinate-change Jacobian or parameter-dependent factor correction. Stan support-
  transform and PyMC value-transform Jacobians are disabled.
- County labels are converted from Stan one-based to NumPy zero-based indices without
  reordering observations or county parameters. Residual SD and all varying-effect SDs
  retain their names and units.
- Each Stan positive-constrained Normal(0,1) SD becomes HalfNormal(1); each contributes
  the constant log(2), with no changed prior shape.
- Both sources retain centered county alpha and/or beta.
- beta order is [log_uppm, floor_measure] in both sources.
- Original Stan SHA-256: cab61c2340d2ba66b5f45f8b9a5ddf3bbe8e43ec19583661542f8e2a853dd14b
- Saved points: models/posteriordb_logp/points/radon_mn-radon_hierarchical_intercept_centered.npz
- Full evidence: models/posteriordb_logp/records/radon_mn-radon_hierarchical_intercept_centered.json
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
            Path(__file__).parent / "data" / "posteriordb_radon_mn.json.zip"
        ) as archive,
        archive.open("radon_mn.json") as stream,
    ):
        data = json.load(stream)
    coords = {"obs_id": np.arange(data["N"]), "county": np.arange(1, data["J"] + 1)}
    coords["coefficient"] = ("log_uppm", "floor_measure")
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
        uranium = pm.Data(
            "log_uppm", np.asarray(data["log_uppm"], dtype=float), dims="obs_id"
        )
        sigma_y = pm.HalfNormal("sigma_y", sigma=1, initval=1)
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1, initval=1)
        alpha = pm.Normal(
            "alpha",
            mu=mu_alpha,
            sigma=sigma_alpha,
            dims="county",
            initval=np.zeros(data["J"]),
        )
        beta = pm.Normal("beta", mu=0, sigma=10, dims="coefficient")
        pm.Normal(
            "log_radon",
            mu=alpha[county] + beta[0] * uranium + beta[1] * floor,
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
