"""
Model: Variable slope hiearchical Radon model (non-centered)
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/radon_variable_slope_noncentered.stan
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/radon_variable_slope_noncentered.py
Upstream catalogue contributor: Mans Magnusson
Description: A linear model with a global intercept and different regression coefficients using an
    hiearchical prior. This model uses a non-centered parametrization.

Data: radon_all (upstream posterior: radon_all-radon_variable_slope_noncentered)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/radon_all.json.zip
    Archive SHA-256: 3f30c7909d530be01e70ab9e98f9f5d5e83371bb15c6dd168696aefd805b5672
    Local archive: data/posteriordb_radon_all.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/radon_all-radon_variable_slope_noncentered.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://mc-stan.org/users/documentation/case-studies/radon.html

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native radon model from the posteriordb radon case study.

Standard Normal effects define the noncentered county hierarchy.
Source: https://mc-stan.org/users/documentation/case-studies/radon.html

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -28837.14039356615; PyMC = -28835.75409926897.
- B: Stan = -28430.70948304006; PyMC = -28429.32318874293.
- Raw delta: Stan = 406.4309105260909; PyMC = 406.43091052603995.
- Raw comparison: MATCH; residual = 5.093170329928398e-11.
- No parameter-dependent adjustment; only constant offsets may cancel.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 5.0643091052609084e-06
  using adjusted delta_Stan when an adjustment is present.
- Identity on all constrained free parameters, including original array order. No
  coordinate-change Jacobian or parameter-dependent factor correction. Stan support-
  transform and PyMC value-transform Jacobians are disabled.
- County labels are converted from Stan one-based to NumPy zero-based indices without
  reordering observations or county parameters. Residual SD and all varying-effect SDs
  retain their names and units.
- Each Stan positive-constrained Normal(0,1) SD becomes HalfNormal(1); each contributes
  the constant log(2), with no changed prior shape.
- Keep alpha_raw and/or beta_raw as density coordinates: both sources form
  alpha=mu_alpha+sigma_alpha*alpha_raw and/or beta=mu_beta+sigma_beta*beta_raw. No
  centered/noncentered conversion or scale Jacobian is applied between these paired
  sources.
- Original Stan SHA-256: df643cc1f652bcd53376378e80a607e1e1622f8eb515f9ca67525b9fd4347891
- Saved points: models/posteriordb_logp/points/radon_all-radon_variable_slope_noncentered.npz
- Full evidence: models/posteriordb_logp/records/radon_all-radon_variable_slope_noncentered.json
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
        beta_raw = pm.Normal("beta_raw", mu=0, sigma=1, dims="county")
        beta = pm.Deterministic("beta", mu_beta + sigma_beta * beta_raw, dims="county")
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
