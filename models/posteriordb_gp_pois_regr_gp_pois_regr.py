"""
Model: Gaussian Process Poisson Regression
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/gp_pois_regr.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/gp_pois_regr.py
Authors: Not specified in upstream model metadata.
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/gp_pois_regr.py
Upstream catalogue contributor: Mans Magnusson
Description: Poisson regression with a one-dimensional latent Gaussian process using the exponential
    quadratic covariance function.

Data: gp_pois_regr (upstream posterior: gp_pois_regr-gp_pois_regr)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/gp_pois_regr.json.zip
    Archive SHA-256: 9aaaa9d5694b74e2aefa6d0bd2f0a48637fcb4d455a74b33985e58154f285da9
    Local archive: data/posteriordb_gp_pois_regr.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/gp_pois_regr-gp_pois_regr.json
Reference URL: https://github.com/stan-dev/stat_comp_benchmarks/tree/master/benchmarks/gp_pois_regr

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Latent Gaussian-process Poisson regression from posteriordb.

The native latent GP uses a noncentered Normal representation and the source jitter.
Source: https://github.com/stan-dev/stat_comp_benchmarks/tree/master/benchmarks/gp_pois_regr

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -1043.0656157679641; PyMC = -1042.3724671461957.
- B: Stan = -1044.0478221953438; PyMC = -1043.3546735735754.
- Raw delta: Stan = -0.9822064273796514; PyMC = -0.9822064273796514.
- Raw comparison: MATCH; residual = 0.0.
- No parameter-dependent adjustment; only constant offsets may cancel.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 1.0098220642737965e-06
  using adjusted delta_Stan when an adjustment is present.
- Set f_rotated_=f_tilde. Both whitened standard Normal vectors generate
  f=chol(K+1e-10*I)*z with K_ij=alpha^2*exp(-(x_i-x_j)^2/(2*rho^2)); neither density is
  evaluated in centered f coordinates, so there is no Cholesky determinant correction.
  PyMC gp/gp.py Latent._build_prior names the free whitened vector f_rotated_.
- HalfNormal versus a Normal density on positive alpha contributes only log(2).
- Original Stan SHA-256: d44142b5a950cfa68302de8d14c57e0ef0f9bcb7069375d22a74f84b7f78c270
- Saved points: models/posteriordb_logp/points/gp_pois_regr-gp_pois_regr.npz
- Full evidence: models/posteriordb_logp/records/gp_pois_regr-gp_pois_regr.json
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
            Path(__file__).parent / "data" / "posteriordb_gp_pois_regr.json.zip"
        ) as archive,
        archive.open("gp_pois_regr.json") as stream,
    ):
        data = json.load(stream)
    coords = {"observation": np.arange(data["N"]), "input": ["x"]}
    with pm.Model(coords=coords) as model:
        x = pm.Data(
            "x",
            np.asarray(data["x"], dtype=float)[:, None],
            dims=("observation", "input"),
        )
        rho = pm.Gamma("rho", alpha=25, beta=4)
        alpha = pm.HalfNormal("alpha", sigma=2)
        process = pm.gp.Latent(cov_func=alpha**2 * pm.gp.cov.ExpQuad(1, ls=rho))
        f = process.prior(
            "f", X=x, reparameterize=True, jitter=1e-10, dims="observation"
        )
        pm.Poisson(
            "k",
            mu=pm.math.exp(f),
            observed=np.asarray(data["k"], dtype=int),
            dims="observation",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
