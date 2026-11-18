"""
Model: Logistic Regression with Regularized Horseshoe Prior
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/logistic_regression_rhs.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/logistic_regression_rhs.py
Authors: Piironen, Juho and Vehtari, Aki and others (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/logistic_regression_rhs.py
Upstream catalogue contributor: Oliver Järnefelt
Description: y ~ Ber(x)

Data: ovarian (upstream posterior: ovarian-logistic_regression_rhs)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/ovarian.json.zip
    Archive SHA-256: 8cd0164e0508d5976ea1f47ed5f1a3e120a682ea949f21c3ff5d2c858e31f39d
    Local archive: data/posteriordb_ovarian.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/ovarian-logistic_regression_rhs.json
References: piironen2017sparsity (see posteriordb/source/bibliography/references.bib)
Reference URL:

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Regularized horseshoe logistic regression from Piironen and Vehtari (2017).

The global half-Student-t scale equals twice the dataset scale_global value.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -4399.5640593160515; PyMC = -3334.196820339009.
- B: Stan = -4569.633927230235; PyMC = -3504.2666882539274.
- Raw delta: Stan = -170.0698679141833; PyMC = -170.06986791491863.
- Raw comparison: MATCH; residual = 7.353264663834125e-10.
- No parameter-dependent adjustment; only constant offsets may cancel.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 2.7006986791491864e-06
  using adjusted delta_Stan when an adjustment is present.
- Identity on all constrained free parameters, including original array order. No
  coordinate-change Jacobian or parameter-dependent factor correction. Stan support-
  transform and PyMC value-transform Jacobians are disabled.
- beta0, z, tau, lambda and caux are unchanged. Both set c^2=slab_scale^2*caux and
  beta=z*tau*sqrt(c^2*lambda^2/(c^2+tau^2*lambda^2)); beta is not a density coordinate.
  Both use tau scale 2*scale_global and caux~InvGamma(slab_df/2,slab_df/2). PyMC's
  positive half-t normalizations for tau and lambda add only (d+1)*log(2) with d=1536.
- Original Stan SHA-256: b5c3167d819c155b81558c8889295c7bd1b1d4a7d137ce1bcd04783825c7b00b
- Saved points: models/posteriordb_logp/points/ovarian-logistic_regression_rhs.npz
- Full evidence: models/posteriordb_logp/records/ovarian-logistic_regression_rhs.json
- Two-point evidence only; no global equivalence, support or inference claim.

Benchmark results:
- Not measured.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_ovarian.json.zip"
        ) as archive,
        archive.open("ovarian.json") as stream,
    ):
        data = json.load(stream)
    coords = {"observation": range(data["n"]), "predictor": range(data["d"])}
    global_scale = 2 * data["scale_global"]
    slab_shape = 0.5 * data["slab_df"]
    with pm.Model(coords=coords) as model:
        x = pm.Data(
            "x",
            np.asarray(data["x"], dtype="float64"),
            dims=("observation", "predictor"),
        )
        beta0 = pm.Normal("beta0", mu=0, sigma=data["scale_icept"], initval=0)
        z = pm.Normal("z", mu=0, sigma=1, dims="predictor", initval=np.zeros(data["d"]))
        tau = pm.HalfStudentT(
            "tau", nu=data["nu_global"], sigma=global_scale, initval=global_scale
        )
        local_scale = pm.HalfStudentT(
            "lambda",
            nu=data["nu_local"],
            sigma=1,
            dims="predictor",
            initval=np.ones(data["d"]),
        )
        caux = pm.InverseGamma(
            "caux",
            alpha=slab_shape,
            beta=slab_shape,
            initval=slab_shape / (slab_shape + 1),
        )
        slab_variance = data["slab_scale"] ** 2 * caux
        regularized_local_scale = pt.sqrt(
            slab_variance * local_scale**2 / (slab_variance + tau**2 * local_scale**2)
        )
        beta = pm.Deterministic(
            "beta", z * tau * regularized_local_scale, dims="predictor"
        )
        pm.Bernoulli(
            "y",
            logit_p=beta0 + pt.dot(x, beta),
            observed=np.asarray(data["y"], dtype="int64"),
            dims="observation",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
