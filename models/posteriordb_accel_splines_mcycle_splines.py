"""
Model: Splines for time-series data with varying mean and std
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/accel_splines.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/accel_splines.py
Authors: Bales, Ben and Pourzanjani, Arya and Vehtari, Aki and Petzold, Linda (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/accel_splines.py
Upstream catalogue contributor: Oliver Järnefelt
Description: accel ~ s(times, k=40), sigma ~ s(times, k=40)

Data: mcycle_splines (upstream posterior: mcycle_splines-accel_splines)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/mcycle_splines.json.zip
    Archive SHA-256: bc53c050edd46b86d4a38e460427dbd7efb62d95eeaf08901c6ce2fc4b1d38c7
    Local archive: data/posteriordb_mcycle_splines.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/mcycle_splines-accel_splines.json
References: bales2019selecting (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/bbbales2/cmdstan-warmup/blob/develop/examples/accel_splines/

Implementation notes:
Acceleration spline model from posteriordb and Bales (2019).

The supplied spline matrices define the fixed model basis.
The unpenalized linear coefficients retain their improper flat priors.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -161646.16470841187; PyMC = -161641.83664461042.
- B: Stan = -129349.91842334527; PyMC = -129345.6220514148.
- Raw delta: Stan = 32296.246285066605; PyMC = 32296.21459319562.
- Raw comparison: MISMATCH; residual = 0.03169187098683324.
- Additive Stan adjustment changed_integer_scale_half_student_t_prior: A =
  4.328063761528535; B = 4.296371890539598.
- Adjusted Stan delta = 32296.214593195615.
- Adjusted comparison: MATCH; residual = -3.637978807091713e-12.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 0.0003239621459319562
  using adjusted delta_Stan.
- In the measured installed PyMC/PyTensor, integer sigma=36 in both HalfStudentT spline-
  scale priors has dtype int8: sigma**2 overflows to 16. Their native density is
  therefore HalfStudentT(3,4), not Stan's normalized positive StudentT(3,0,36). The
  named adjustment adds the two source-derived prior log-density differences to Stan.
  Original targets differ; this runtime discrepancy is not a best-practice
  reparameterization. Recheck this adjustment after engine upgrades.
- Original Stan SHA-256: 9996e181a5d23d8c2a127adfa19c4c1e12051abf91539ce40a14249fc94d8bae
- Saved points: models/posteriordb_logp/points/mcycle_splines-accel_splines.npz
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
            Path(__file__).parent / "data" / "posteriordb_mcycle_splines.json.zip"
        ) as archive,
        archive.open("mcycle_splines.json") as stream,
    ):
        data = json.load(stream)
    coords = {
        "observation": np.arange(data["N"]),
        "mean_linear": np.arange(data["Ks"]),
        "scale_linear": np.arange(data["Ks_sigma"]),
        "mean_spline": np.arange(data["knots_1"]),
        "scale_spline": np.arange(data["knots_sigma_1"]),
    }
    with pm.Model(coords=coords) as model:
        mean_linear = pm.Data(
            "Xs",
            np.asarray(data["Xs"], dtype=float),
            dims=("observation", "mean_linear"),
        )
        scale_linear = pm.Data(
            "Xs_sigma",
            np.asarray(data["Xs_sigma"], dtype=float),
            dims=("observation", "scale_linear"),
        )
        mean_spline = pm.Data(
            "Zs_1_1",
            np.asarray(data["Zs_1_1"], dtype=float),
            dims=("observation", "mean_spline"),
        )
        scale_spline = pm.Data(
            "Zs_sigma_1_1",
            np.asarray(data["Zs_sigma_1_1"], dtype=float),
            dims=("observation", "scale_spline"),
        )
        intercept = pm.StudentT("Intercept", nu=3, mu=-13, sigma=36, initval=-13)
        intercept_sigma = pm.StudentT(
            "Intercept_sigma", nu=3, mu=0, sigma=10, initval=0
        )
        bs = pm.Flat("bs", dims="mean_linear")
        bs_sigma = pm.Flat("bs_sigma", dims="scale_linear")
        z = pm.Normal("zs_1_1", 0, 1, dims="mean_spline")
        z_sigma = pm.Normal("zs_sigma_1_1", 0, 1, dims="scale_spline")
        # Runtime discrepancy in the measured PyMC/PyTensor: integer sigma=36
        # is int8, so HalfStudentT's sigma**2 becomes 16. These native priors
        # therefore have scale 4 rather than Stan's 36. The comparison
        # explicitly accounts for that changed prior; this builder is left
        # unchanged so the measured discrepancy remains reproducible.
        sd = pm.HalfStudentT("sds_1_1", nu=3, sigma=36, initval=36)
        sd_sigma = pm.HalfStudentT("sds_sigma_1_1", nu=3, sigma=36, initval=1)
        coefficients = pm.Deterministic("s_1_1", sd * z, dims="mean_spline")
        coefficients_sigma = pm.Deterministic(
            "s_sigma_1_1", sd_sigma * z_sigma, dims="scale_spline"
        )
        mu = pm.Deterministic(
            "mu",
            intercept + mean_linear @ bs + mean_spline @ coefficients,
            dims="observation",
        )
        sigma = pm.Deterministic(
            "sigma",
            pt.exp(
                intercept_sigma
                + scale_linear @ bs_sigma
                + scale_spline @ coefficients_sigma
            ),
            dims="observation",
        )
        if not data["prior_only"]:
            pm.Normal(
                "Y",
                mu=mu,
                sigma=sigma,
                observed=np.asarray(data["Y"]),
                dims="observation",
            )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
