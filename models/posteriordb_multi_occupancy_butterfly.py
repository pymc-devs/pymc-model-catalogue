"""
Model: Multiple Species-Site Occupancy Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/multi_occupancy.stan
Authors: Dorazio, Robert M and Royle, J Andrew and Söderström, Bo and Glimsk{\"a}r, Anders (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/multi_occupancy.py
Upstream catalogue contributor: Oliver Järnefelt
Description: a hierarchical occupancy model for estimating species richness and accumulation by
    forming these community-level attributes as functions of model-based estimators of species
    occurence while accounting for imperfect detection of individual species.

Data: butterfly (upstream posterior: butterfly-multi_occupancy)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/butterfly.json.zip
    Archive SHA-256: ff2caec8dbabf91867035a5c186cbbba6671f0f7702717f64a0b6639b124e853
    Local archive: data/posteriordb_butterfly.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/butterfly-multi_occupancy.json
References: dorazio2006estimating (see posteriordb/source/bibliography/references.bib)
Reference URL: https://mc-stan.org/users/documentation/case-studies/dorazio-royle-occupancy.html

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native multiple-species occupancy model from Dorazio and Royle.

The likelihood marginalizes site occupancy and species availability exactly.
Species effects use the centered bivariate Gaussian law.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -2216.558948724705; PyMC = -2215.172654428967.
- B: Stan = -2567.8440496981566; PyMC = -2566.457755402418.
- Raw delta: Stan = -351.2851009734518; PyMC = -351.28510097345134.
- Raw comparison: MATCH; residual = -4.547473508864641e-13.
- No parameter-dependent adjustment; only constant offsets may cancel.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 4.512851009734518e-06
  using adjusted delta_Stan when an adjustment is present.
- Map correlation_unit=(rho_uv+1)/2 and uv=column_stack(uv1,uv2), retaining all S
  species, including S-n never-observed species. The PyMC Cholesky factor induces the
  same covariance with diagonals sigma_uv**2 and off-diagonal
  sigma_uv[0]*sigma_uv[1]*rho_uv. Stan already evaluates Beta(2,2) at (rho_uv+1)/2
  without a Jacobian; its value equals PyMC's Beta density at correlation_unit. The
  coordinate determinant |d rho_uv/d correlation_unit|=2 is constant (log(2));
  normalizing Stan's rho prior removes the same constant. The two positive Cauchy-to-
  HalfCauchy normalizations add only 2*log(2) to raw PyMC logp. Observed and never-
  observed inclusion/occupancy mixtures match, including the J-fold zero-site likelihood
  for each never-observed species; no latent species are discarded.
- No nonconstant density adjustment is required by these source definitions; parameter-
  independent constants are omitted from two-point deltas.
- Original Stan SHA-256: 2da8b97338d43107b8c9d97500ca3b9f526891ad6ace2eb11715c06c2c45b052
- Saved points: models/posteriordb_logp/points/butterfly-multi_occupancy.npz
- Full evidence: models/posteriordb_logp/records/butterfly-multi_occupancy.json
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
            Path(__file__).parent / "data" / "posteriordb_butterfly.json.zip"
        ) as archive,
        archive.open("butterfly.json") as stream,
    ):
        data = json.load(stream)
    detected = np.asarray(data["X"], dtype="int64")
    n_observed = data["n"]
    n_species = data["S"]
    coords = {
        "species": np.arange(1, n_species + 1),
        "observed_species": np.arange(1, n_observed + 1),
        "site": np.arange(1, data["J"] + 1),
        "species_parameter": ["occupancy", "detection"],
    }
    with pm.Model(coords=coords) as model:
        alpha = pm.Cauchy("alpha", alpha=0, beta=2.5)
        beta = pm.Cauchy("beta", alpha=0, beta=2.5)
        omega = pm.Beta("Omega", alpha=2, beta=2)
        sigma = pm.HalfCauchy(
            "sigma_uv", beta=2.5, dims="species_parameter", initval=np.ones(2)
        )
        correlation_unit = pm.Beta("correlation_unit", alpha=2, beta=2)
        rho = pm.Deterministic("rho_uv", 2 * correlation_unit - 1)
        chol = pt.stack(
            [
                pt.stack([sigma[0], pt.zeros_like(rho)]),
                pt.stack([sigma[1] * rho, sigma[1] * pt.sqrt(1 - rho**2)]),
            ]
        )
        uv = pm.MvNormal(
            "uv",
            mu=np.zeros(2),
            chol=chol,
            dims=("species", "species_parameter"),
            initval=np.zeros((n_species, 2)),
        )
        occupancy = pm.math.sigmoid(alpha + uv[:, 0])
        detection = pm.math.sigmoid(beta + uv[:, 1])
        pm.Bernoulli(
            "available",
            p=omega,
            observed=np.ones(n_observed, dtype="int64"),
            dims="observed_species",
        )
        # A zero-inflated Binomial expresses detection conditional on species availability.
        pm.ZeroInflatedBinomial(
            "X",
            psi=occupancy[:n_observed, None],
            n=data["K"],
            p=detection[:n_observed, None],
            observed=detected,
            dims=("observed_species", "site"),
        )
        zero_site_logp = pm.logp(
            pm.ZeroInflatedBinomial.dist(
                psi=occupancy[n_observed:],
                n=data["K"],
                p=detection[n_observed:],
            ),
            0,
        )
        pm.Potential(
            "never_observed",
            pt.logaddexp(
                pt.log1p(-omega),
                pt.log(omega) + data["J"] * zero_site_logp,
            ).sum(),
        )
        pm.Deterministic("E_N", n_species * omega)
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
