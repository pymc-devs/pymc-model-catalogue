"""
Model: Structured Time Series model with stochastic level and stochastic seasonal
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/state_space_stochastic_level_stochastic_seasonal.stan
Authors: Not specified in upstream model metadata.
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/state_space_stochastic_level_stochastic_seasonal.py
Upstream catalogue contributor: Keane Nguyen
Description: Structured Time Series model with stochastic level and stochastic seasonal

Data: uk_drivers (upstream posterior: uk_drivers-state_space_stochastic_level_stochastic_seasonal)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/uk_drivers.json.zip
    Archive SHA-256: fc7459d061bf739debec146026d693c6780adba9baa2fec3f01bcf545ed6fc4b
    Local archive: data/posteriordb_uk_drivers.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/uk_drivers-state_space_stochastic_level_stochastic_seasonal.json
Reference URL: https://github.com/sinhrks/stan-statespace/blob/master/models/fig07_02.stan

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Seasonal state model from the posteriordb stochastic level model.

The initial eleven seasonal states and regression coefficients have flat densities.
The level bounds apply to every time point without transition normalization.
Source: sinhrks/stan-statespace, fig07_02, as specified by posteriordb.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -890.3549543316059; PyMC = -893.569279107265.
- B: Stan = -897.3775675171354; PyMC = -900.591892292794.
- Raw delta: Stan = -7.022613185529508; PyMC = -7.022613185529053.
- Raw comparison: MATCH; residual = -4.547473508864641e-13.
- No parameter-dependent adjustment; only constant offsets may cancel.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 1.070226131855295e-06
  using adjusted delta_Stan when an adjustment is present.
- Identity of bounded level mu, seasonal states, flat beta/lambda and three positive
  ordered SDs (seasonal,level,observation). Native AR rho=-ones(11) with flat initial
  eleven states equals Stan's seasonal[t]~Normal(-sum(previous eleven),sigma[0]); level
  transitions remain unnormalized Normal factors inside the fixed
  mean(y)+/-3*sample_sd(y) level bounds.
- Uniform level normalizers and three HalfStudentT versus positive Student-t scale
  normalizers are parameter-independent.
- Shared-support limitation: original Stan points enforce all ordered or positive-
  ordered constraints. PyMC ordering supplied only by a transform is not separately
  enforced after transforms are removed. These evaluations compare points on the shared
  Stan-valid support, not global support or model equivalence.
- Original Stan SHA-256: f0103754c910380b24d6a6619053eb04fb7ee9f74f4acfa301b6f6234e9b8dec
- Saved points: models/posteriordb_logp/points/uk_drivers-state_space_stochastic_level_stochastic_seasonal.npz
- Full evidence: models/posteriordb_logp/records/uk_drivers-state_space_stochastic_level_stochastic_seasonal.json
- Two-point evidence only; no global equivalence, support or inference claim.

Benchmark results:
- Not measured.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm
from pymc.distributions.transforms import Ordered


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_uk_drivers.json.zip"
        ) as archive,
        archive.open("uk_drivers.json") as stream,
    ):
        data = json.load(stream)
    y = np.asarray(data["y"], dtype=float)
    n_time = int(data["n"])
    center = float(y.mean())
    radius = 3 * float(y.std(ddof=1))
    coords = {"time": np.arange(n_time), "scale": ["seasonal", "level", "observation"]}
    with pm.Model(coords=coords) as model:
        # The three positive scales have an ordered half-Student-t density.
        sigma = pm.HalfStudentT(
            "sigma",
            nu=4,
            sigma=1,
            dims="scale",
            default_transform=None,
            transform=Ordered(positive=True),
            initval=np.array([0.05, 0.1, 0.2]),
        )
        beta = pm.Flat("beta", initval=0)
        regression_lambda = pm.Flat("lambda", initval=0)
        mu = pm.Uniform(
            "mu",
            lower=center - radius,
            upper=center + radius,
            dims="time",
            initval=np.full(n_time, center),
        )
        pm.Potential(
            "level_transition",
            pm.logp(pm.Normal.dist(mu=mu[:-1], sigma=sigma[1]), mu[1:]).sum(),
        )
        seasonal = pm.AR(
            "seasonal",
            rho=-np.ones(11),
            sigma=sigma[0],
            ar_order=11,
            init_dist=pm.Flat.dist(shape=11),
            dims="time",
            initval=np.zeros(n_time),
        )
        x = pm.Data("x", np.asarray(data["x"], dtype=float), dims="time")
        w = pm.Data("w", np.asarray(data["w"], dtype=float), dims="time")
        expectation = mu + beta * x + regression_lambda * w + seasonal
        pm.Normal("y", mu=expectation, sigma=sigma[2], observed=y, dims="time")
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
