"""
Model: A Two-Dimensional (unordered) Gaussian Mixture Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/low_dim_gauss_mix_collapse.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/low_dim_gauss_mix_collapse.py
Authors: Not specified in upstream model metadata.
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/low_dim_gauss_mix_collapse.py
Upstream catalogue contributor: Mans Magnusson
Description: A Two-Dimensional Gaussian Mixture Model

Data: low_dim_gauss_mix_collapse (upstream posterior: low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/low_dim_gauss_mix_collapse.json.zip
    Archive SHA-256: 13bed61589755ffd3824635189191a66d55647dad61942ca0441d8934506c860
    Local archive: data/posteriordb_low_dim_gauss_mix_collapse.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse.json
Reference URL: https://github.com/stan-dev/stat_comp_benchmarks/tree/master/benchmarks/low_dim_gauss_mix_collapse

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Two-component Gaussian mixture with unrestricted means.

Source: posteriordb low_dim_gauss_mix_collapse model metadata at revision 91027f9.
This named variant keeps both label permutations and does not order the means.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -1789.561133899328; PyMC = -1788.1748396462788.
- B: Stan = -1687.6652702311083; PyMC = -1686.2789759780596.
- Raw delta: Stan = 101.89586366821982; PyMC = 101.89586366821914.
- Raw comparison: MATCH; residual = 6.821210263296962e-13.
- No parameter-dependent adjustment; only constant offsets may cancel.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 2.018958636682198e-06
  using adjusted delta_Stan when an adjustment is present.
- Identity of unrestricted two-component mu, positive sigma and scalar theta; no
  relabeling or sorting. Both likelihoods marginalize with weights [theta,1-theta] per
  observation. HalfNormal scale normalizers differ from positive Normal densities only
  by two log(2) constants.
- Original Stan SHA-256: 5aef8bff324b2c25497ef2920153ca6359d1cb846468552b1c9a40e9970b0d52
- Saved points: models/posteriordb_logp/points/low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse.npz
- Full evidence: models/posteriordb_logp/records/low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse.json
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
            Path(__file__).parent
            / "data"
            / "posteriordb_low_dim_gauss_mix_collapse.json.zip"
        ) as archive,
        archive.open("low_dim_gauss_mix_collapse.json") as stream,
    ):
        data = json.load(stream)
    y = np.asarray(data["y"], dtype=float)
    coords = {"component": np.arange(2), "observation": np.arange(data["N"])}
    with pm.Model(coords=coords) as model:
        mu = pm.Normal(
            "mu", mu=0, sigma=2, dims="component", initval=np.array([-1.0, 1.0])
        )
        sigma = pm.HalfNormal(
            "sigma", sigma=2, dims="component", initval=np.full(2, 2.0)
        )
        theta = pm.Beta("theta", alpha=5, beta=5, initval=0.5)
        weights = pt.stack([theta, 1 - theta])
        pm.NormalMixture(
            "y", w=weights, mu=mu, sigma=sigma, observed=y, dims="observation"
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
