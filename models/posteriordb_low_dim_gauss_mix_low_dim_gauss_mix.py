"""
Model: A Two-Dimensional Gaussian Mixture Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/low_dim_gauss_mix.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/low_dim_gauss_mix.py
Authors: Not specified in upstream model metadata.
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/low_dim_gauss_mix.py
Upstream catalogue contributor: Mans Magnusson
Description: A Two-Dimensional Gaussian Mixture Model

Data: low_dim_gauss_mix (upstream posterior: low_dim_gauss_mix-low_dim_gauss_mix)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/low_dim_gauss_mix.json.zip
    Archive SHA-256: 9b5ea284321417a7816b0426a0c96788a6824e6ceb4ec4ecd89493779720b8f9
    Local archive: data/posteriordb_low_dim_gauss_mix.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/low_dim_gauss_mix-low_dim_gauss_mix.json
Reference URL: https://github.com/stan-dev/stat_comp_benchmarks/tree/master/benchmarks/low_dim_gauss_mix

Implementation notes:
Two-component Gaussian mixture with ordered means.

Source: posteriordb low_dim_gauss_mix model metadata at revision 91027f9.
The ordered transform keeps the source constraint on the component means.
Each observation selects one Normal component through the native marginalized likelihood.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -5061.501489160019; PyMC = -5060.11519490697.
- B: Stan = -4432.7715974554385; PyMC = -4431.38530320239.
- Raw delta: Stan = 628.729891704581; PyMC = 628.72989170458.
- Raw comparison: MATCH; residual = 9.094947017729282e-13.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 7.287298917045809e-06
- HalfNormal scale normalizers differ from positive Normal densities only by two log(2)
  constants.
- Shared-support limitation: original Stan points enforce all ordered or positive-
  ordered constraints. PyMC ordering supplied only by a transform is not separately
  enforced after transforms are removed. These evaluations compare points on the shared
  Stan-valid support, not global support or model equivalence.
- Original Stan SHA-256: b94ee44ab73502a0053c4a52b7e9d582ca4f466994d7beef4a4e606ff8582344
- Saved points: models/posteriordb_logp/points/low_dim_gauss_mix-low_dim_gauss_mix.npz
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
            Path(__file__).parent / "data" / "posteriordb_low_dim_gauss_mix.json.zip"
        ) as archive,
        archive.open("low_dim_gauss_mix.json") as stream,
    ):
        data = json.load(stream)
    y = np.asarray(data["y"], dtype=float)
    coords = {
        "component": ["lower_mean", "upper_mean"],
        "observation": np.arange(data["N"]),
    }
    with pm.Model(coords=coords) as model:
        mu = pm.Normal(
            "mu",
            mu=0,
            sigma=2,
            dims="component",
            transform=pm.distributions.transforms.ordered,
            initval=np.array([-1.0, 1.0]),
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
