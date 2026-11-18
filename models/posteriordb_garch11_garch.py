"""
Model: Generalized Autoregressive Conditional Heteroscedastic Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/garch11.stan
Authors: Not specified in upstream model metadata.
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/garch11.py
Upstream catalogue contributor: Mans Magnusson
Description: A GARCH(1, 1) time series model.

Data: garch (upstream posterior: garch-garch11)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/garch.json.zip
    Archive SHA-256: 056100504a96a72ee6be6af80228d26189dea7343a30df6ecfefac70cc9b3957
    Local archive: data/posteriordb_garch.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/garch-garch11.json
Reference URL: https://github.com/stan-dev/stat_comp_benchmarks/tree/master/benchmarks/garch

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
GARCH(1, 1) from the posteriordb garch11 probability model.

The coefficient pair has a flat joint density on its triangular support.
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -752.9671745691298; PyMC = -752.96717456913.
- B: Stan = -761.0105067059353; PyMC = -761.0105067059361.
- Raw delta: Stan = -8.043332136805475; PyMC = -8.043332136806157.
- Raw comparison: MATCH; residual = 6.821210263296962e-13.
- No parameter-dependent adjustment; only constant offsets may cancel.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 1.0804333213680616e-06
  using adjusted delta_Stan when an adjustment is present.
- Identity on the triangle alpha1>0, beta1>0, alpha1+beta1<1. The PyMC log1p(-alpha1)
  potential cancels its conditional Uniform beta1 normalizer; the joint constrained
  coefficient density is flat as in Stan.
- Both recursions initialize variance=sigma1^2, then add alpha0 + alpha1*(previous_y-
  mu)^2 + beta1*previous_variance. No scale squaring or observation reordering is
  introduced by this map.
- Original Stan SHA-256: bbfe1929a9dbf6f5a4066f6099d90fc8c3e1bf1c65e1072b47af19534c60a679
- Saved points: models/posteriordb_logp/points/garch-garch11.npz
- Full evidence: models/posteriordb_logp/records/garch-garch11.json
- Two-point evidence only; no global equivalence, support or inference claim.

Benchmark results:
- Not measured.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt


def build_model():
    def _garch11_variance_step(
        previous_y, previous_variance, mu, alpha0, alpha1, beta1
    ):
        return alpha0 + alpha1 * pt.square(previous_y - mu) + beta1 * previous_variance

    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_garch.json.zip"
        ) as archive,
        archive.open("garch.json") as stream,
    ):
        data = json.load(stream)
    y = np.asarray(data["y"], dtype=float)
    with pm.Model(coords={"time": np.arange(int(data["T"]))}) as model:
        mu = pm.Flat("mu", initval=float(y.mean()))
        alpha0 = pm.HalfFlat("alpha0", initval=1)
        alpha1 = pm.Uniform("alpha1", lower=0, upper=1, initval=0.2)
        beta1 = pm.Uniform("beta1", lower=0, upper=1 - alpha1, initval=0.5)
        # Remove the conditional uniform normalizer to retain the flat joint density.
        pm.Potential("coefficient_density", pt.log1p(-alpha1))
        observations = pm.Data("observations", y, dims="time")
        first_variance = pt.as_tensor_variable(np.float64(data["sigma1"]) ** 2)
        if len(y) > 1:
            later_variance, _ = pytensor.scan(
                fn=_garch11_variance_step,
                sequences=observations[:-1],
                outputs_info=first_variance,
                non_sequences=[mu, alpha0, alpha1, beta1],
                strict=True,
                name="garch11_variance",
            )
            variance = pt.concatenate([first_variance[None], later_variance])
        else:
            variance = first_variance[None]
        pm.Normal("y", mu=mu, sigma=pt.sqrt(variance), observed=y, dims="time")
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
