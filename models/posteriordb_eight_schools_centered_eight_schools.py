"""
Model: A centered hiearchical model for 8 schools
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/eight_schools_centered.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/eight_schools_centered.py
Authors: Rubin, Donald B; Gelman, Andrew and Carlin, John B and Stern, Hal S and Dunson, David B and Vehtari, Aki and Rubin, Donald B (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/eight_schools_centered.py
Upstream catalogue contributor: Mans Magnusson
Description: A centered hiearchical model for the 8 schools example of Rubin (1981)

Data: eight_schools (upstream posterior: eight_schools-eight_schools_centered)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/eight_schools.json.zip
    Archive SHA-256: 8e37cc01fe05b0e1779255e9bd386261f3d61daf7f75f02f72f38cdc9750531c
    Local archive: data/posteriordb_eight_schools.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/eight_schools-eight_schools_centered.json
References: rubin1981estimation, gelman2013bayesian (see posteriordb/source/bibliography/references.bib)
Reference URL: http://www.stat.columbia.edu/~gelman/arm/examples/schools

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native centered school effects from posteriordb eight_schools_centered.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -46.57206565751112; PyMC = -45.87891853518809.
- B: Stan = -43.81356876639853; PyMC = -43.12042164407549.
- Raw delta: Stan = 2.7584968911125856; PyMC = 2.7584968911125998.
- Raw comparison: MATCH; residual = -1.4210854715202004e-14.
- No parameter-dependent adjustment; only constant offsets may cancel.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 1.027584968911126e-06
  using adjusted delta_Stan when an adjustment is present.
- Identity on all constrained free parameters, including original array order. No
  coordinate-change Jacobian or parameter-dependent factor correction. Stan support-
  transform and PyMC value-transform Jacobians are disabled.
- mu, tau and the native school coordinates are unchanged. Both use centered
  theta~Normal(mu,tau). Stan tau>0 with Cauchy(0,5) versus PyMC HalfCauchy(5) differs
  only by the constant log(2).
- Original Stan SHA-256: ccaf2d7beed22602f62f276cd4096194ab5de5b5c1d6bb2bae9a3655a3845c05
- Saved points: models/posteriordb_logp/points/eight_schools-eight_schools_centered.npz
- Full evidence: models/posteriordb_logp/records/eight_schools-eight_schools_centered.json
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
            Path(__file__).parent / "data" / "posteriordb_eight_schools.json.zip"
        ) as archive,
        archive.open("eight_schools.json") as stream,
    ):
        data = json.load(stream)
    with pm.Model(coords={"school": np.arange(1, data["J"] + 1)}) as model:
        mu = pm.Normal("mu", 0, 5)
        tau = pm.HalfCauchy("tau", 5)
        theta = pm.Normal("theta", mu, tau, dims="school")
        pm.Normal(
            "y",
            theta,
            sigma=np.asarray(data["sigma"], dtype=float),
            observed=data["y"],
            dims="school",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
