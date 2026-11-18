"""
Model: Normal Hierarchical Model to Model Rats' Weight Over Time
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/rats_model.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/rats_model.py
Authors: Alan E. Gelfand and Susan E. Hills and Amy Racine-Poon and Adrian F. M. Smith (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/rats_model.py
Upstream catalogue contributor: Kane Lindsay
Description: The model is essentially a random effects linear growth curve.

Data: rats_data (upstream posterior: rats_data-rats_model)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/rats_data.json.zip
    Archive SHA-256: 10ceca35c2f65e56b442a4adc7c4e8e26be0edb7ad11e76cd3ddc28a22cb3789
    Local archive: data/posteriordb_rats_data.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/rats_data-rats_model.json
References: gelfand1990illustration (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/example-models/tree/master/bugs_examples/vol1/rats

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native centered growth effects from posteriordb rats_model.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -5468930.328995618; PyMC = -5468930.328995745.
- B: Stan = -4398362.244870762; PyMC = -4398362.244870891.
- Raw delta: Stan = 1070568.0841248557; PyMC = 1070568.0841248538.
- Raw comparison: MATCH; residual = 1.862645149230957e-09.
- No parameter-dependent adjustment; only constant offsets may cancel.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 0.010706680841248557
  using adjusted delta_Stan when an adjustment is present.
- Keep centered alpha and beta in rat order. PyMC rat_index=rat-1 and centered_x=x-xbar
  reproduce the Stan observation predictor. Positive sigma_y, sigma_alpha and sigma_beta
  have the same improper flat kernels as HalfFlat, not proper half-Normal priors.
  Independent Normal alpha/beta hierarchy and hypermeans are unchanged.
- No nonconstant density adjustment is required by these source definitions; parameter-
  independent constants are omitted from two-point deltas.
- Original Stan SHA-256: c01395015d523c02784e538927b7c2b9cf36114d93729bd25a81d566e05eca19
- Saved points: models/posteriordb_logp/points/rats_data-rats_model.npz
- Full evidence: models/posteriordb_logp/records/rats_data-rats_model.json
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
            Path(__file__).parent / "data" / "posteriordb_rats_data.json.zip"
        ) as archive,
        archive.open("rats_data.json") as stream,
    ):
        data = json.load(stream)
    coords = {
        "rat_id": np.arange(1, data["N"] + 1),
        "observation": np.arange(data["Npts"]),
    }
    with pm.Model(coords=coords) as model:
        rat = pm.Data(
            "rat_index", np.asarray(data["rat"], dtype="int64") - 1, dims="observation"
        )
        centered_x = pm.Data(
            "centered_x",
            np.asarray(data["x"], dtype=float) - data["xbar"],
            dims="observation",
        )
        mu_alpha = pm.Normal("mu_alpha", 0, 100, initval=200.0)
        mu_beta = pm.Normal("mu_beta", 0, 100, initval=6.0)
        # These improper scale densities stay flat on positive scales.
        sigma_y = pm.HalfFlat("sigma_y", initval=10.0)
        sigma_alpha = pm.HalfFlat("sigma_alpha", initval=20.0)
        sigma_beta = pm.HalfFlat("sigma_beta", initval=1.0)
        alpha = pm.Normal(
            "alpha",
            mu_alpha,
            sigma_alpha,
            dims="rat_id",
            initval=np.full(data["N"], 200.0),
        )
        beta = pm.Normal(
            "beta", mu_beta, sigma_beta, dims="rat_id", initval=np.full(data["N"], 6.0)
        )
        pm.Normal(
            "y",
            mu=alpha[rat] + beta[rat] * centered_x,
            sigma=sigma_y,
            observed=data["y"],
            dims="observation",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
