"""
Model: Logistic regression model for decision to switch wells
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/wells_dist.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/wells_dist.py
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/wells_dist.py
Upstream catalogue contributor: Phil Clemson
Description: Performs logistic regression using one predictor (distance from nearest safe well).

Data: wells_data (upstream posterior: wells_data-wells_dist)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/wells_data.json.zip
    Archive SHA-256: 0ecd7efe3e64c63d849c562e61658fe322509942a882bc515176e016cc57c969
    Local archive: data/posteriordb_wells_data.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/wells_data-wells_dist.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/example-models/tree/master/ARM/Ch.5

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native well-switching logistic regression from Gelman and Hill (2006), Chapter 5.

All coefficient densities are constant. Distance uses meters.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -3351.6919122498107; PyMC = -3351.6919122498107.
- B: Stan = -8565.625928281908; PyMC = -8565.625928283498.
- Raw delta: Stan = -5213.934016032097; PyMC = -5213.934016033687.
- Raw comparison: MATCH; residual = 1.5897967386990786e-09.
- No parameter-dependent adjustment; only constant offsets may cancel.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 5.313934016033687e-05
  using adjusted delta_Stan when an adjustment is present.
- Identity on all constrained free parameters, including original array order. No
  coordinate-change Jacobian or parameter-dependent factor correction. Stan support-
  transform and PyMC value-transform Jacobians are disabled.
- Both use predictors [1, dist], with the intercept stored in beta[0]; cd=(dist-
  mean(dist))/100, ca=arsenic-mean(arsenic), ce=(educ-mean(educ))/4 where present.
  Original observation and coefficient order are retained; all coefficients are flat.
  Except wells_dist, alpha is the separate intercept.
- Original Stan SHA-256: 33d67128b28b9d4b57c4cd1881a0da40a627334d1fe2ea9b07736c451199afef
- Saved points: models/posteriordb_logp/points/wells_data-wells_dist.npz
- Full evidence: models/posteriordb_logp/records/wells_data-wells_dist.json
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
            Path(__file__).parent / "data" / "posteriordb_wells_data.json.zip"
        ) as archive,
        archive.open("wells_data.json") as stream,
    ):
        data = json.load(stream)
    switched = np.asarray(data["switched"], dtype=np.int64)
    dist = np.asarray(data["dist"], dtype=float)
    design = np.column_stack((np.ones(switched.size), dist))
    coords = {
        "household": np.arange(switched.size),
        "coefficient": ["intercept", "dist"],
    }
    with pm.Model(coords=coords) as model:
        predictors = pm.Data("predictors", design, dims=("household", "coefficient"))
        beta = pm.Flat("beta", dims="coefficient")
        pm.Bernoulli(
            "switched",
            logit_p=pm.math.dot(predictors, beta),
            observed=switched,
            dims="household",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
