"""
Model: 3-Predictor logistic regression model for decision to switch wells.
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/wells_dae_model.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/wells_dae_model.py
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/wells_dae_model.py
Upstream catalogue contributor: Kane Lindsay
Description: Performs logistic regression using three predictors (distance from nearest safe well,
    arsenic levels, and years in education).

Data: wells_data (upstream posterior: wells_data-wells_dae_model)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/wells_data.json.zip
    Archive SHA-256: 0ecd7efe3e64c63d849c562e61658fe322509942a882bc515176e016cc57c969
    Local archive: data/posteriordb_wells_data.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/wells_data-wells_dae_model.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/example-models/tree/master/ARM/Ch.5

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native well-switching logistic regression from Gelman and Hill (2006), Chapter 5.

All coefficient densities are constant. Distance uses hundreds of meters.
Education uses four-year units without centering.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -2216.442683200764; PyMC = -2216.442683200764.
- B: Stan = -2055.0655135993466; PyMC = -2055.0655135993466.
- Raw delta: Stan = 161.37716960141734; PyMC = 161.37716960141734.
- Raw comparison: MATCH; residual = 0.0.
- No parameter-dependent adjustment; only constant offsets may cancel.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 2.6137716960141735e-06
  using adjusted delta_Stan when an adjustment is present.
- Identity on all constrained free parameters, including original array order. No
  coordinate-change Jacobian or parameter-dependent factor correction. Stan support-
  transform and PyMC value-transform Jacobians are disabled.
- Both use predictors [dist/100, arsenic, educ/4]; cd=(dist-mean(dist))/100, ca=arsenic-
  mean(arsenic), ce=(educ-mean(educ))/4 where present. Original observation and
  coefficient order are retained; all coefficients are flat. Except wells_dist, alpha is
  the separate intercept.
- Original Stan SHA-256: 5c0dc0d1ea6c201de429ad018a7abeeb13593cecb1be0dbf59c4a388a2c5f5e0
- Saved points: models/posteriordb_logp/points/wells_data-wells_dae_model.npz
- Full evidence: models/posteriordb_logp/records/wells_data-wells_dae_model.json
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
    dist = np.asarray(data["dist"], dtype=float) / 100.0
    arsenic = np.asarray(data["arsenic"], dtype=float)
    educ = np.asarray(data["educ"], dtype=float) / 4.0
    design = np.column_stack((dist, arsenic, educ))
    coords = {
        "household": np.arange(switched.size),
        "coefficient": ["dist100", "arsenic", "educ4"],
    }
    with pm.Model(coords=coords) as model:
        predictors = pm.Data("predictors", design, dims=("household", "coefficient"))
        alpha = pm.Flat("alpha")
        beta = pm.Flat("beta", dims="coefficient")
        pm.Bernoulli(
            "switched",
            logit_p=alpha + pm.math.dot(predictors, beta),
            observed=switched,
            dims="household",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
