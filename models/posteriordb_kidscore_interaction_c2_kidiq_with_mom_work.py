"""
Model: Multiple Interacting Predictors Conventionally Centered Linear Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/kidscore_interaction_c2.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/kidscore_interaction_c2.py
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/kidscore_interaction_c2.py
Upstream catalogue contributor: Oliver Järnefelt
Description: kid_score ~ c2_mom_hs + c2_mom_iq + c2_mom_hs:c2_mom_iq, c2_mom_hs <- mom_hs - 0.5,
    c2_mom_iq <- mom_iq - 100

Data: kidiq_with_mom_work (upstream posterior: kidiq_with_mom_work-kidscore_interaction_c2)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/kidiq_with_mom_work.json.zip
    Archive SHA-256: 0c82d40ace4c75bba4bc8824d4ece044e1d5f8700d479ea672001ab475254913
    Local archive: data/posteriordb_kidiq_with_mom_work.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/kidiq_with_mom_work-kidscore_interaction_c2.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://raw.githubusercontent.com/stan-dev/example-models/master/ARM/Ch.4/kidiq_interaction_c2.stan

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native reference-centered child-score interaction regression from Gelman and Hill (2006), Chapter 4.

Education centers at 0.5 and IQ centers at 100. The coefficients and positive noise scale have constant densities.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -866198.4706636631; PyMC = -866198.4706636635.
- B: Stan = -1876522.7716739054; PyMC = -1876522.7716739047.
- Raw delta: Stan = -1010324.3010102423; PyMC = -1010324.3010102413.
- Raw comparison: MATCH; residual = -1.0477378964424133e-09.
- No parameter-dependent adjustment; only constant offsets may cancel.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 0.010104243010102423
  using adjusted delta_Stan when an adjustment is present.
- Identity on all constrained free parameters, including original array order. No
  coordinate-change Jacobian or parameter-dependent factor correction. Stan support-
  transform and PyMC value-transform Jacobians are disabled.
- Both predictors use [1, hs-0.5, iq-100, (hs-0.5)*(iq-100)]; beta order is unchanged.
- Stan implicit flat beta/sigma>0 match PyMC Flat/HalfFlat.
- Original Stan SHA-256: f0e2de7b978b8ee263d3aa5fccdc4170e2f4d0678af5d6b8a0090162d084e9be
- Saved points: models/posteriordb_logp/points/kidiq_with_mom_work-kidscore_interaction_c2.npz
- Full evidence: models/posteriordb_logp/records/kidiq_with_mom_work-kidscore_interaction_c2.json
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
            Path(__file__).parent / "data" / "posteriordb_kidiq_with_mom_work.json.zip"
        ) as archive,
        archive.open("kidiq_with_mom_work.json") as stream,
    ):
        data = json.load(stream)
    score = np.asarray(data["kid_score"], dtype=float)
    hs = np.asarray(data["mom_hs"], dtype=float) - 0.5
    iq = np.asarray(data["mom_iq"], dtype=float) - 100.0
    design = np.column_stack((np.ones(score.size), hs, iq, hs * iq))
    coords = {
        "child": np.arange(score.size),
        "coefficient": ["intercept", "reference_mom_hs", "reference_mom_iq", "hs_iq"],
    }
    with pm.Model(coords=coords) as model:
        predictors = pm.Data("predictors", design, dims=("child", "coefficient"))
        beta = pm.Flat("beta", dims="coefficient")
        sigma = pm.HalfFlat("sigma", initval=20.0)
        pm.Normal(
            "kid_score",
            mu=pm.math.dot(predictors, beta),
            sigma=sigma,
            observed=score,
            dims="child",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
