"""
Model: Multiple Interacting Predictors Centered Linear Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/kidscore_interaction_c.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/kidscore_interaction_c.py
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/kidscore_interaction_c.py
Upstream catalogue contributor: Oliver Järnefelt
Description: kid_score ~ c_mom_hs + c_mom_iq + c_mom_hs:c_mom_iq

Data: kidiq_with_mom_work (upstream posterior: kidiq_with_mom_work-kidscore_interaction_c)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/kidiq_with_mom_work.json.zip
    Archive SHA-256: 0c82d40ace4c75bba4bc8824d4ece044e1d5f8700d479ea672001ab475254913
    Local archive: data/posteriordb_kidiq_with_mom_work.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/kidiq_with_mom_work-kidscore_interaction_c.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://raw.githubusercontent.com/stan-dev/example-models/master/ARM/Ch.4/kidiq_interaction_c.stan

Implementation notes:
Native centered child-score interaction regression from Gelman and Hill (2006), Chapter 4.

Both predictors use their sample means. The coefficients and positive noise scale have constant densities.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -1329245.3365258172; PyMC = -1329245.3365258162.
- B: Stan = -1916575.1875751258; PyMC = -1916575.187575127.
- Raw delta: Stan = -587329.8510493087; PyMC = -587329.8510493108.
- Raw comparison: MATCH; residual = 2.0954757928848267e-09.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 0.005874298510493108
- Original Stan SHA-256: ab4821a0785f15ebcdea112d6436db3c2433f26aca8d10798f2824f752406727
- Saved points: models/posteriordb_logp/points/kidiq_with_mom_work-kidscore_interaction_c.npz
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
    hs = np.asarray(data["mom_hs"], dtype=float)
    iq = np.asarray(data["mom_iq"], dtype=float)
    hs = hs - hs.mean()
    iq = iq - iq.mean()
    design = np.column_stack((np.ones(score.size), hs, iq, hs * iq))
    coords = {
        "child": np.arange(score.size),
        "coefficient": ["intercept", "centered_mom_hs", "centered_mom_iq", "hs_iq"],
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
