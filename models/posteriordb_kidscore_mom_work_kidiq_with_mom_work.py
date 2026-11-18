"""
Model: Multiple Factor Level Predictors Linear Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/kidscore_mom_work.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/kidscore_mom_work.py
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/kidscore_mom_work.py
Upstream catalogue contributor: Oliver Järnefelt
Description: kid_score ~ as.factor(mom_work)

Data: kidiq_with_mom_work (upstream posterior: kidiq_with_mom_work-kidscore_mom_work)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/kidiq_with_mom_work.json.zip
    Archive SHA-256: 0c82d40ace4c75bba4bc8824d4ece044e1d5f8700d479ea672001ab475254913
    Local archive: data/posteriordb_kidiq_with_mom_work.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/kidiq_with_mom_work-kidscore_mom_work.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://raw.githubusercontent.com/stan-dev/example-models/master/ARM/Ch.4/kidscore_momwork.stan

Implementation notes:
Native child-score work-category regression from Gelman and Hill (2006), Chapter 4.

Work category one is the reference. The coefficients and positive noise scale have constant densities.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -3489148.4822633765; PyMC = -3489148.4822633765.
- B: Stan = -1357896.0004124728; PyMC = -1357896.0004124732.
- Raw delta: Stan = 2131252.4818509035; PyMC = 2131252.4818509035.
- Raw comparison: MATCH; residual = 0.0.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 0.021313524818509035
- Original Stan SHA-256: fe41ba28792cddf91df4f46e5b48c2bcf33c0f1a193148fc36b4a52483640d27
- Saved points: models/posteriordb_logp/points/kidiq_with_mom_work-kidscore_mom_work.npz
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
    work = np.asarray(data["mom_work"])
    design = np.column_stack((np.ones(score.size), work == 2, work == 3, work == 4))
    coords = {
        "child": np.arange(score.size),
        "coefficient": ["intercept", "work2", "work3", "work4"],
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
