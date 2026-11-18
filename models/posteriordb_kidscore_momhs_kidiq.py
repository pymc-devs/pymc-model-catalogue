"""
Model: One Predictor Linear Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/kidscore_momhs.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/kidscore_momhs.py
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/kidscore_momhs.py
Upstream catalogue contributor: Mans Magnusson
Description: kid_score ~ mom_hs

Data: kidiq (upstream posterior: kidiq-kidscore_momhs)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/kidiq.json.zip
    Archive SHA-256: 16d341a99ec8178231eab660d8fda084a44145f6a241daf35276d0192d13b92d
    Local archive: data/posteriordb_kidiq.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/kidiq-kidscore_momhs.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://raw.githubusercontent.com/stan-dev/example-models/master/ARM/Ch.3/kidscore_momhs.stan

Implementation notes:
Native child-score regression from Gelman and Hill (2006), Chapter 3.

The coefficient density is constant. The positive noise scale follows a half-Cauchy law.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -2089221.6174608155; PyMC = -2089220.9243136612.
- B: Stan = -1326230.7365812259; PyMC = -1326230.0434340718.
- Raw delta: Stan = 762990.8808795896; PyMC = 762990.8808795894.
- Raw comparison: MATCH; residual = 2.3283064365386963e-10.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 0.007630908808795896
- Stan sigma>0 with Cauchy(0,2.5) versus PyMC HalfCauchy(2.5): PyMC-Stan is the constant
  log(2).
- Original Stan SHA-256: 555e35e50b5628fb981b5f0dafcf899feee4be0e4fcbbc5255f3d7df4128f15b
- Saved points: models/posteriordb_logp/points/kidiq-kidscore_momhs.npz
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
            Path(__file__).parent / "data" / "posteriordb_kidiq.json.zip"
        ) as archive,
        archive.open("kidiq.json") as stream,
    ):
        data = json.load(stream)
    score = np.asarray(data["kid_score"], dtype=float)
    hs = np.asarray(data["mom_hs"], dtype=float)
    design = np.column_stack((np.ones(score.size), hs))
    coords = {
        "child": np.arange(score.size),
        "coefficient": ["intercept", "mom_hs"],
    }
    with pm.Model(coords=coords) as model:
        predictors = pm.Data("predictors", design, dims=("child", "coefficient"))
        beta = pm.Flat("beta", dims="coefficient")
        sigma = pm.HalfCauchy("sigma", beta=2.5, initval=20.0)
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
