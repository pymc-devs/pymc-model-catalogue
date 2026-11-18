"""
Model: Multiple Predictors Linear Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/kidscore_momhsiq.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/kidscore_momhsiq.py
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/kidscore_momhsiq.py
Upstream catalogue contributor: Oliver Järnefelt
Description: kid_score ~ mom_hs + mom_iq

Data: kidiq (upstream posterior: kidiq-kidscore_momhsiq)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/kidiq.json.zip
    Archive SHA-256: 16d341a99ec8178231eab660d8fda084a44145f6a241daf35276d0192d13b92d
    Local archive: data/posteriordb_kidiq.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/kidiq-kidscore_momhsiq.json
References: gelman2006data (see posteriordb/source/bibliography/references.bib)
Reference URL: https://raw.githubusercontent.com/stan-dev/example-models/master/ARM/Ch.3/kidiq_multi_preds.stan

Implementation notes:
Native child-score regression from Gelman and Hill (2006), Chapter 3.

The coefficient density is constant. The positive noise scale follows a half-Cauchy law.

Stan comparison:
- A: Stan = -1001555.0012968204; PyMC = -1001554.3081496654.
- B: Stan = -4268980.8297550175; PyMC = -4268980.136607862.
- Raw delta: Stan = -3267425.8284581974; PyMC = -3267425.8284581965.
- Raw comparison: MATCH; residual = -9.313225746154785e-10.
- Tolerance: 0.032675258284581975
- Stan sigma>0 with Cauchy(0,2.5) versus PyMC HalfCauchy(2.5): PyMC-Stan is the constant
  log(2).
- Original Stan SHA-256: de5689ba783aca3612123239b62e8e7da2b5a61785ac001bb4f7a6a601c99467

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
    iq = np.asarray(data["mom_iq"], dtype=float)
    design = np.column_stack((np.ones(score.size), hs, iq))
    coords = {
        "child": np.arange(score.size),
        "coefficient": ["intercept", "mom_hs", "mom_iq"],
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
