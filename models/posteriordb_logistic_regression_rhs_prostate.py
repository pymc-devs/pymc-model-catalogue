"""
Model: Logistic Regression with Regularized Horseshoe Prior
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/logistic_regression_rhs.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/logistic_regression_rhs.py
Authors: Piironen, Juho and Vehtari, Aki and others (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/logistic_regression_rhs.py
Upstream catalogue contributor: Oliver Järnefelt
Description: y ~ Ber(x)

Data: prostate (upstream posterior: prostate-logistic_regression_rhs)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/prostate.json.zip
    Archive SHA-256: 52c35477f06a84cf0d8475e4a4aae09b448ec9a3fc1739b3f7bf0969c37cd8a4
    Local archive: data/posteriordb_prostate.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/prostate-logistic_regression_rhs.json
References: piironen2017sparsity (see posteriordb/source/bibliography/references.bib)
Reference URL:

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Regularized horseshoe logistic regression from Piironen and Vehtari (2017).

The global half-Student-t scale equals twice the dataset scale_global value.

Stan comparison:
- Not recorded.

Benchmark results:
- Not measured.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_prostate.json.zip"
        ) as archive,
        archive.open("prostate.json") as stream,
    ):
        data = json.load(stream)
    coords = {"observation": range(data["n"]), "predictor": range(data["d"])}
    global_scale = 2 * data["scale_global"]
    slab_shape = 0.5 * data["slab_df"]
    with pm.Model(coords=coords) as model:
        x = pm.Data(
            "x",
            np.asarray(data["x"], dtype="float64"),
            dims=("observation", "predictor"),
        )
        beta0 = pm.Normal("beta0", mu=0, sigma=data["scale_icept"], initval=0)
        z = pm.Normal("z", mu=0, sigma=1, dims="predictor", initval=np.zeros(data["d"]))
        tau = pm.HalfStudentT(
            "tau", nu=data["nu_global"], sigma=global_scale, initval=global_scale
        )
        local_scale = pm.HalfStudentT(
            "lambda",
            nu=data["nu_local"],
            sigma=1,
            dims="predictor",
            initval=np.ones(data["d"]),
        )
        caux = pm.InverseGamma(
            "caux",
            alpha=slab_shape,
            beta=slab_shape,
            initval=slab_shape / (slab_shape + 1),
        )
        slab_variance = data["slab_scale"] ** 2 * caux
        regularized_local_scale = pt.sqrt(
            slab_variance * local_scale**2 / (slab_variance + tau**2 * local_scale**2)
        )
        beta = pm.Deterministic(
            "beta", z * tau * regularized_local_scale, dims="predictor"
        )
        pm.Bernoulli(
            "y",
            logit_p=beta0 + pt.dot(x, beta),
            observed=np.asarray(data["y"], dtype="int64"),
            dims="observation",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
