"""Logistic Regression with Regularized Horseshoe Prior.

Stan comparison: MATCH.
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
