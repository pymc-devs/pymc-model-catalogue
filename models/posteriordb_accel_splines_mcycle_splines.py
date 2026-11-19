"""Spline model for acceleration mean and standard deviation.

Stan comparison: MISMATCH; MATCH after prior adjustment.
Integer overflow changed the HalfStudentT scale priors in the measured PyMC runtime.
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
            Path(__file__).parent / "data" / "posteriordb_mcycle_splines.json.zip"
        ) as archive,
        archive.open("mcycle_splines.json") as stream,
    ):
        data = json.load(stream)
    coords = {
        "observation": np.arange(data["N"]),
        "mean_linear": np.arange(data["Ks"]),
        "scale_linear": np.arange(data["Ks_sigma"]),
        "mean_spline": np.arange(data["knots_1"]),
        "scale_spline": np.arange(data["knots_sigma_1"]),
    }
    with pm.Model(coords=coords) as model:
        mean_linear = pm.Data(
            "Xs",
            np.asarray(data["Xs"], dtype=float),
            dims=("observation", "mean_linear"),
        )
        scale_linear = pm.Data(
            "Xs_sigma",
            np.asarray(data["Xs_sigma"], dtype=float),
            dims=("observation", "scale_linear"),
        )
        mean_spline = pm.Data(
            "Zs_1_1",
            np.asarray(data["Zs_1_1"], dtype=float),
            dims=("observation", "mean_spline"),
        )
        scale_spline = pm.Data(
            "Zs_sigma_1_1",
            np.asarray(data["Zs_sigma_1_1"], dtype=float),
            dims=("observation", "scale_spline"),
        )
        intercept = pm.StudentT("Intercept", nu=3, mu=-13, sigma=36, initval=-13)
        intercept_sigma = pm.StudentT(
            "Intercept_sigma", nu=3, mu=0, sigma=10, initval=0
        )
        bs = pm.Flat("bs", dims="mean_linear")
        bs_sigma = pm.Flat("bs_sigma", dims="scale_linear")
        z = pm.Normal("zs_1_1", 0, 1, dims="mean_spline")
        z_sigma = pm.Normal("zs_sigma_1_1", 0, 1, dims="scale_spline")
        # Runtime discrepancy in the measured PyMC/PyTensor: integer sigma=36
        # is int8, so HalfStudentT's sigma**2 becomes 16. These native priors
        # therefore have scale 4 rather than Stan's 36. The comparison
        # explicitly accounts for that changed prior; this builder is left
        # unchanged so the measured discrepancy remains reproducible.
        sd = pm.HalfStudentT("sds_1_1", nu=3, sigma=36, initval=36)
        sd_sigma = pm.HalfStudentT("sds_sigma_1_1", nu=3, sigma=36, initval=1)
        coefficients = pm.Deterministic("s_1_1", sd * z, dims="mean_spline")
        coefficients_sigma = pm.Deterministic(
            "s_sigma_1_1", sd_sigma * z_sigma, dims="scale_spline"
        )
        mu = pm.Deterministic(
            "mu",
            intercept + mean_linear @ bs + mean_spline @ coefficients,
            dims="observation",
        )
        sigma = pm.Deterministic(
            "sigma",
            pt.exp(
                intercept_sigma
                + scale_linear @ bs_sigma
                + scale_spline @ coefficients_sigma
            ),
            dims="observation",
        )
        if not data["prior_only"]:
            pm.Normal(
                "Y",
                mu=mu,
                sigma=sigma,
                observed=np.asarray(data["Y"]),
                dims="observation",
            )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
