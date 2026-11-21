"""
Model: Gaussian Processes that model the mean and std of acceleration
Source: posteriordb/posterior_database/models/stan/accel_gp.stan
Authors: Ben Bales, Arya Pourzanjani, Aki Vehtari, Linda Petzold
Description: Finite-basis Gaussian processes model acceleration mean and log standard deviation.

Stan comparison:
- A: Stan = -155371.18965842773; PyMC = -155371.18965838786.
- B: Stan = -209414.58318585833; PyMC = -209414.58318581842.
- Raw delta: Stan = -54043.3935274306; PyMC = -54043.39352743057.
- Raw comparison: MATCH; residual = -2.9103830456733704e-11.
- Tolerance: 0.000541433935274306
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
            Path(__file__).parent / "data" / "posteriordb_mcycle_gp.json.zip"
        ) as archive,
        archive.open("mcycle_gp.json") as stream,
    ):
        data = json.load(stream)
    mean_basis = np.asarray(data["Xgp_1"], dtype=float)
    scale_basis = np.asarray(data["Xgp_sigma_1"], dtype=float)
    mean_frequency_sq = np.square(np.asarray(data["slambda_1"], dtype=float)).sum(
        axis=1
    )
    scale_frequency_sq = np.square(
        np.asarray(data["slambda_sigma_1"], dtype=float)
    ).sum(axis=1)
    coords = {
        "observation": np.arange(data["N"]),
        "mean_basis": np.arange(data["NBgp_1"]),
        "scale_basis": np.arange(data["NBgp_sigma_1"]),
    }
    with pm.Model(coords=coords) as model:
        mean_design = pm.Data(
            "mean_design", mean_basis, dims=("observation", "mean_basis")
        )
        scale_design = pm.Data(
            "scale_design", scale_basis, dims=("observation", "scale_basis")
        )
        intercept = pm.StudentT("Intercept", nu=3, mu=-13, sigma=36, initval=-13)
        intercept_sigma = pm.StudentT(
            "Intercept_sigma", nu=3, mu=0, sigma=10, initval=0
        )
        # A float keeps sigma**2 from overflowing an integer tensor.
        sdgp = pm.HalfStudentT("sdgp_1", nu=3, sigma=36.0, initval=36)
        sdgp_sigma = pm.HalfStudentT("sdgp_sigma_1", nu=3, sigma=36.0, initval=1)
        length = pm.InverseGamma("lscale_1", alpha=1.124909, beta=0.0177, initval=0.1)
        length_sigma = pm.InverseGamma(
            "lscale_sigma_1", alpha=1.124909, beta=0.0177, initval=0.1
        )
        z = pm.Normal("zgp_1", 0, 1, dims="mean_basis")
        z_sigma = pm.Normal("zgp_sigma_1", 0, 1, dims="scale_basis")
        mean_spectral_sd = (
            sdgp
            * (np.sqrt(2 * np.pi) * length) ** (data["Dgp_1"] / 2)
            * pt.exp(-0.25 * length**2 * mean_frequency_sq)
        )
        scale_spectral_sd = (
            sdgp_sigma
            * (np.sqrt(2 * np.pi) * length_sigma) ** (data["Dgp_sigma_1"] / 2)
            * pt.exp(-0.25 * length_sigma**2 * scale_frequency_sq)
        )
        mu = pm.Deterministic(
            "mu", intercept + mean_design @ (mean_spectral_sd * z), dims="observation"
        )
        sigma = pm.Deterministic(
            "sigma",
            pt.exp(intercept_sigma + scale_design @ (scale_spectral_sd * z_sigma)),
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
