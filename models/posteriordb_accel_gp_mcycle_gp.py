"""
Model: Gaussian Processes that model the mean and std of acceleration
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/accel_gp.stan
Authors: Bales, Ben and Pourzanjani, Arya and Vehtari, Aki and Petzold, Linda (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/accel_gp.py
Upstream catalogue contributor: Oliver Järnefelt
Description: accel ~ gp(times, k=40, c=3/2, sigma ~ gp(times, k=20, c=3/2)

Data: mcycle_gp (upstream posterior: mcycle_gp-accel_gp)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/mcycle_gp.json.zip
    Archive SHA-256: 159ed1327b350374e5e299e94cc257f6b9b7428da90658a37233b24aedcf7f65
    Local archive: data/posteriordb_mcycle_gp.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/mcycle_gp-accel_gp.json
References: bales2019selecting (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/bbbales2/cmdstan-warmup/blob/develop/examples/accel_gp/

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Finite-basis acceleration model from posteriordb and Bales (2019).

The supplied eigenfunctions and frequencies define both finite Gaussian processes.
The two processes control the mean and log standard deviation.

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
        sdgp = pm.HalfStudentT("sdgp_1", nu=3, sigma=36, initval=36)
        sdgp_sigma = pm.HalfStudentT("sdgp_sigma_1", nu=3, sigma=36, initval=1)
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
