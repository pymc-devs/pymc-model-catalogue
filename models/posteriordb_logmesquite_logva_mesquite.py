"""
Model: Log-Log Model
Source: posteriordb/posterior_database/models/stan/logmesquite_logva.stan
Authors: Andrew Gelman, Jennifer Hill
Description: log(weight) ~ log(canopy.volume) + log(canopy.area) + group canopy_volume <- diam1 *
    diam2 * canopy_height canopy_area <- diam1 * diam2

Stan comparison:
- A: Stan = -704.8164216957422; PyMC = -704.816421695742.
- B: Stan = -881.2893371061535; PyMC = -881.2893371061538.
- Raw delta: Stan = -176.47291541041136; PyMC = -176.47291541041182.
- Raw comparison: MATCH; residual = 4.547473508864641e-13.
- Tolerance: 2.764729154104118e-06
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_mesquite.json.zip"
        ) as archive,
        archive.open("mesquite.json") as stream,
    ):
        data = json.load(stream)
    response = np.log(np.asarray(data["weight"], dtype=float))
    diam1 = np.asarray(data["diam1"], dtype=float)
    diam2 = np.asarray(data["diam2"], dtype=float)
    canopy_height = np.asarray(data["canopy_height"], dtype=float)
    log_canopy_volume = np.log(diam1 * diam2 * canopy_height)
    log_canopy_area = np.log(diam1 * diam2)
    group = np.asarray(data["group"], dtype=float)
    design = np.column_stack(
        (
            np.ones(len(response)),
            log_canopy_volume,
            log_canopy_area,
            group,
        )
    )
    coords = {
        "obs_id": np.arange(len(response)),
        "coefficient": ["intercept", "log_canopy_volume", "log_canopy_area", "group"],
    }
    initial_beta = np.zeros(design.shape[1])
    initial_beta[0] = response.mean()
    with pm.Model(coords=coords) as model:
        x = pm.Data("design", design, dims=("obs_id", "coefficient"))
        beta = pm.Flat("beta", dims="coefficient", initval=initial_beta)
        sigma = pm.HalfFlat("sigma", initval=max(float(response.std()), 1.0))
        pm.Normal(
            "log_weight",
            mu=pm.math.dot(x, beta),
            sigma=sigma,
            observed=response,
            dims="obs_id",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
