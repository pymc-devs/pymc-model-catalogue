"""
Model: Log-Log Model
Source: posteriordb/posterior_database/models/stan/logmesquite_logvas.stan
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
Description: log(weight) ~ log(canopy.volume) + log(canopy.area) + log(canopy.shape) +
    log(total.height) + log(density) + group canopy_volume <- diam1 * diam2 * canopy_height
    canopy_area <- diam1 * diam2 canopy_shape <- diam1 / diam2

Stan comparison:
- A: Stan = -888.1385657866779; PyMC = -888.1385657866781.
- B: Stan = -839.6181961051307; PyMC = -839.6181961051307.
- Raw delta: Stan = 48.52036968154721; PyMC = 48.520369681547436.
- Raw comparison: MATCH; residual = -2.2737367544323206e-13.
- Tolerance: 1.4852036968154742e-06
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
    log_canopy_shape = np.log(diam1 / diam2)
    log_total_height = np.log(np.asarray(data["total_height"], dtype=float))
    log_density = np.log(np.asarray(data["density"], dtype=float))
    group = np.asarray(data["group"], dtype=float)
    design = np.column_stack(
        (
            np.ones(len(response)),
            log_canopy_volume,
            log_canopy_area,
            log_canopy_shape,
            log_total_height,
            log_density,
            group,
        )
    )
    coords = {
        "obs_id": np.arange(len(response)),
        "coefficient": [
            "intercept",
            "log_canopy_volume",
            "log_canopy_area",
            "log_canopy_shape",
            "log_total_height",
            "log_density",
            "group",
        ],
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
