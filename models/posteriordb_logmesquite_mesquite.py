"""
Model: Multiple Predictors Log-Log Model
Source: posteriordb/posterior_database/models/stan/logmesquite.stan
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
Description: log(weight) ~ log(diam1) + log(diam2) + log(canopy.height) + log(total.height) +
    log(density) + group

Stan comparison:
- A: Stan = -727.4638896522694; PyMC = -727.4638896522692.
- B: Stan = -960.0720060170487; PyMC = -960.0720060170488.
- Raw delta: Stan = -232.60811636477933; PyMC = -232.60811636477956.
- Raw comparison: MATCH; residual = 2.2737367544323206e-13.
- Tolerance: 3.3260811636477953e-06
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
    log_diam1 = np.log(np.asarray(data["diam1"], dtype=float))
    log_diam2 = np.log(np.asarray(data["diam2"], dtype=float))
    log_canopy_height = np.log(np.asarray(data["canopy_height"], dtype=float))
    log_total_height = np.log(np.asarray(data["total_height"], dtype=float))
    log_density = np.log(np.asarray(data["density"], dtype=float))
    group = np.asarray(data["group"], dtype=float)
    design = np.column_stack(
        (
            np.ones(len(response)),
            log_diam1,
            log_diam2,
            log_canopy_height,
            log_total_height,
            log_density,
            group,
        )
    )
    coords = {
        "obs_id": np.arange(len(response)),
        "coefficient": [
            "intercept",
            "log_diam1",
            "log_diam2",
            "log_canopy_height",
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
