"""Log-Log Model.

Stan comparison:
- A: Stan = -371.95733646573; PyMC = -371.9573364657301.
- B: Stan = -741.7119178580231; PyMC = -741.711917858023.
- Raw delta: Stan = -369.75458139229306; PyMC = -369.75458139229283.
- Raw comparison: MATCH; residual = -2.2737367544323206e-13.
- Tolerance: 4.697545813922931e-06
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
    design = np.column_stack(
        (
            np.ones(len(response)),
            log_canopy_volume,
        )
    )
    coords = {
        "obs_id": np.arange(len(response)),
        "coefficient": ["intercept", "log_canopy_volume"],
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
