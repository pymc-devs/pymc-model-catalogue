"""Multiple Predictors Linear Model.

Stan comparison: MATCH.
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
    response = np.asarray(data["weight"], dtype=float)
    diam1 = np.asarray(data["diam1"], dtype=float)
    diam2 = np.asarray(data["diam2"], dtype=float)
    canopy_height = np.asarray(data["canopy_height"], dtype=float)
    total_height = np.asarray(data["total_height"], dtype=float)
    density = np.asarray(data["density"], dtype=float)
    group = np.asarray(data["group"], dtype=float)
    design = np.column_stack(
        (
            np.ones(len(response)),
            diam1,
            diam2,
            canopy_height,
            total_height,
            density,
            group,
        )
    )
    coords = {
        "obs_id": np.arange(len(response)),
        "coefficient": [
            "intercept",
            "diam1",
            "diam2",
            "canopy_height",
            "total_height",
            "density",
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
            "weight",
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
