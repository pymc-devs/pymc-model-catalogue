"""One Predictor Linear Model.

Stan comparison:
- A: Stan = -400986929008.98285; PyMC = -400986929008.98267.
- B: Stan = -470226871825.1733; PyMC = -470226871825.1741.
- Raw delta: Stan = -69239942816.19043; PyMC = -69239942816.1914.
- Raw comparison: MATCH; residual = 0.0009765625.
- Tolerance: 692.3994291619141
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_earnings.json.zip"
        ) as archive,
        archive.open("earnings.json") as stream,
    ):
        data = json.load(stream)
    response = np.asarray(data["earn"], dtype=float)
    height = np.asarray(data["height"], dtype=float)
    design = np.column_stack(
        (
            np.ones(len(response)),
            height,
        )
    )
    coords = {
        "obs_id": np.arange(len(response)),
        "coefficient": ["intercept", "height"],
    }
    initial_beta = np.zeros(design.shape[1])
    initial_beta[0] = response.mean()
    with pm.Model(coords=coords) as model:
        x = pm.Data("design", design, dims=("obs_id", "coefficient"))
        beta = pm.Flat("beta", dims="coefficient", initval=initial_beta)
        sigma = pm.HalfFlat("sigma", initval=max(float(response.std()), 1.0))
        pm.Normal(
            "earn",
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
