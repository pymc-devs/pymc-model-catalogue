"""Multiple Predictors Log-Log Model.

Stan comparison:
- A: Stan = -41526.19797084844; PyMC = -41526.19797084852.
- B: Stan = -40580.409830456476; PyMC = -40580.40983045639.
- Raw delta: Stan = 945.7881403919673; PyMC = 945.7881403921347.
- Raw comparison: MATCH; residual = -1.673470251262188e-10.
- Tolerance: 1.0457881403921347e-05
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
    response = np.log(np.asarray(data["earn"], dtype=float))
    log_height = np.log(np.asarray(data["height"], dtype=float))
    male = np.asarray(data["male"], dtype=float)
    design = np.column_stack(
        (
            np.ones(len(response)),
            log_height,
            male,
        )
    )
    coords = {
        "obs_id": np.arange(len(response)),
        "coefficient": ["intercept", "log_height", "male"],
    }
    initial_beta = np.zeros(design.shape[1])
    initial_beta[0] = response.mean()
    with pm.Model(coords=coords) as model:
        x = pm.Data("design", design, dims=("obs_id", "coefficient"))
        beta = pm.Flat("beta", dims="coefficient", initval=initial_beta)
        sigma = pm.HalfFlat("sigma", initval=max(float(response.std()), 1.0))
        pm.Normal(
            "log_earn",
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
