"""Multiple Predictors Log-linear Model.

Stan comparison:
- A: Stan = -5158.349540037462; PyMC = -5158.349540037472.
- B: Stan = -298568.5512107776; PyMC = -298568.5512107769.
- Raw delta: Stan = -293410.20167074015; PyMC = -293410.20167073945.
- Raw comparison: MATCH; residual = -6.984919309616089e-10.
- Tolerance: 0.0029351020167074017
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
    height = np.asarray(data["height"], dtype=float)
    male = np.asarray(data["male"], dtype=float)
    design = np.column_stack(
        (
            np.ones(len(response)),
            height,
            male,
        )
    )
    coords = {
        "obs_id": np.arange(len(response)),
        "coefficient": ["intercept", "height", "male"],
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
