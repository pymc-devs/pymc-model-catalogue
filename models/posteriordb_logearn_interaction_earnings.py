"""Multiple Predictors Interacting Log-linear Model.

Stan comparison:
- A: Stan = -7115.324664263505; PyMC = -7115.324664263505.
- B: Stan = -188564.7367665402; PyMC = -188564.7367665403.
- Raw delta: Stan = -181449.41210227672; PyMC = -181449.4121022768.
- Raw comparison: MATCH; residual = 8.731149137020111e-11.
- Tolerance: 0.001815494121022768
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
            height * male,
        )
    )
    coords = {
        "obs_id": np.arange(len(response)),
        "coefficient": ["intercept", "height", "male", "height:male"],
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
