"""A pooled linear model for the Radon dataset.

Stan comparison:
- A: Stan = -2074.1853621070295; PyMC = -2073.49221499042.
- B: Stan = -1725.9305325752941; PyMC = -1725.237385458685.
- Raw delta: Stan = 348.25482953173537; PyMC = 348.25482953173514.
- Raw comparison: MATCH; residual = 2.2737367544323206e-13.
- Tolerance: 4.482548295317354e-06
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_radon_mn.json.zip"
        ) as archive,
        archive.open("radon_mn.json") as stream,
    ):
        data = json.load(stream)
    coords = {"obs_id": np.arange(data["N"])}
    with pm.Model(coords=coords) as model:
        floor = pm.Data(
            "floor_measure",
            np.asarray(data["floor_measure"], dtype=float),
            dims="obs_id",
        )
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma_y = pm.HalfNormal("sigma_y", sigma=1, initval=1)
        pm.Normal(
            "log_radon",
            mu=alpha + beta * floor,
            sigma=sigma_y,
            observed=np.asarray(data["log_radon"], dtype=float),
            dims="obs_id",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
