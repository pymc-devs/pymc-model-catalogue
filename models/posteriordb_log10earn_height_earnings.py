"""
Model: One Predictor Log10-linear Model
Source: posteriordb/posterior_database/models/stan/log10earn_height.stan
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
Description: log10(earn) ~ height

Stan comparison:
- A: Stan = -6910.397305276828; PyMC = -6910.397305276836.
- B: Stan = -185649.29039842458; PyMC = -185649.2903984243.
- Raw delta: Stan = -178738.89309314775; PyMC = -178738.89309314746.
- Raw comparison: MATCH; residual = -2.9103830456733704e-10.
- Tolerance: 0.0017883889309314774
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
    response = np.log10(np.asarray(data["earn"], dtype=float))
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
            "log10_earn",
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
