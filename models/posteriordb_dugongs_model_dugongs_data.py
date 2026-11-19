"""
Model: Dugong Age and Length
Source: posteriordb/posterior_database/models/stan/dugongs_model.stan
Authors: MRC Biostatistics Unit (cited work)
Description: Model of age and length data of dugongs using a nonlinear growth curve with no
    inflection point.

Stan comparison:
- A: Stan = -131.96739041849747; PyMC = -131.96739060844538.
- B: Stan = -109.4324822421098; PyMC = -109.43248243205778.
- Raw delta: Stan = 22.534908176387674; PyMC = 22.534908176387603.
- Raw comparison: MATCH; residual = 7.105427357601002e-14.
- Tolerance: 1.2253490817638768e-06
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_dugongs_data.json.zip"
        ) as archive,
        archive.open("dugongs_data.json") as stream,
    ):
        data = json.load(stream)
    with pm.Model(coords={"animal": np.arange(1, data["N"] + 1)}) as model:
        x = pm.Data("x", np.asarray(data["x"], dtype=float), dims="animal")
        alpha = pm.Normal("alpha", 0, 1000, initval=2.5)
        beta = pm.Normal("beta", 0, 1000, initval=1.0)
        growth = pm.Uniform("lambda", 0.5, 1)
        tau = pm.Gamma("tau", alpha=0.0001, beta=0.0001, initval=1.0)
        sigma = pm.Deterministic("sigma", 1 / pm.math.sqrt(tau))
        pm.Normal(
            "Y",
            mu=alpha - beta * growth**x,
            sigma=sigma,
            observed=data["Y"],
            dims="animal",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
