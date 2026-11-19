"""
Model: Two Component Gaussian Mixture Model
Source: posteriordb/posterior_database/models/stan/normal_mixture.stan
Description: a gaussian mixture with unknown proportion and means, known variance p(y|mu,theta) =
    theta * Normal(y|mu[1],1) + (1-theta) * Normal(y|mu[2],1)

Stan comparison:
- A: Stan = -52362.52849736699; PyMC = -52362.52849743095.
- B: Stan = -51359.06365560339; PyMC = -51359.06365566734.
- Raw delta: Stan = 1003.4648417635981; PyMC = 1003.4648417636054.
- Raw comparison: MATCH; residual = -7.275957614183426e-12.
- Tolerance: 1.1034648417636055e-05
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_normal_2.json.zip"
        ) as archive,
        archive.open("normal_2.json") as stream,
    ):
        data = json.load(stream)
    y = np.asarray(data["y"], dtype=float)
    coords = {"component": np.arange(2), "observation": np.arange(data["N"])}
    with pm.Model(coords=coords) as model:
        theta = pm.Uniform("theta", lower=0, upper=1, initval=0.5)
        mu = pm.Normal(
            "mu", mu=0, sigma=10, dims="component", initval=np.array([-1.0, 1.0])
        )
        weights = pt.stack([theta, 1 - theta])
        pm.NormalMixture(
            "y", w=weights, mu=mu, sigma=1.0, observed=y, dims="observation"
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
