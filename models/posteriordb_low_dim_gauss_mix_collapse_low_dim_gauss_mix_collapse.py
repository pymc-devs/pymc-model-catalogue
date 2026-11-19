"""
Model: A Two-Dimensional (unordered) Gaussian Mixture Model
Source: posteriordb/posterior_database/models/stan/low_dim_gauss_mix_collapse.stan
Description: A Two-Dimensional Gaussian Mixture Model

Stan comparison:
- A: Stan = -1789.561133899328; PyMC = -1788.1748396462788.
- B: Stan = -1687.6652702311083; PyMC = -1686.2789759780596.
- Raw delta: Stan = 101.89586366821982; PyMC = 101.89586366821914.
- Raw comparison: MATCH; residual = 6.821210263296962e-13.
- Tolerance: 2.018958636682198e-06
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
            Path(__file__).parent
            / "data"
            / "posteriordb_low_dim_gauss_mix_collapse.json.zip"
        ) as archive,
        archive.open("low_dim_gauss_mix_collapse.json") as stream,
    ):
        data = json.load(stream)
    y = np.asarray(data["y"], dtype=float)
    coords = {"component": np.arange(2), "observation": np.arange(data["N"])}
    with pm.Model(coords=coords) as model:
        mu = pm.Normal(
            "mu", mu=0, sigma=2, dims="component", initval=np.array([-1.0, 1.0])
        )
        sigma = pm.HalfNormal(
            "sigma", sigma=2, dims="component", initval=np.full(2, 2.0)
        )
        theta = pm.Beta("theta", alpha=5, beta=5, initval=0.5)
        weights = pt.stack([theta, 1 - theta])
        pm.NormalMixture(
            "y", w=weights, mu=mu, sigma=sigma, observed=y, dims="observation"
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
