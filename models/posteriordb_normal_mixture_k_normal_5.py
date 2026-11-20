"""
Model: K Component Gaussian Mixture Model
Source: posteriordb/posterior_database/models/stan/normal_mixture_k.stan
Description: a Gaussian mixture with unknown proportions, means and variances

Stan comparison:
- A: Stan = -5480.534220963223; PyMC = -5488.869093394437.
- B: Stan = -5514.887037286612; PyMC = -5523.221909717825.
- Raw delta: Stan = -34.35281632338956; PyMC = -34.35281632338774.
- Raw comparison: MATCH; residual = -1.8189894035458565e-12.
- Tolerance: 1.3435281632338956e-06
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_normal_5.json.zip"
        ) as archive,
        archive.open("normal_5.json") as stream,
    ):
        data = json.load(stream)
    y = np.asarray(data["y"], dtype=float)
    n_components = int(data["K"])
    coords = {
        "component": np.arange(n_components),
        "observation": np.arange(data["N"]),
    }
    with pm.Model(coords=coords) as model:
        theta = pm.Dirichlet(
            "theta",
            a=np.ones(n_components),
            dims="component",
            initval=np.full(n_components, 1.0 / n_components),
        )
        mu = pm.Normal(
            "mu",
            mu=0,
            sigma=10,
            dims="component",
            initval=np.linspace(-2.0, 2.0, n_components),
        )
        sigma = pm.Uniform(
            "sigma",
            lower=0,
            upper=10,
            dims="component",
            initval=np.full(n_components, 5.0),
        )
        pm.NormalMixture(
            "y", w=theta, mu=mu, sigma=sigma, observed=y, dims="observation"
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
