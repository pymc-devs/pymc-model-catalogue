"""
Model: Structured Time Series model with stochastic level and stochastic seasonal
Source: posteriordb/posterior_database/models/stan/state_space_stochastic_level_stochastic_seasonal.stan
Description: Structured Time Series model with stochastic level and stochastic seasonal

Stan comparison:
- A: Stan = -890.3549543316059; PyMC = -893.569279107265.
- B: Stan = -897.3775675171354; PyMC = -900.591892292794.
- Raw delta: Stan = -7.022613185529508; PyMC = -7.022613185529053.
- Raw comparison: MATCH; residual = -4.547473508864641e-13.
- Tolerance: 1.070226131855295e-06
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm
from pymc.distributions.transforms import Ordered


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_uk_drivers.json.zip"
        ) as archive,
        archive.open("uk_drivers.json") as stream,
    ):
        data = json.load(stream)
    y = np.asarray(data["y"], dtype=float)
    n_time = int(data["n"])
    center = float(y.mean())
    radius = 3 * float(y.std(ddof=1))
    coords = {"time": np.arange(n_time), "scale": ["seasonal", "level", "observation"]}
    with pm.Model(coords=coords) as model:
        # The three positive scales have an ordered half-Student-t density.
        sigma = pm.HalfStudentT(
            "sigma",
            nu=4,
            sigma=1,
            dims="scale",
            default_transform=None,
            transform=Ordered(positive=True),
            initval=np.array([0.05, 0.1, 0.2]),
        )
        beta = pm.Flat("beta", initval=0)
        regression_lambda = pm.Flat("lambda", initval=0)
        mu = pm.Uniform(
            "mu",
            lower=center - radius,
            upper=center + radius,
            dims="time",
            initval=np.full(n_time, center),
        )
        pm.Potential(
            "level_transition",
            pm.logp(pm.Normal.dist(mu=mu[:-1], sigma=sigma[1]), mu[1:]).sum(),
        )
        seasonal = pm.AR(
            "seasonal",
            rho=-np.ones(11),
            sigma=sigma[0],
            ar_order=11,
            init_dist=pm.Flat.dist(shape=11),
            dims="time",
            initval=np.zeros(n_time),
        )
        x = pm.Data("x", np.asarray(data["x"], dtype=float), dims="time")
        w = pm.Data("w", np.asarray(data["w"], dtype=float), dims="time")
        expectation = mu + beta * x + regression_lambda * w + seasonal
        pm.Normal("y", mu=expectation, sigma=sigma[2], observed=y, dims="time")
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
