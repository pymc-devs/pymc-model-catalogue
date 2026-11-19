"""Lotka-Volterra Error Model.

Stan comparison: MISMATCH.
Different ODE solvers are a suspected cause; this is not confirmed.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.ode import DifferentialEquation


def build_model():
    def _lotka_volterra_dynamics(population, time, rates):
        prey, predator = population[0], population[1]
        growth, predation, mortality, reproduction = (
            rates[0],
            rates[1],
            rates[2],
            rates[3],
        )
        interaction = prey * predator
        return [
            growth * prey - predation * interaction,
            reproduction * interaction - mortality * predator,
        ]

    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_hudson_lynx_hare.json.zip"
        ) as archive,
        archive.open("hudson_lynx_hare.json") as stream,
    ):
        data = json.load(stream)
    times = np.asarray(data["ts"], dtype=float)
    initial_observed = np.asarray(data["y_init"], dtype=float)
    observed = np.asarray(data["y"], dtype=float)
    # Stan uses RK45 with rtol=1e-5, atol=1e-3 and max_num_steps=500.
    # PyMC 6.4 DifferentialEquation uses SciPy odeint (LSODA) defaults.
    # The equations are unchanged, but solver and tolerance differences can
    # change evaluated logp and gradients, not just a normalization constant.
    dynamics = DifferentialEquation(
        func=_lotka_volterra_dynamics,
        times=times,
        n_states=2,
        n_theta=4,
        t0=0,
    )
    coords = {
        "time": times,
        "species": ["hare", "lynx"],
        "rate": ["growth", "predation", "mortality", "reproduction"],
    }
    with pm.Model(coords=coords) as model:
        # Stan evaluates Normal densities on positive-constrained theta.
        # TruncatedNormal additionally normalizes that restriction; fixed
        # prior means and scales make this only a constant logp offset.
        theta = pm.TruncatedNormal(
            "theta",
            mu=np.array([1.0, 0.05, 1.0, 0.05]),
            sigma=np.array([0.5, 0.05, 0.5, 0.05]),
            lower=0,
            dims="rate",
            initval=np.array([0.55, 0.028, 0.84, 0.026]),
        )
        z_init = pm.LogNormal(
            "z_init", mu=np.log(10.0), sigma=1, dims="species", initval=initial_observed
        )
        sigma = pm.LogNormal(
            "sigma", mu=-1, sigma=1, dims="species", initval=np.full(2, np.exp(-1.0))
        )
        populations = pm.Deterministic(
            "z", dynamics(y0=z_init, theta=theta), dims=("time", "species")
        )
        pm.LogNormal(
            "y_init",
            mu=pt.log(z_init),
            sigma=sigma,
            observed=initial_observed,
            dims="species",
        )
        pm.LogNormal(
            "y",
            mu=pt.log(populations),
            sigma=sigma,
            observed=observed,
            dims=("time", "species"),
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
