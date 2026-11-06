"""Lotka-Volterra populations with lognormal measurement errors.

Source: posteriordb lotka_volterra, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4.
Positive normal rates retain their zero lower bound.
The initial populations remain latent and have their own observed measurements.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.ode import DifferentialEquation
from pytensor import config
from pytensor.compile.mode import Mode

from ._data import load_data


def _lotka_volterra_dynamics(population, time, rates):
    prey, predator = population[0], population[1]
    growth, predation, mortality, reproduction = rates[0], rates[1], rates[2], rates[3]
    interaction = prey * predator
    return [
        growth * prey - predation * interaction,
        reproduction * interaction - mortality * predator,
    ]


def build_model(posterior: str | None = None):
    """Build the native model for one canonical posterior pair."""
    data = load_data("lotka_volterra", posterior)
    times = np.asarray(data["ts"], dtype=float)
    initial_observed = np.asarray(data["y_init"], dtype=float)
    observed = np.asarray(data["y"], dtype=float)
    with config.change_flags(mode=Mode(linker="cvm", optimizer="fast_run")):
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
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("lotka_volterra")
