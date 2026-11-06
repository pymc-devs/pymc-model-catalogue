"""Waterborne SIR dynamics with Poisson incidence and lognormal bacteria counts.

Source: posteriordb sir, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4.
Initial compartment sizes and the water concentration constant remain fixed.
Incidence counts measure susceptible losses between consecutive observation times.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.ode import DifferentialEquation
from pytensor import config
from pytensor.compile.mode import Mode

from ._data import load_data


def _sir_waterborne_dynamics(state, time, rates):
    susceptible, infected, bacteria = state[0], state[1], state[3]
    beta, gamma, xi, delta = rates[0], rates[1], rates[2], rates[3]
    infections = beta * susceptible * bacteria / (bacteria + 1_000_000.0)
    recoveries = gamma * infected
    return [
        -infections,
        infections - recoveries,
        recoveries,
        xi * infected - delta * bacteria,
    ]


def build_model(posterior: str | None = None):
    """Build the native model for one canonical posterior pair."""
    data = load_data("sir", posterior)
    times = np.asarray(data["t"], dtype=float)
    initial_state = np.asarray(data["y0"], dtype=float)
    incidence_observed = np.asarray(data["stoi_hat"], dtype="int64")
    bacteria_observed = np.asarray(data["B_hat"], dtype=float)
    with config.change_flags(mode=Mode(linker="cvm", optimizer="fast_run")):
        dynamics = DifferentialEquation(
            func=_sir_waterborne_dynamics,
            times=times,
            n_states=4,
            n_theta=4,
            t0=0,
        )
    coords = {
        "time": times,
        "compartment": ["susceptible", "infected", "recovered", "bacteria"],
    }
    with pm.Model(coords=coords) as model:
        beta = pm.HalfCauchy("beta", beta=2.5, initval=0.2)
        gamma = pm.HalfCauchy("gamma", beta=1, initval=0.1)
        xi = pm.HalfCauchy("xi", beta=25, initval=10.0)
        delta = pm.HalfCauchy("delta", beta=1, initval=0.1)
        trajectory = pm.Deterministic(
            "y",
            dynamics(y0=initial_state, theta=pt.stack([beta, gamma, xi, delta])),
            dims=("time", "compartment"),
        )
        susceptible = pt.concatenate([initial_state[:1], trajectory[:, 0]])
        incidence = pm.Deterministic(
            "incidence", susceptible[:-1] - susceptible[1:], dims="time"
        )
        pm.Poisson("stoi_hat", mu=incidence, observed=incidence_observed, dims="time")
        pm.LogNormal(
            "B_hat",
            mu=pt.log(trajectory[:, 3]),
            sigma=0.15,
            observed=bacteria_observed,
            dims="time",
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("sir")
