"""
Model: Simple SIR model
Source: posteriordb/posterior_database/models/stan/sir.stan
Authors: Grad, Yonatan H and Miller, Joel C and Lipsitch, Marc (cited work)
Description: Waterborne SIR dynamics with Poisson incidence and lognormal bacteria observations.

Stan comparison:
- A: Stan = -9172.264842237479; PyMC = -9169.49208716104.
- B: Stan = -8161.621032291843; PyMC = -8158.848429295431.
- Raw delta: Stan = 1010.6438099456354; PyMC = 1010.6436578656085.
- Raw comparison: MISMATCH; residual = 0.0001520800269645406.
- Tolerance: 1.1106438099456355e-05
- Different ODE solvers are a suspected cause; this is not confirmed.

Benchmark results:
- Original:  logp = -8849.2832, grad norm = 6315.5231, 14027.6 us/call (5 evals).
- Frozen:    logp = -8849.2832, grad norm = 6315.5231, 12936.7 us/call (5 evals).
- Local NUMBA validation, including object-mode ODE execution; short timings are indicative.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.ode import DifferentialEquation


def build_model():
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

    with (
        ZipFile(Path(__file__).parent / "data" / "posteriordb_sir.json.zip") as archive,
        archive.open("sir.json") as stream,
    ):
        data = json.load(stream)
    times = np.asarray(data["t"], dtype=float)
    initial_state = np.asarray(data["y0"], dtype=float)
    incidence_observed = np.asarray(data["stoi_hat"], dtype="int64")
    bacteria_observed = np.asarray(data["B_hat"], dtype=float)
    # Stan uses integrate_ode_rk45 with its default solver controls.
    # PyMC 6.4 DifferentialEquation uses SciPy odeint (LSODA) defaults.
    # The equations are unchanged, but solver and tolerance differences can
    # change evaluated logp and gradients, not just a normalization constant.
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
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
