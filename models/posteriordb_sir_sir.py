"""
Model: Simple SIR model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/sir.stan
Authors: Grad, Yonatan H and Miller, Joel C and Lipsitch, Marc (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/sir.py
Upstream catalogue contributor: Mans Magnusson
Description: A relatively simple susceptible-infected-recovered (SIR) model using the RK45 ODE
    integrator.

Data: sir (upstream posterior: sir-sir)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/sir.json.zip
    Archive SHA-256: 76cb34e20302aa8d5db11804ad1cb5535012fc16a53b6e15915b80a1bb26471d
    Local archive: data/posteriordb_sir.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/sir-sir.json
References: grad2012cholera (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/stat_comp_benchmarks/tree/master/benchmarks/sir

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Waterborne SIR dynamics with Poisson incidence and lognormal bacteria counts.

Source: posteriordb sir, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4.
Initial compartment sizes and the water concentration constant remain fixed.
Incidence counts measure susceptible losses between consecutive observation times.

Benchmark results:
- Original: logp = -8849.2832, grad norm = 6315.5231, 14027.6 us/call (5 evals).
- Frozen: logp = -8849.2832, grad norm = 6315.5231, 12936.7 us/call (5 evals).
- Local NUMBA validation, including object-mode ODE execution; short timings are indicative.
- Environment: PyMC 6.4.0+9.gaf565aa28, PyTensor 3.3.8+30.g9a7f24ce0, Numba 0.68.0.
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
