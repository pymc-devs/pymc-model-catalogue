"""
Model: Two Pool Linear Model with Feedback
Source: posteriordb/posterior_database/models/stan/soil_incubation.stan
Authors: Carlos Sierra, Markus Müller, Susan E Trumbore
Description: A statistical two-pool model with decomposition rates for each pool as well as transfer
    rates from one pool to another.

Stan comparison:
- A: Stan = -32.35121812756171; PyMC = -28.885478931878026.
- B: Stan = -31.536547018749822; PyMC = -28.070807010787178.
- Raw delta: Stan = 0.8146711088118863; PyMC = 0.8146719210908486.
- Raw comparison: MATCH; residual = -8.122789623143944e-07.
- Tolerance: 1.0081467192109083e-06

Benchmark results:
- Original:  logp = -42.8159, grad norm = 15.6056, 3636.9 us/call (5 evals).
- Frozen:    logp = -42.8159, grad norm = 15.6056, 4063.1 us/call (5 evals).
- The ODE solver ran in object mode.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.ode import DifferentialEquation


def build_model():
    def _soil_incubation_dynamics(carbon, time, rates):
        """Return carbon derivatives for decomposition and transfer between two pools."""
        k1, k2, alpha21, alpha12 = rates[0], rates[1], rates[2], rates[3]
        return [
            -k1 * carbon[0] + alpha12 * k2 * carbon[1],
            alpha21 * k1 * carbon[0] - k2 * carbon[1],
        ]

    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_soil_carbon.json.zip"
        ) as archive,
        archive.open("soil_carbon.json") as stream,
    ):
        data = json.load(stream)
    times = np.asarray(data["ts"], dtype=float)
    total_carbon = float(data["totalC_t0"])
    # Stan uses integrate_ode_rk45 with its default solver controls.
    # PyMC 6.4 DifferentialEquation uses SciPy odeint (LSODA) defaults.
    # The equations are unchanged, but solver and tolerance differences can
    # change evaluated logp and gradients, not just a normalization constant.
    dynamics = DifferentialEquation(
        func=_soil_incubation_dynamics,
        times=times,
        n_states=2,
        n_theta=4,
        t0=float(data["t0"]),
    )
    with pm.Model(coords={"time": times, "pool": [1, 2]}) as model:
        gamma = pm.Beta("gamma", alpha=10, beta=1, initval=0.9)
        k1 = pm.HalfNormal("k1", sigma=1, initval=0.1)
        k2 = pm.HalfNormal("k2", sigma=1, initval=0.01)
        alpha21 = pm.HalfNormal("alpha21", sigma=1, initval=0.1)
        alpha12 = pm.HalfNormal("alpha12", sigma=1, initval=0.1)
        sigma = pm.HalfCauchy("sigma", beta=1)
        initial_carbon = total_carbon * pt.stack([gamma, 1 - gamma])
        rates = pt.stack([k1, k2, alpha21, alpha12])
        carbon = dynamics(y0=initial_carbon, theta=rates)
        evolved_carbon = total_carbon - carbon.sum(axis=1)
        pm.Normal(
            "eCO2mean",
            mu=evolved_carbon,
            sigma=sigma,
            observed=np.asarray(data["eCO2mean"], dtype=float),
            dims="time",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
