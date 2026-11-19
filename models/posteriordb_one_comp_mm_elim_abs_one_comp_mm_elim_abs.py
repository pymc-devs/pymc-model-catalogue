"""A one compartment model.

Stan comparison: MATCH.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.ode import DifferentialEquation


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_one_comp_mm_elim_abs.json.zip"
        ) as archive,
        archive.open("one_comp_mm_elim_abs.json") as stream,
    ):
        data = json.load(stream)
    times = np.asarray(data["times"], dtype=float)
    observed = np.asarray(data["C_hat"], dtype=float)
    dose = float(data["D"])
    volume = float(data["V"])

    def concentration_dynamics(concentration, time, rates):
        absorption, half_saturation, maximum_elimination = rates[0], rates[1], rates[2]
        absorbed_dose = pt.switch(
            time > 0, dose * absorption * pt.exp(-absorption * time) / volume, 0.0
        )
        eliminated_dose = (
            maximum_elimination
            * concentration[0]
            / (volume * (half_saturation + concentration[0]))
        )
        return [absorbed_dose - eliminated_dose]

    # Stan uses integrate_ode_bdf with its default solver controls.
    # PyMC 6.4 DifferentialEquation uses SciPy odeint (LSODA) defaults.
    # The equations are unchanged, but solver and tolerance differences can
    # change evaluated logp and gradients, not just a normalization constant.
    dynamics = DifferentialEquation(
        func=concentration_dynamics,
        times=times,
        n_states=1,
        n_theta=3,
        t0=float(data["t0"]),
    )
    with pm.Model(coords={"time": times}) as model:
        k_a = pm.HalfCauchy("k_a", beta=1, initval=1.0)
        K_m = pm.HalfCauchy("K_m", beta=1, initval=1.0)
        V_m = pm.HalfCauchy("V_m", beta=1, initval=1.0)
        sigma = pm.HalfCauchy("sigma", beta=1, initval=0.2)
        concentration = pm.Deterministic(
            "C",
            dynamics(y0=np.zeros(1), theta=pt.stack([k_a, K_m, V_m]))[:, 0],
            dims="time",
        )
        pm.LogNormal(
            "C_hat",
            mu=pt.log(concentration),
            sigma=sigma,
            observed=observed,
            dims="time",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
