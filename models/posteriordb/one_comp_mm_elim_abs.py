"""One-compartment absorption with Michaelis-Menten elimination.

Source: posteriordb one_comp_mm_elim_abs, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4.
The initial concentration is zero, as specified by the published database model.
Native ODE integration separates the concentration dynamics from lognormal measurement errors.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.ode import DifferentialEquation
from pytensor import config
from pytensor.compile.mode import Mode

from ._data import load_data


def build_model(posterior: str | None = None):
    """Build the native model for one canonical posterior pair."""
    data = load_data("one_comp_mm_elim_abs", posterior)
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

    with config.change_flags(mode=Mode(linker="cvm", optimizer="fast_run")):
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
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("one_comp_mm_elim_abs")
