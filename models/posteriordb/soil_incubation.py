"""Two-pool carbon dynamics from the posteriordb soil_incubation model.

The observation model uses cumulative carbon loss, not measurement error scales.
Source: Sierra et al. (2014), as specified by posteriordb.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.ode import DifferentialEquation
from pytensor import config
from pytensor.compile.mode import Mode

from ._data import load_data


def _soil_incubation_dynamics(carbon, time, rates):
    """Return carbon derivatives for decomposition and transfer between two pools."""
    k1, k2, alpha21, alpha12 = rates[0], rates[1], rates[2], rates[3]
    return [
        -k1 * carbon[0] + alpha12 * k2 * carbon[1],
        alpha21 * k1 * carbon[0] - k2 * carbon[1],
    ]


def build_model(posterior: str | None = None):
    data = load_data("soil_incubation", posterior)
    times = np.asarray(data["ts"], dtype=float)
    total_carbon = float(data["totalC_t0"])
    with config.change_flags(mode=Mode(linker="cvm", optimizer="fast_run")):
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
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("soil_incubation")
