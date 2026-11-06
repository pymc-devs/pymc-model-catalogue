"""Seasonal state model from the posteriordb stochastic level model.

The initial eleven seasonal states and regression coefficients have flat densities.
The level bounds apply to every time point without transition normalization.
Source: sinhrks/stan-statespace, fig07_02, as specified by posteriordb.
"""

import numpy as np
import pymc as pm
from pymc.distributions.transforms import Ordered

from ._data import load_data


def build_model(posterior: str | None = None):
    model_name = "state_space_stochastic_level_stochastic_seasonal"
    data = load_data(model_name, posterior)
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
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("state_space_stochastic_level_stochastic_seasonal")
