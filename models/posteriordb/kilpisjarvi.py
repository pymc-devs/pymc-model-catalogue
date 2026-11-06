"""Gaussian temperature regression from the posteriordb kilpisjarvi probability model.

The observation scale has the source model's improper positive flat density.
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4.
"""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("kilpisjarvi", posterior)
    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    with pm.Model(coords={"year": np.arange(int(data["N"]))}) as model:
        alpha = pm.Normal("alpha", mu=data["pmualpha"], sigma=data["psalpha"])
        beta = pm.Normal("beta", mu=data["pmubeta"], sigma=data["psbeta"])
        sigma = pm.HalfFlat("sigma", initval=1)
        time = pm.Data("x", x, dims="year")
        pm.Normal("y", mu=alpha + beta * time, sigma=sigma, observed=y, dims="year")
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("kilpisjarvi")
