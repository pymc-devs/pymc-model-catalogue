"""Explicit population inclusion with constant detection.

Source: Kéry and Schaub (2011), chapter 6.
"""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("M0_model_discrete", posterior)
    history = np.asarray(data["y"], dtype="int64")
    coords = {
        "individual": np.arange(data["M"]),
        "capture_period": np.arange(data["T"]),
    }
    with pm.Model(coords=coords) as model:
        omega = pm.Uniform("omega", 0, 1)
        p = pm.Uniform("p", 0, 1)
        z = pm.Bernoulli(
            "z",
            p=omega,
            dims="individual",
            initval=np.ones(data["M"], dtype="int64"),
        )
        pm.Bernoulli(
            "y",
            p=z[:, None] * p,
            observed=history,
            dims=("individual", "capture_period"),
        )
        pm.Deterministic("N", z.sum())
        no_capture = (1 - p) ** int(data["T"])
        pm.Deterministic(
            "omega_nd", omega * no_capture / (omega * no_capture + 1 - omega)
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("M0_model_discrete")
