"""Detection probabilities vary by occasion.

Source: Kéry and Schaub (2011), chapter 6.
"""

import numpy as np
import pymc as pm

from ._capture_rates import _capture_history_likelihood
from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("Mt_model", posterior)
    history = np.asarray(data["y"], dtype="int64")
    coords = {
        "individual": np.arange(data["M"]),
        "capture_period": np.arange(data["T"]),
    }
    with pm.Model(coords=coords) as model:
        omega = pm.Uniform("omega", 0, 1)
        p = pm.Uniform("p", 0, 1, dims="capture_period")
        _capture_history_likelihood(history, omega, p=p)
        no_capture = (1 - p).prod()
        pm.Deterministic(
            "omega_nd", omega * no_capture / (omega * no_capture + 1 - omega)
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("Mt_model")
