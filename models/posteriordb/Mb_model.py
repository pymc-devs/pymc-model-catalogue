"""Detection depends on capture at the previous occasion.

Source: Kéry and Schaub (2011), chapter 6.
"""

import numpy as np
import pymc as pm

from ._capture_rates import _capture_history_likelihood
from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("Mb_model", posterior)
    history = np.asarray(data["y"], dtype="int64")
    previous_capture = np.concatenate(
        [np.zeros((data["M"], 1), dtype="int64"), history[:, :-1]], axis=1
    )
    coords = {
        "individual": np.arange(data["M"]),
        "capture_period": np.arange(data["T"]),
    }
    with pm.Model(coords=coords) as model:
        omega = pm.Uniform("omega", 0, 1)
        p = pm.Uniform("p", 0, 1)
        c = pm.Uniform("c", 0, 1)
        p_eff = pm.Deterministic(
            "p_eff",
            p + previous_capture * (c - p),
            dims=("individual", "capture_period"),
        )
        _capture_history_likelihood(history, omega, p=p_eff)
        pm.Deterministic("trap_response", c - p)
        no_capture = (1 - p) ** int(data["T"])
        pm.Deterministic(
            "omega_nd", omega * no_capture / (omega * no_capture + 1 - omega)
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("Mb_model")
