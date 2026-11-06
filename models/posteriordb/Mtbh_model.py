"""Detection has additive occasion, behavior, and individual effects on log odds.

Source: Kéry and Schaub (2011), chapter 6.
"""

import numpy as np
import pymc as pm

from ._capture_rates import _capture_history_likelihood
from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("Mtbh_model", posterior)
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
        mean_p = pm.Uniform("mean_p", 0, 1, dims="capture_period")
        gamma = pm.Normal("gamma", 0, 10)
        sigma = pm.Uniform("sigma", 0, 3)
        eps_raw = pm.Normal("eps_raw", 0, 1, dims="individual")
        eps = pm.Deterministic("eps", sigma * eps_raw, dims="individual")
        alpha = pm.Deterministic("alpha", pm.math.logit(mean_p), dims="capture_period")
        logit_p = pm.Deterministic(
            "logit_p",
            alpha[None, :] + eps[:, None] + gamma * previous_capture,
            dims=("individual", "capture_period"),
        )
        _capture_history_likelihood(history, omega, logit_p=logit_p)
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("Mtbh_model")
