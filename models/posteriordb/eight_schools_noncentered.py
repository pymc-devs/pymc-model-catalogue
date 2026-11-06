"""Native noncentered school effects from posteriordb eight_schools_noncentered."""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    """Return the native model and its initial point for the selected pair."""
    data = load_data("eight_schools_noncentered", posterior)
    with pm.Model(coords={"school": np.arange(1, data["J"] + 1)}) as model:
        mu = pm.Normal("mu", 0, 5)
        tau = pm.HalfCauchy("tau", 5)
        theta_trans = pm.Normal("theta_trans", 0, 1, dims="school")
        theta = pm.Deterministic("theta", mu + tau * theta_trans, dims="school")
        pm.Normal(
            "y",
            theta,
            sigma=np.asarray(data["sigma"], dtype=float),
            observed=data["y"],
            dims="school",
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("eight_schools_noncentered")
