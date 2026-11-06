"""Native centered growth effects from posteriordb rats_model."""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    """Return the native model and its initial point for the selected pair."""
    data = load_data("rats_model", posterior)
    coords = {
        "rat_id": np.arange(1, data["N"] + 1),
        "observation": np.arange(data["Npts"]),
    }
    with pm.Model(coords=coords) as model:
        rat = pm.Data(
            "rat_index", np.asarray(data["rat"], dtype="int64") - 1, dims="observation"
        )
        centered_x = pm.Data(
            "centered_x",
            np.asarray(data["x"], dtype=float) - data["xbar"],
            dims="observation",
        )
        mu_alpha = pm.Normal("mu_alpha", 0, 100, initval=200.0)
        mu_beta = pm.Normal("mu_beta", 0, 100, initval=6.0)
        # These improper scale densities stay flat on positive scales.
        sigma_y = pm.HalfFlat("sigma_y", initval=10.0)
        sigma_alpha = pm.HalfFlat("sigma_alpha", initval=20.0)
        sigma_beta = pm.HalfFlat("sigma_beta", initval=1.0)
        alpha = pm.Normal(
            "alpha",
            mu_alpha,
            sigma_alpha,
            dims="rat_id",
            initval=np.full(data["N"], 200.0),
        )
        beta = pm.Normal(
            "beta", mu_beta, sigma_beta, dims="rat_id", initval=np.full(data["N"], 6.0)
        )
        pm.Normal(
            "y",
            mu=alpha[rat] + beta[rat] * centered_x,
            sigma=sigma_y,
            observed=data["y"],
            dims="observation",
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("rats_model")
