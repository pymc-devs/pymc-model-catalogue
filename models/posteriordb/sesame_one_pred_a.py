"""Native Gaussian regression for Sesame Street viewing and encouragement.

Source: Gelman and Hill (2006), ARM.
"""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("sesame_one_pred_a", posterior)
    response = np.asarray(data["watched"], dtype=float)
    encouraged = np.asarray(data["encouraged"], dtype=float)
    design = np.column_stack(
        (
            np.ones(len(response)),
            encouraged,
        )
    )
    coords = {
        "obs_id": np.arange(len(response)),
        "coefficient": ["intercept", "encouraged"],
    }
    initial_beta = np.zeros(design.shape[1])
    initial_beta[0] = response.mean()
    with pm.Model(coords=coords) as model:
        x = pm.Data("design", design, dims=("obs_id", "coefficient"))
        beta = pm.Flat("beta", dims="coefficient", initval=initial_beta)
        sigma = pm.HalfFlat("sigma", initval=max(float(response.std()), 1.0))
        pm.Normal(
            "watched",
            mu=pm.math.dot(x, beta),
            sigma=sigma,
            observed=response,
            dims="obs_id",
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("sesame_one_pred_a")
