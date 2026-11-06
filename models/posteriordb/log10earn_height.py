"""Native Gaussian model for base-ten log earnings and height.

Source: Gelman and Hill (2006), ARM.

The likelihood uses the transformed response, without a response Jacobian.
"""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("log10earn_height", posterior)
    response = np.log10(np.asarray(data["earn"], dtype=float))
    height = np.asarray(data["height"], dtype=float)
    design = np.column_stack(
        (
            np.ones(len(response)),
            height,
        )
    )
    coords = {
        "obs_id": np.arange(len(response)),
        "coefficient": ["intercept", "height"],
    }
    initial_beta = np.zeros(design.shape[1])
    initial_beta[0] = response.mean()
    with pm.Model(coords=coords) as model:
        x = pm.Data("design", design, dims=("obs_id", "coefficient"))
        beta = pm.Flat("beta", dims="coefficient", initval=initial_beta)
        sigma = pm.HalfFlat("sigma", initval=max(float(response.std()), 1.0))
        pm.Normal(
            "log10_earn",
            mu=pm.math.dot(x, beta),
            sigma=sigma,
            observed=response,
            dims="obs_id",
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("log10earn_height")
