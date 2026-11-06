"""Native Gaussian model for log mesquite weight and log predictors.

Source: Gelman and Hill (2006), ARM.

The likelihood uses the transformed response, without a response Jacobian.
"""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("logmesquite", posterior)
    response = np.log(np.asarray(data["weight"], dtype=float))
    log_diam1 = np.log(np.asarray(data["diam1"], dtype=float))
    log_diam2 = np.log(np.asarray(data["diam2"], dtype=float))
    log_canopy_height = np.log(np.asarray(data["canopy_height"], dtype=float))
    log_total_height = np.log(np.asarray(data["total_height"], dtype=float))
    log_density = np.log(np.asarray(data["density"], dtype=float))
    group = np.asarray(data["group"], dtype=float)
    design = np.column_stack(
        (
            np.ones(len(response)),
            log_diam1,
            log_diam2,
            log_canopy_height,
            log_total_height,
            log_density,
            group,
        )
    )
    coords = {
        "obs_id": np.arange(len(response)),
        "coefficient": [
            "intercept",
            "log_diam1",
            "log_diam2",
            "log_canopy_height",
            "log_total_height",
            "log_density",
            "group",
        ],
    }
    initial_beta = np.zeros(design.shape[1])
    initial_beta[0] = response.mean()
    with pm.Model(coords=coords) as model:
        x = pm.Data("design", design, dims=("obs_id", "coefficient"))
        beta = pm.Flat("beta", dims="coefficient", initval=initial_beta)
        sigma = pm.HalfFlat("sigma", initval=max(float(response.std()), 1.0))
        pm.Normal(
            "log_weight",
            mu=pm.math.dot(x, beta),
            sigma=sigma,
            observed=response,
            dims="obs_id",
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("logmesquite")
