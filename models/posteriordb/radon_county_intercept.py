"""Native radon model from the posteriordb radon case study.

Source: https://mc-stan.org/users/documentation/case-studies/radon.html
"""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    """Build the selected posterior from its published data."""
    data = load_data("radon_county_intercept", posterior)
    coords = {"obs_id": np.arange(data["N"]), "county": np.arange(1, data["J"] + 1)}
    with pm.Model(coords=coords) as model:
        county = pm.Data(
            "county_idx",
            np.asarray(data["county_idx"], dtype="int64") - 1,
            dims="obs_id",
        )
        floor = pm.Data(
            "floor_measure",
            np.asarray(data["floor_measure"], dtype=float),
            dims="obs_id",
        )
        alpha = pm.Normal("alpha", mu=0, sigma=10, dims="county")
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma_y = pm.HalfNormal("sigma_y", sigma=1, initval=1)
        pm.Normal(
            "log_radon",
            mu=alpha[county] + beta * floor,
            sigma=sigma_y,
            observed=np.asarray(data["log_radon"], dtype=float),
            dims="obs_id",
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("radon_county_intercept")
