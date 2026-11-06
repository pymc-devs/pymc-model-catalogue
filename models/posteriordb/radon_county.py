"""Native county radon hierarchy from posteriordb, Bales et al. (2019).

The county model uses bounded uniform scale distributions.
"""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    """Build the selected posterior from its published data."""
    data = load_data("radon_county", posterior)
    coords = {"obs_id": np.arange(data["N"]), "county": np.arange(1, data["J"] + 1)}
    with pm.Model(coords=coords) as model:
        county = pm.Data(
            "county_idx", np.asarray(data["county"], dtype="int64") - 1, dims="obs_id"
        )
        mu_a = pm.Normal("mu_a", mu=0, sigma=1)
        sigma_a = pm.Uniform("sigma_a", lower=0, upper=100, initval=1)
        sigma_y = pm.Uniform("sigma_y", lower=0, upper=100, initval=1)
        a = pm.Normal(
            "a", mu=mu_a, sigma=sigma_a, dims="county", initval=np.zeros(data["J"])
        )
        pm.Normal(
            "y",
            mu=a[county],
            sigma=sigma_y,
            observed=np.asarray(data["y"], dtype=float),
            dims="obs_id",
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("radon_county")
