"""Native crossed Gaussian effects for pilot groups and scenarios.

Source: Gelman and Hill (2006).
The group and scenario effects retain their centered parameterization.
"""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("pilots", posterior)
    response = np.asarray(data["y"], dtype=float)
    group_id = np.asarray(data["group_id"], dtype=int) - 1
    scenario_id = np.asarray(data["scenario_id"], dtype=int) - 1
    coords = {
        "obs_id": np.arange(len(response)),
        "group": np.arange(1, int(data["n_groups"]) + 1),
        "scenario": np.arange(1, int(data["n_scenarios"]) + 1),
    }
    with pm.Model(coords=coords) as model:
        group_index = pm.Data("group_id", group_id, dims="obs_id")
        scenario_index = pm.Data("scenario_id", scenario_id, dims="obs_id")
        mu_a = pm.Normal("mu_a", mu=0, sigma=1)
        mu_b = pm.Normal("mu_b", mu=0, sigma=1)
        sigma_a = pm.Uniform("sigma_a", lower=0, upper=100)
        sigma_b = pm.Uniform("sigma_b", lower=0, upper=100)
        sigma_y = pm.Uniform("sigma_y", lower=0, upper=100)
        a = pm.Normal("a", mu=10 * mu_a, sigma=sigma_a, dims="group")
        b = pm.Normal("b", mu=10 * mu_b, sigma=sigma_b, dims="scenario")
        pm.Normal(
            "y",
            mu=a[group_index] + b[scenario_index],
            sigma=sigma_y,
            observed=response,
            dims="obs_id",
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("pilots")
