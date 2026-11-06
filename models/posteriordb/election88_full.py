"""Native crossed logistic effects from posteriordb election88_full."""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    """Return the native model and its initial point for the selected pair."""
    data = load_data("election88_full", posterior)
    groups = (
        ("age", "a"),
        ("edu", "b"),
        ("age_edu", "c"),
        ("state", "d"),
        ("region_full", "e"),
    )
    coords = {name: np.arange(1, data[f"n_{name}"] + 1) for name, _ in groups}
    coords.update(
        observation=np.arange(data["N"]),
        coefficient=["intercept", "black", "female", "previous_vote", "female_black"],
    )
    black = np.asarray(data["black"], dtype=float)
    female = np.asarray(data["female"], dtype=float)
    predictors = np.column_stack(
        (np.ones(data["N"]), black, female, data["v_prev_full"], female * black)
    )
    with pm.Model(coords=coords) as model:
        x = pm.Data("X", predictors, dims=("observation", "coefficient"))
        beta = pm.Normal("beta", 0, 100, dims="coefficient")
        logit_p = x @ beta
        for group, effect_name in groups:
            group_index = pm.Data(
                f"{group}_index",
                np.asarray(data[group], dtype="int64") - 1,
                dims="observation",
            )
            sigma = pm.Uniform(f"sigma_{effect_name}", 0, 100)
            effect = pm.Normal(effect_name, 0, sigma, dims=group)
            logit_p = logit_p + effect[group_index]
        pm.Bernoulli("y", logit_p=logit_p, observed=data["y"], dims="observation")
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("election88_full")
