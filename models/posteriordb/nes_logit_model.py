"""Native logistic regression for National Election Study votes.

Source: Gelman and Hill (2006), ARM.
"""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("nes_logit_model", posterior)
    income = np.asarray(data["income"], dtype=float)
    vote = np.asarray(data["vote"], dtype=int)
    coords = {"obs_id": np.arange(len(vote)), "coefficient": ["income"]}
    with pm.Model(coords=coords) as model:
        x = pm.Data("income", income, dims="obs_id")
        alpha = pm.Flat("alpha")
        beta = pm.Flat("beta", dims="coefficient")
        pm.Bernoulli("vote", logit_p=alpha + beta[0] * x, observed=vote, dims="obs_id")
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("nes_logit_model")
