"""Native generalized partial-credit model with latent regression from edstan."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ._data import load_data
from ._item_response import _item_response_coords, _item_response_latent_regression


def build_model(posterior: str | None = None):
    data = load_data("gpcm_latent_reg_irt", posterior)
    item = np.asarray(data["ii"], dtype="int64") - 1
    person = np.asarray(data["jj"], dtype="int64") - 1
    response = np.asarray(data["y"], dtype="int64")
    steps = np.zeros(data["I"], dtype="int64")
    np.maximum.at(steps, item, response)
    offsets = np.concatenate(([0], np.cumsum(steps[:-1])))
    n_steps = int(steps.sum())
    max_steps = int(steps.max())
    category = np.arange(max_steps + 1)
    threshold_index = np.minimum(
        offsets[:, None] + np.arange(max_steps)[None, :],
        n_steps - 1,
    )
    coords = _item_response_coords(data)
    coords["item_step"] = np.arange(n_steps)
    with pm.Model(coords=coords) as model:
        theta, alpha = _item_response_latent_regression(data)
        beta = pm.ZeroSumNormal("beta", sigma=3, dims="item_step")
        threshold_sum = pt.concatenate(
            [
                pt.zeros((data["I"], 1)),
                pt.cumsum(beta[threshold_index], axis=1),
            ],
            axis=1,
        )
        # Category logits sum the adjacent-category log odds.
        logits = (alpha[item] * theta[person])[:, None] * category[
            None, :
        ] - threshold_sum[item]
        logits = pt.where(category[None, :] <= steps[item, None], logits, -np.inf)
        pm.Categorical("y", logit_p=logits, observed=response, dims="response")
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("gpcm_latent_reg_irt")
