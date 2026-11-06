"""Native generalized rating-scale model with latent regression from edstan."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ._data import load_data
from ._item_response import _item_response_coords, _item_response_latent_regression


def build_model(posterior: str | None = None):
    data = load_data("grsm_latent_reg_irt", posterior)
    item = np.asarray(data["ii"], dtype="int64") - 1
    person = np.asarray(data["jj"], dtype="int64") - 1
    response = np.asarray(data["y"], dtype="int64")
    n_steps = int(response.max())
    category = np.arange(n_steps + 1)
    coords = _item_response_coords(data)
    coords["step"] = np.arange(1, n_steps + 1)
    with pm.Model(coords=coords) as model:
        theta, alpha = _item_response_latent_regression(data)
        beta = pm.ZeroSumNormal("beta", sigma=3, dims="item")
        kappa = pm.ZeroSumNormal("kappa", sigma=3, dims="step")
        threshold_sum = pt.concatenate([pt.zeros(1), pt.cumsum(kappa)])
        # Category logits sum the adjacent-category log odds.
        logits = (alpha[item] * theta[person] - beta[item])[:, None] * category[
            None, :
        ] - threshold_sum[None, :]
        pm.Categorical("y", logit_p=logits, observed=response, dims="response")
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("grsm_latent_reg_irt")
