"""Native two-parameter logistic model with latent regression from edstan."""

import numpy as np
import pymc as pm

from ._data import load_data
from ._item_response import _item_response_coords, _item_response_latent_regression


def build_model(posterior: str | None = None):
    data = load_data("2pl_latent_reg_irt", posterior)
    item = np.asarray(data["ii"], dtype="int64") - 1
    person = np.asarray(data["jj"], dtype="int64") - 1
    with pm.Model(coords=_item_response_coords(data)) as model:
        theta, alpha = _item_response_latent_regression(data)
        # Zero-sum Gaussian effects preserve the constrained Gaussian law.
        beta = pm.ZeroSumNormal("beta", sigma=3, dims="item")
        pm.Bernoulli(
            "y",
            logit_p=alpha[item] * theta[person] - beta[item],
            observed=np.asarray(data["y"], dtype="int64"),
            dims="response",
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("2pl_latent_reg_irt")
