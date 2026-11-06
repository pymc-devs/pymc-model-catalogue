"""Native LSAT item-response model from the BUGS example.

Cumulative pattern counts expand to individual student responses.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("lsat_model", posterior)
    cumulative = np.asarray(data["culm"], dtype="int64")
    counts = np.diff(np.concatenate(([0], cumulative)))
    responses = np.repeat(np.asarray(data["response"], dtype="int64"), counts, axis=0)
    coords = {
        "item": np.arange(1, data["T"] + 1),
        "person": np.arange(1, data["N"] + 1),
    }
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", mu=0, sigma=100, dims="item")
        theta = pm.Normal("theta", mu=0, sigma=1, dims="person")
        beta = pm.HalfNormal("beta", sigma=100, initval=1)
        pm.Bernoulli(
            "response",
            logit_p=beta * theta[:, None] - alpha[None, :],
            observed=responses,
            dims=("person", "item"),
        )
        pm.Deterministic("a", alpha - pt.mean(alpha), dims="item")
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("lsat_model")
