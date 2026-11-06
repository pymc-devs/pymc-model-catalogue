"""The independent comparison rate retains its unconditioned Beta distribution.

Source: Lee and Wagenmakers (2013), binomial rate examples.
Predictive counts do not add free variables to the probability model.
"""

import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("Rate_4_model", posterior)
    with pm.Model() as model:
        theta = pm.Beta("theta", alpha=1, beta=1)
        pm.Beta("thetaprior", alpha=1, beta=1)
        pm.Binomial("k", n=data["n"], p=theta, observed=data["k"])
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("Rate_4_model")
