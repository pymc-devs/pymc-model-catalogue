"""One success rate with a uniform Beta distribution.

Source: Lee and Wagenmakers (2013), binomial rate examples.
"""

import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("Rate_1_model", posterior)
    with pm.Model() as model:
        theta = pm.Beta("theta", alpha=1, beta=1)
        pm.Binomial("k", n=data["n"], p=theta, observed=data["k"])
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("Rate_1_model")
