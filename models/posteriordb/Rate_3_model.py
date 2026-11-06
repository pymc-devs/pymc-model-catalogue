"""Two count observations share one success rate.

Source: Lee and Wagenmakers (2013), binomial rate examples.
"""

import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("Rate_3_model", posterior)
    with pm.Model() as model:
        theta = pm.Beta("theta", alpha=1, beta=1)
        pm.Binomial("k1", n=data["n1"], p=theta, observed=data["k1"])
        pm.Binomial("k2", n=data["n2"], p=theta, observed=data["k2"])
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("Rate_3_model")
