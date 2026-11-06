"""Two independent success rates and their difference.

Source: Lee and Wagenmakers (2013), binomial rate examples.
"""

import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("Rate_2_model", posterior)
    with pm.Model() as model:
        theta1 = pm.Beta("theta1", alpha=1, beta=1)
        theta2 = pm.Beta("theta2", alpha=1, beta=1)
        pm.Binomial("k1", n=data["n1"], p=theta1, observed=data["k1"])
        pm.Binomial("k2", n=data["n2"], p=theta2, observed=data["k2"])
        pm.Deterministic("delta", theta1 - theta2)
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("Rate_2_model")
