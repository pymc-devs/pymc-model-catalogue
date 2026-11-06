"""Individual detection uses noncentered Gaussian effects on log odds.

Source: Kéry and Schaub (2011), chapter 6.
"""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("Mh_model", posterior)
    counts = np.asarray(data["y"], dtype="int64")
    with pm.Model(coords={"individual": np.arange(data["M"])}) as model:
        omega = pm.Uniform("omega", 0, 1)
        mean_p = pm.Uniform("mean_p", 0, 1)
        sigma = pm.Uniform("sigma", 0, 5)
        eps_raw = pm.Normal("eps_raw", 0, 1, dims="individual")
        eps = pm.Deterministic(
            "eps", pm.math.logit(mean_p) + sigma * eps_raw, dims="individual"
        )
        pm.ZeroInflatedBinomial(
            "captures",
            psi=omega,
            n=data["T"],
            p=pm.math.sigmoid(eps),
            observed=counts,
            dims="individual",
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("Mh_model")
