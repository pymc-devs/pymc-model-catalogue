"""Native Gaussian regression for National Election Study party identification.

Source: Gelman and Hill (2006), ARM.
"""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("nes", posterior)
    response = np.asarray(data["partyid7"], dtype=float)
    real_ideo = np.asarray(data["real_ideo"], dtype=float)
    race_adj = np.asarray(data["race_adj"], dtype=float)
    age = np.asarray(data["age_discrete"], dtype=int)
    educ1 = np.asarray(data["educ1"], dtype=float)
    gender = np.asarray(data["gender"], dtype=float)
    income = np.asarray(data["income"], dtype=float)
    design = np.column_stack(
        (
            np.ones(len(response)),
            real_ideo,
            race_adj,
            age == 2,
            age == 3,
            age == 4,
            educ1,
            gender,
            income,
        )
    )
    coords = {
        "obs_id": np.arange(len(response)),
        "coefficient": [
            "intercept",
            "real_ideo",
            "race_adj",
            "age30_44",
            "age45_64",
            "age65up",
            "educ1",
            "gender",
            "income",
        ],
    }
    initial_beta = np.zeros(design.shape[1])
    initial_beta[0] = response.mean()
    with pm.Model(coords=coords) as model:
        x = pm.Data("design", design, dims=("obs_id", "coefficient"))
        beta = pm.Flat("beta", dims="coefficient", initval=initial_beta)
        sigma = pm.HalfFlat("sigma", initval=max(float(response.std()), 1.0))
        pm.Normal(
            "partyid7",
            mu=pm.math.dot(x, beta),
            sigma=sigma,
            observed=response,
            dims="obs_id",
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("nes")
