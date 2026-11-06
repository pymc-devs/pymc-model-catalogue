"""Hierarchical loss development from the posteriordb losscurve_sislob model.

The data select Weibull or log-logistic cumulative growth.
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("losscurve_sislob", posterior)
    cohort_index = np.asarray(data["cohort_id"], dtype=int) - 1
    time_index = np.asarray(data["t_idx"], dtype=int) - 1
    coords = {
        "cohort": np.arange(1, int(data["n_cohort"]) + 1),
        "time": np.asarray(data["t_value"], dtype=float),
        "observation": np.arange(int(data["n_data"])),
    }
    with pm.Model(coords=coords) as model:
        omega = pm.LogNormal("omega", mu=0, sigma=0.5)
        theta = pm.LogNormal("theta", mu=0, sigma=0.5)
        mu_LR = pm.Normal("mu_LR", mu=0, sigma=0.5)
        sd_LR = pm.LogNormal("sd_LR", mu=0, sigma=0.5)
        LR = pm.LogNormal("LR", mu=mu_LR, sigma=sd_LR, dims="cohort")
        loss_sd = pm.LogNormal("loss_sd", mu=0, sigma=0.7)
        time = pm.Data("t_value", np.asarray(data["t_value"], dtype=float), dims="time")
        premium = pm.Data(
            "premium", np.asarray(data["premium"], dtype=float), dims="cohort"
        )
        if int(data["growthmodel_id"]) == 1:
            growth = -pt.expm1(-pt.power(time / theta, omega))
        else:
            growth = pt.sigmoid(omega * (pt.log(time) - pt.log(theta)))
        gf = pm.Deterministic("gf", growth, dims="time")
        expected = LR[cohort_index] * premium[cohort_index] * gf[time_index]
        pm.Normal(
            "loss",
            mu=expected,
            sigma=loss_sd * premium[cohort_index],
            observed=np.asarray(data["loss"], dtype=float),
            dims="observation",
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("losscurve_sislob")
