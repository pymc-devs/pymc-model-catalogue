"""Constant detection with exact population-inclusion marginalization.

Source: Kéry and Schaub (2011), chapter 6.
"""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("M0_model", posterior)
    counts = np.asarray(data["y"], dtype="int64").sum(axis=1)
    occasions = int(data["T"])
    with pm.Model(coords={"individual": np.arange(data["M"])}) as model:
        omega = pm.Uniform("omega", 0, 1)
        p = pm.Uniform("p", 0, 1)
        pm.ZeroInflatedBinomial(
            "captures",
            psi=omega,
            n=occasions,
            p=p,
            observed=counts,
            dims="individual",
        )
        no_capture = (1 - p) ** occasions
        pm.Deterministic(
            "omega_nd", omega * no_capture / (omega * no_capture + 1 - omega)
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("M0_model")
