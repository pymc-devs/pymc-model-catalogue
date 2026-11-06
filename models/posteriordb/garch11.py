"""GARCH(1, 1) from the posteriordb garch11 probability model.

The coefficient pair has a flat joint density on its triangular support.
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4.
"""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from ._data import load_data


def _garch11_variance_step(previous_y, previous_variance, mu, alpha0, alpha1, beta1):
    return alpha0 + alpha1 * pt.square(previous_y - mu) + beta1 * previous_variance


def build_model(posterior: str | None = None):
    data = load_data("garch11", posterior)
    y = np.asarray(data["y"], dtype=float)
    with pm.Model(coords={"time": np.arange(int(data["T"]))}) as model:
        mu = pm.Flat("mu", initval=float(y.mean()))
        alpha0 = pm.HalfFlat("alpha0", initval=1)
        alpha1 = pm.Uniform("alpha1", lower=0, upper=1, initval=0.2)
        beta1 = pm.Uniform("beta1", lower=0, upper=1 - alpha1, initval=0.5)
        # Remove the conditional uniform normalizer to retain the flat joint density.
        pm.Potential("coefficient_density", pt.log1p(-alpha1))
        observations = pm.Data("observations", y, dims="time")
        first_variance = pt.as_tensor_variable(np.float64(data["sigma1"]) ** 2)
        if len(y) > 1:
            later_variance, _ = pytensor.scan(
                fn=_garch11_variance_step,
                sequences=observations[:-1],
                outputs_info=first_variance,
                non_sequences=[mu, alpha0, alpha1, beta1],
                strict=True,
                name="garch11_variance",
            )
            variance = pt.concatenate([first_variance[None], later_variance])
        else:
            variance = first_variance[None]
        pm.Normal("y", mu=mu, sigma=pt.sqrt(variance), observed=y, dims="time")
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("garch11")
