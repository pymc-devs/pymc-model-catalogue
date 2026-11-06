"""Two-component Gaussian mixture with ordered means.

Source: posteriordb low_dim_gauss_mix model metadata at revision 91027f9.
The ordered transform keeps the source constraint on the component means.
Each observation selects one Normal component through the native marginalized likelihood.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ._data import load_data


def build_model(posterior: str | None = None):
    """Return the native model and its initial point for the selected posterior."""
    data = load_data("low_dim_gauss_mix", posterior)
    y = np.asarray(data["y"], dtype=float)
    coords = {
        "component": ["lower_mean", "upper_mean"],
        "observation": np.arange(data["N"]),
    }
    with pm.Model(coords=coords) as model:
        mu = pm.Normal(
            "mu",
            mu=0,
            sigma=2,
            dims="component",
            transform=pm.distributions.transforms.ordered,
            initval=np.array([-1.0, 1.0]),
        )
        sigma = pm.HalfNormal(
            "sigma", sigma=2, dims="component", initval=np.full(2, 2.0)
        )
        theta = pm.Beta("theta", alpha=5, beta=5, initval=0.5)
        weights = pt.stack([theta, 1 - theta])
        pm.NormalMixture(
            "y", w=weights, mu=mu, sigma=sigma, observed=y, dims="observation"
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("low_dim_gauss_mix")
