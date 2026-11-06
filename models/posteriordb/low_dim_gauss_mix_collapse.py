"""Two-component Gaussian mixture with unrestricted means.

Source: posteriordb low_dim_gauss_mix_collapse model metadata at revision 91027f9.
This named variant keeps both label permutations and does not order the means.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ._data import load_data


def build_model(posterior: str | None = None):
    """Return the native model and its initial point for the selected posterior."""
    data = load_data("low_dim_gauss_mix_collapse", posterior)
    y = np.asarray(data["y"], dtype=float)
    coords = {"component": np.arange(2), "observation": np.arange(data["N"])}
    with pm.Model(coords=coords) as model:
        mu = pm.Normal(
            "mu", mu=0, sigma=2, dims="component", initval=np.array([-1.0, 1.0])
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

    main("low_dim_gauss_mix_collapse")
