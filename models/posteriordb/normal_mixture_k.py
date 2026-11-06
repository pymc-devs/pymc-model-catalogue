"""K-component Normal mixture with unknown means, weights, and standard deviations.

Source: posteriordb normal_mixture_k model metadata at revision 91027f9.
A uniform simplex density gives Dirichlet weights with unit concentrations.
The likelihood uses only y. Other dataset fields describe the data generator.
"""

import numpy as np
import pymc as pm

from ._data import load_data


def build_model(posterior: str | None = None):
    """Return the native model and its initial point for the selected posterior."""
    data = load_data("normal_mixture_k", posterior)
    y = np.asarray(data["y"], dtype=float)
    n_components = int(data["K"])
    coords = {
        "component": np.arange(n_components),
        "observation": np.arange(data["N"]),
    }
    with pm.Model(coords=coords) as model:
        theta = pm.Dirichlet(
            "theta",
            a=np.ones(n_components),
            dims="component",
            initval=np.full(n_components, 1.0 / n_components),
        )
        mu = pm.Normal(
            "mu",
            mu=0,
            sigma=10,
            dims="component",
            initval=np.linspace(-2.0, 2.0, n_components),
        )
        sigma = pm.Uniform(
            "sigma",
            lower=0,
            upper=10,
            dims="component",
            initval=np.full(n_components, 5.0),
        )
        pm.NormalMixture(
            "y", w=theta, mu=mu, sigma=sigma, observed=y, dims="observation"
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("normal_mixture_k")
