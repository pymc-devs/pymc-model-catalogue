"""Two-component Normal mixture with known unit standard deviations.

Source: posteriordb normal_mixture model metadata at revision 91027f9.
The likelihood uses only y. Other dataset fields describe the data generator.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ._data import load_data


def build_model(posterior: str | None = None):
    """Return the native model and its initial point for the selected posterior."""
    data = load_data("normal_mixture", posterior)
    y = np.asarray(data["y"], dtype=float)
    coords = {"component": np.arange(2), "observation": np.arange(data["N"])}
    with pm.Model(coords=coords) as model:
        theta = pm.Uniform("theta", lower=0, upper=1, initval=0.5)
        mu = pm.Normal(
            "mu", mu=0, sigma=10, dims="component", initval=np.array([-1.0, 1.0])
        )
        weights = pt.stack([theta, 1 - theta])
        pm.NormalMixture(
            "y", w=weights, mu=mu, sigma=1.0, observed=y, dims="observation"
        )
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("normal_mixture")
