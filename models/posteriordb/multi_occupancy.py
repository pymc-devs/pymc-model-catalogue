"""Native multiple-species occupancy model from Dorazio and Royle.

The likelihood marginalizes site occupancy and species availability exactly.
Species effects use the centered bivariate Gaussian law.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ._data import load_data


def build_model(posterior: str | None = None):
    data = load_data("multi_occupancy", posterior)
    detected = np.asarray(data["X"], dtype="int64")
    n_observed = data["n"]
    n_species = data["S"]
    coords = {
        "species": np.arange(1, n_species + 1),
        "observed_species": np.arange(1, n_observed + 1),
        "site": np.arange(1, data["J"] + 1),
        "species_parameter": ["occupancy", "detection"],
    }
    with pm.Model(coords=coords) as model:
        alpha = pm.Cauchy("alpha", alpha=0, beta=2.5)
        beta = pm.Cauchy("beta", alpha=0, beta=2.5)
        omega = pm.Beta("Omega", alpha=2, beta=2)
        sigma = pm.HalfCauchy(
            "sigma_uv", beta=2.5, dims="species_parameter", initval=np.ones(2)
        )
        correlation_unit = pm.Beta("correlation_unit", alpha=2, beta=2)
        rho = pm.Deterministic("rho_uv", 2 * correlation_unit - 1)
        chol = pt.stack(
            [
                pt.stack([sigma[0], pt.zeros_like(rho)]),
                pt.stack([sigma[1] * rho, sigma[1] * pt.sqrt(1 - rho**2)]),
            ]
        )
        uv = pm.MvNormal(
            "uv",
            mu=np.zeros(2),
            chol=chol,
            dims=("species", "species_parameter"),
            initval=np.zeros((n_species, 2)),
        )
        occupancy = pm.math.sigmoid(alpha + uv[:, 0])
        detection = pm.math.sigmoid(beta + uv[:, 1])
        pm.Bernoulli(
            "available",
            p=omega,
            observed=np.ones(n_observed, dtype="int64"),
            dims="observed_species",
        )
        # A zero-inflated Binomial expresses detection conditional on species availability.
        pm.ZeroInflatedBinomial(
            "X",
            psi=occupancy[:n_observed, None],
            n=data["K"],
            p=detection[:n_observed, None],
            observed=detected,
            dims=("observed_species", "site"),
        )
        zero_site_logp = pm.logp(
            pm.ZeroInflatedBinomial.dist(
                psi=occupancy[n_observed:],
                n=data["K"],
                p=detection[n_observed:],
            ),
            0,
        )
        pm.Potential(
            "never_observed",
            pt.logaddexp(
                pt.log1p(-omega),
                pt.log(omega) + data["J"] * zero_site_logp,
            ).sum(),
        )
        pm.Deterministic("E_N", n_species * omega)
    return model, model.initial_point()


if __name__ == "__main__":
    from ._benchmark import main

    main("multi_occupancy")
