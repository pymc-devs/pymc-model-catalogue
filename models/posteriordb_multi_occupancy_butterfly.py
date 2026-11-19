"""
Model: Multiple Species-Site Occupancy Model
Source: posteriordb/posterior_database/models/stan/multi_occupancy.stan
Authors: Dorazio, Robert M and Royle, J Andrew and Söderström, Bo and Glimsk{"a}r, Anders (cited
    work)
Description: Hierarchical species-site occupancy model for species richness, accounting for
    imperfect detection.

Stan comparison:
- A: Stan = -2216.558948724705; PyMC = -2215.172654428967.
- B: Stan = -2567.8440496981566; PyMC = -2566.457755402418.
- Raw delta: Stan = -351.2851009734518; PyMC = -351.28510097345134.
- Raw comparison: MATCH; residual = -4.547473508864641e-13.
- Tolerance: 4.512851009734518e-06
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_butterfly.json.zip"
        ) as archive,
        archive.open("butterfly.json") as stream,
    ):
        data = json.load(stream)
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
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
