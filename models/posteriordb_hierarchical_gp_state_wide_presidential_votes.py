"""
Model: Hierarchical Gaussian Process
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/hierarchical_gp.stan
Authors: Not specified in upstream model metadata.
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/hierarchical_gp.py
Upstream catalogue contributor: Keane Nguyen
Description: Hierarchical Gaussian Process

Data: state_wide_presidential_votes (upstream posterior: state_wide_presidential_votes-hierarchical_gp)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/state_wide_presidential_votes.json.zip
    Archive SHA-256: fe67ff6de8983a1eedf4b4e6c10ceae8b13c3fd265d3cd498a6c7b344a777dfe
    Local archive: data/posteriordb_state_wide_presidential_votes.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/state_wide_presidential_votes-hierarchical_gp.json
Reference URL: https://mc-stan.org/events/stancon2017-notebooks/stancon2017-trangucci-hierarchical-gps.pdf

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Hierarchical election Gaussian process from posteriordb and Trangucci (2017).

A shared variance allocation controls independent year, state, region, and GP effects.
Each temporal kernel sums long-scale and short-scale squared-exponential components.

Stan comparison:
- Not recorded.

Benchmark results:
- Not measured.
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
            Path(__file__).parent
            / "data"
            / "posteriordb_state_wide_presidential_votes.json.zip"
        ) as archive,
        archive.open("state_wide_presidential_votes.json") as stream,
    ):
        data = json.load(stream)
    n_years = data["N_years"]
    years = np.arange(1, n_years + 1, dtype=float)[:, None]
    state_region = np.asarray(data["state_region_ind"], dtype=int) - 1
    variance_names = (
        ["year", "region"]
        + [f"state_region_{i}" for i in range(1, 11)]
        + ["region_long", "state_long", "region_short", "state_short", "error"]
    )
    coords = {
        "observation": np.arange(data["N"]),
        "year": np.arange(1, n_years + 1),
        "observed_year": np.arange(1, data["N_years_obs"] + 1),
        "state": np.arange(1, data["N_states"] + 1),
        "region": np.arange(1, data["N_regions"] + 1),
        "variance_component": variance_names,
    }
    with pm.Model(coords=coords) as model:
        state_index = pm.Data(
            "state_ind",
            np.asarray(data["state_ind"], dtype=int) - 1,
            dims="observation",
        )
        region_index = pm.Data(
            "region_ind",
            np.asarray(data["region_ind"], dtype=int) - 1,
            dims="observation",
        )
        year_index = pm.Data(
            "year_ind", np.asarray(data["year_ind"], dtype=int) - 1, dims="observation"
        )
        mu = pm.Normal("mu", mu=0.5, sigma=0.5)
        total_variance = pm.Gamma("tot_var", alpha=3, beta=3)
        proportion = pm.Dirichlet(
            "prop_var", a=np.full(17, 2.0), dims="variance_component"
        )
        component_sd = pm.Deterministic(
            "component_sd",
            pt.sqrt(17 * total_variance * proportion),
            dims="variance_component",
        )
        year_std = pm.Normal("year_std", 0, 1, dims="observed_year")
        state_std = pm.Normal("state_std", 0, 1, dims="state")
        region_std = pm.Normal("region_std", 0, 1, dims="region")
        year_effect = pm.Deterministic(
            "year_re", component_sd[0] * year_std, dims="observed_year"
        )
        region_effect = pm.Deterministic(
            "region_re", component_sd[1] * region_std, dims="region"
        )
        state_effect = pm.Deterministic(
            "state_re", component_sd[2:12][state_region] * state_std, dims="state"
        )
        region_long = pm.Weibull("length_GP_region_long", alpha=30, beta=8, initval=8)
        state_long = pm.Weibull("length_GP_state_long", alpha=30, beta=8, initval=8)
        region_short = pm.Weibull("length_GP_region_short", alpha=30, beta=3, initval=3)
        state_short = pm.Weibull("length_GP_state_short", alpha=30, beta=3, initval=3)
        region_kernel = component_sd[12] ** 2 * pm.gp.cov.ExpQuad(
            1, ls=region_long
        ) + component_sd[14] ** 2 * pm.gp.cov.ExpQuad(1, ls=region_short)
        state_kernel = component_sd[13] ** 2 * pm.gp.cov.ExpQuad(
            1, ls=state_long
        ) + component_sd[15] ** 2 * pm.gp.cov.ExpQuad(1, ls=state_short)
        region_chol = pt.linalg.cholesky(region_kernel(years) + 1e-6 * pt.eye(n_years))
        state_chol = pt.linalg.cholesky(state_kernel(years) + 1e-6 * pt.eye(n_years))
        region_gp_std = pm.Normal("GP_region_std", 0, 1, dims=("year", "region"))
        state_gp_std = pm.Normal("GP_state_std", 0, 1, dims=("year", "state"))
        region_gp = pm.Deterministic(
            "GP_region", region_chol @ region_gp_std, dims=("year", "region")
        )
        state_gp = pm.Deterministic(
            "GP_state", state_chol @ state_gp_std, dims=("year", "state")
        )
        observation_mu = (
            mu
            + year_effect[year_index]
            + state_effect[state_index]
            + region_effect[region_index]
            + region_gp[year_index, region_index]
            + state_gp[year_index, state_index]
        )
        pm.Normal(
            "y",
            mu=observation_mu,
            sigma=component_sd[16],
            observed=np.asarray(data["y"]),
            dims="observation",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
