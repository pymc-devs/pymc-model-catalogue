"""
Model: Epidemic model v3 of Flaxman et al (2020)
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/covid19imperial_v3.stan
Authors: Not specified in upstream model metadata.
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/covid19imperial_v3.py
Upstream catalogue contributor: Mans Magnusson
Description: A Bayesian model for estimating intervetion effects in the 2020 covid-19 pandemic.

Data: ecdc0401 (upstream posterior: ecdc0401-covid19imperial_v3)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/ecdc0401.json.zip
    Archive SHA-256: 68a300d04fbab1589ea42eba08cffb3592878320182c31fcb2ad4251b81597fe
    Local archive: data/posteriordb_ecdc0401.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/ecdc0401-covid19imperial_v3.json
Reference URL: https://github.com/ImperialCollegeLondon/covid19model

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Flaxman et al. (2020) intervention renewal model, version 3.

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
import pytensor
import pytensor.tensor as pt


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_ecdc0401.json.zip"
        ) as archive,
        archive.open("ecdc0401.json") as stream,
    ):
        data = json.load(stream)
    n_country = int(data["M"])
    n_day = int(data["N2"])
    n_seed = int(data["N0"])
    n_intervention = int(data["P"])
    design = np.asarray(data["X"], dtype=np.float64).transpose(1, 0, 2)
    population = np.asarray(data["pop"], dtype=np.float64)
    serial_interval = np.asarray(data["SI"], dtype=np.float64)
    death_delay = np.asarray(data["f"], dtype=np.float64)
    deaths = np.asarray(data["deaths"], dtype=np.int64)
    starts = np.asarray(data["EpidemicStart"], dtype=np.int64) - 1
    ends = np.asarray(data["N"], dtype=np.int64)

    lag = np.arange(n_day)[:, None] - np.arange(n_day)[None, :]
    past = lag > 0
    delay_index = np.maximum(lag - 1, 0)
    serial_weights = np.where(past, serial_interval[delay_index], 0.0)
    death_weights = np.where(past[:, :, None], death_delay[delay_index], 0.0)
    death_weights = death_weights.transpose(2, 0, 1)

    observed_day = np.concatenate(
        [np.arange(starts[c], ends[c]) for c in range(n_country)]
    )
    observed_country = np.repeat(np.arange(n_country), ends - starts)
    coords = {
        "country": np.arange(1, n_country + 1),
        "day": np.arange(1, n_day + 1),
        "infection_day": np.arange(1, n_day + 1),
        "intervention": np.arange(1, n_intervention + 1),
        "observation": np.arange(len(observed_day)),
    }

    with pm.Model(coords=coords) as model:
        x = pm.Data("interventions", design, dims=("day", "country", "intervention"))
        pop = pm.Data("population", population, dims="country")
        generation = pm.Data(
            "serial_interval_weights", serial_weights, dims=("day", "infection_day")
        )
        fatality_delay = pm.Data(
            "death_delay_weights",
            death_weights,
            dims=("country", "day", "infection_day"),
        )
        death_day = pm.Data("death_day", observed_day, dims="observation")
        death_country = pm.Data("death_country", observed_country, dims="observation")

        tau = pm.Exponential("tau", lam=0.03, initval=1.0 / 0.03)
        y = pm.Exponential(
            "y", lam=1.0 / tau, dims="country", initval=np.ones(n_country)
        )
        phi = pm.HalfNormal("phi", sigma=5.0, initval=5.0)
        kappa = pm.HalfNormal("kappa", sigma=0.5, initval=0.5)
        mu = pm.TruncatedNormal(
            "mu",
            mu=3.28,
            sigma=kappa,
            lower=0.0,
            dims="country",
            initval=np.ones(n_country),
        )
        # The log-CDF factor preserves the source density for positive mu and random kappa.
        pm.Potential(
            "mu_constraint_normalization",
            n_country * pm.logcdf(pm.Normal.dist(0.0, 1.0), 3.28 / kappa),
        )
        alpha_hier = pm.Gamma(
            "alpha_hier",
            alpha=0.1667,
            beta=1.0,
            dims="intervention",
            initval=np.full(n_intervention, 0.1667),
        )
        alpha = pm.Deterministic(
            "alpha", alpha_hier - np.log(1.05) / 6.0, dims="intervention"
        )
        ifr_noise = pm.TruncatedNormal(
            "ifr_noise",
            mu=1.0,
            sigma=0.1,
            lower=0.0,
            dims="country",
            initval=np.ones(n_country),
        )
        reproduction = pm.Deterministic(
            "Rt",
            mu[None, :] * pt.exp(-pt.sum(x * alpha, axis=2)),
            dims=("day", "country"),
        )
        seeds = pt.broadcast_to(y, (n_seed, n_country))
        initial_history = pt.concatenate(
            [seeds, pt.zeros((n_day - n_seed, n_country), dtype=y.dtype)], axis=0
        )

        def renewal_step(day, rt, lag_weights, history, cumulative, populations):
            # The cumulative state counts all infections before the current day.
            force = pt.sum(history * lag_weights[:, None], axis=0)
            incidence = rt * (1.0 - cumulative / populations) * force
            return pt.set_subtensor(history[day], incidence), cumulative + incidence

        (histories, _), _ = pytensor.scan(
            fn=renewal_step,
            sequences=[
                pt.arange(n_seed, n_day),
                reproduction[n_seed:],
                generation[n_seed:],
            ],
            outputs_info=[initial_history, n_seed * y],
            non_sequences=[pop],
            strict=True,
            name="epidemic_renewal",
        )
        prediction = pm.Deterministic(
            "prediction", histories[-1], dims=("day", "country")
        )
        cumulative_before = pt.concatenate(
            [
                pt.zeros((1, n_country), dtype=y.dtype),
                pt.cumsum(prediction, axis=0)[:-1],
            ],
            axis=0,
        )
        susceptible = 1.0 - cumulative_before / pop
        susceptible = pt.set_subtensor(susceptible[:n_seed], 1.0)
        pm.Deterministic("Rt_adj", reproduction * susceptible, dims=("day", "country"))
        expected = (
            pt.matmul(fatality_delay, prediction.T[:, :, None])[:, :, 0].T
            * ifr_noise[None, :]
        )
        # The first death mean follows the specified positive seed.
        expected = pt.set_subtensor(expected[0], 1e-15 * y)
        expected_deaths = pm.Deterministic(
            "E_deaths", expected, dims=("day", "country")
        )
        pm.NegativeBinomial(
            "deaths",
            mu=expected_deaths[death_day, death_country],
            alpha=phi,
            observed=deaths[observed_day, observed_country],
            dims="observation",
        )

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
