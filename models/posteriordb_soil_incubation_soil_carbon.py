"""
Model: Two Pool Linear Model with Feedback
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/soil_incubation.stan
Authors: Sierra, Carlos and Müller, Markus and Trumbore, Susan E (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/soil_incubation.py
Upstream catalogue contributor: Oliver Järnefelt
Description: A statistical two-pool model with decomposition rates for each pool as well as transfer
    rates from one pool to another.

Data: soil_carbon (upstream posterior: soil_carbon-soil_incubation)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/soil_carbon.json.zip
    Archive SHA-256: 5d878e8153e01148c5dc01336cee802ce455c853213ef46ade327b5d98edf5eb
    Local archive: data/posteriordb_soil_carbon.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/soil_carbon-soil_incubation.json
References: sierra2014modeling (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/example-models/blob/master/knitr/soil-carbon/soil_incubation.stan

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Two-pool carbon dynamics from the posteriordb soil_incubation model.

The observation model uses cumulative carbon loss, not measurement error scales.
Source: Sierra et al. (2014), as specified by posteriordb.

Benchmark results:
- Original: logp = -42.8159, grad norm = 15.6056, 3636.9 us/call (5 evals).
- Frozen: logp = -42.8159, grad norm = 15.6056, 4063.1 us/call (5 evals).
- Local NUMBA validation, including object-mode ODE execution; short timings are indicative.
- Environment: PyMC 6.4.0+9.gaf565aa28, PyTensor 3.3.8+30.g9a7f24ce0, Numba 0.68.0.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.ode import DifferentialEquation


def build_model():
    def _soil_incubation_dynamics(carbon, time, rates):
        """Return carbon derivatives for decomposition and transfer between two pools."""
        k1, k2, alpha21, alpha12 = rates[0], rates[1], rates[2], rates[3]
        return [
            -k1 * carbon[0] + alpha12 * k2 * carbon[1],
            alpha21 * k1 * carbon[0] - k2 * carbon[1],
        ]

    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_soil_carbon.json.zip"
        ) as archive,
        archive.open("soil_carbon.json") as stream,
    ):
        data = json.load(stream)
    times = np.asarray(data["ts"], dtype=float)
    total_carbon = float(data["totalC_t0"])
    dynamics = DifferentialEquation(
        func=_soil_incubation_dynamics,
        times=times,
        n_states=2,
        n_theta=4,
        t0=float(data["t0"]),
    )
    with pm.Model(coords={"time": times, "pool": [1, 2]}) as model:
        gamma = pm.Beta("gamma", alpha=10, beta=1, initval=0.9)
        k1 = pm.HalfNormal("k1", sigma=1, initval=0.1)
        k2 = pm.HalfNormal("k2", sigma=1, initval=0.01)
        alpha21 = pm.HalfNormal("alpha21", sigma=1, initval=0.1)
        alpha12 = pm.HalfNormal("alpha12", sigma=1, initval=0.1)
        sigma = pm.HalfCauchy("sigma", beta=1)
        initial_carbon = total_carbon * pt.stack([gamma, 1 - gamma])
        rates = pt.stack([k1, k2, alpha21, alpha12])
        carbon = dynamics(y0=initial_carbon, theta=rates)
        evolved_carbon = total_carbon - carbon.sum(axis=1)
        pm.Normal(
            "eCO2mean",
            mu=evolved_carbon,
            sigma=sigma,
            observed=np.asarray(data["eCO2mean"], dtype=float),
            dims="time",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
