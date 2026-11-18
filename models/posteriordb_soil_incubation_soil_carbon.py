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

Implementation notes:
Two-pool carbon dynamics from the posteriordb soil_incubation model.

The observation model uses cumulative carbon loss, not measurement error scales.
Source: Sierra et al. (2014), as specified by posteriordb.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -32.35121812756171; PyMC = -28.885478931878026.
- B: Stan = -31.536547018749822; PyMC = -28.070807010787178.
- Raw delta: Stan = 0.8146711088118863; PyMC = 0.8146719210908486.
- Raw comparison: MATCH; residual = -8.122789623143944e-07.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 1.0081467192109083e-06
- Stan ODE integration and PyMC LSODA differ numerically; no solver-residual correction
  is applied.
- Original Stan SHA-256: 6c845fcee0e9e5a49508c5b3395bb09613495b2773d93dbeef9f72664ce10e57
- Saved points: models/posteriordb_logp/points/soil_carbon-soil_incubation.npz
- Two-point evidence only; no global equivalence, support or inference claim.

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
    # Stan uses integrate_ode_rk45 with its default solver controls.
    # PyMC 6.4 DifferentialEquation uses SciPy odeint (LSODA) defaults.
    # The equations are unchanged, but solver and tolerance differences can
    # change evaluated logp and gradients, not just a normalization constant.
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
