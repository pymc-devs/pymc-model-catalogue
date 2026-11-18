"""
Model: Lotka-Volterra Error Model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/lotka_volterra.stan
Authors: Lotka, AJ; Volterra, Vito (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/lotka_volterra.py
Upstream catalogue contributor: Oliver Järnefelt
Description: A statistical model to account for measurement error and unexplained variation that
    uses the deterministic solutions to the Lotka-Volterra equations as expected population sizes

Data: hudson_lynx_hare (upstream posterior: hudson_lynx_hare-lotka_volterra)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/hudson_lynx_hare.json.zip
    Archive SHA-256: 36d116efc116fee9c9e7286df5545cbf11d08171c4f48b84476740464a4e33ec
    Local archive: data/posteriordb_hudson_lynx_hare.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/hudson_lynx_hare-lotka_volterra.json
References: lotka1925principles, volterra1926fluctuations, volterra1927variazioni (see posteriordb/source/bibliography/references.bib)
Reference URL: https://raw.githubusercontent.com/stan-dev/example-models/master/knitr/lotka-volterra/lotka-volterra.stan

Implementation notes:
Lotka-Volterra populations with lognormal measurement errors.

Source: posteriordb lotka_volterra, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4.
Positive normal rates retain their zero lower bound.
The initial populations remain latent and have their own observed measurements.

Stan comparison:
- A: Stan = -257.10417041859887; PyMC = -256.72246199321796.
- B: Stan = -182.3569016532067; PyMC = -181.98098398931793.
- Raw delta: Stan = 74.74726876539216; PyMC = 74.74147800390003.
- Raw comparison: MISMATCH; residual = 0.005790761492136198.
- Tolerance: 1.7474726876539216e-06
- Stan ODE integration and PyMC LSODA differ numerically; no solver-residual correction
  is applied.
- Original Stan SHA-256: f1cb4dda6af07ea721d5a38c6cbe8e3b60f7d48294db0374f331dc9803fe4328

Benchmark results:
- Original: logp = -148.2680, grad norm = 62.0095, 26068.9 us/call (5 evals).
- Frozen: logp = -148.2680, grad norm = 62.0095, 32025.4 us/call (5 evals).
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
    def _lotka_volterra_dynamics(population, time, rates):
        prey, predator = population[0], population[1]
        growth, predation, mortality, reproduction = (
            rates[0],
            rates[1],
            rates[2],
            rates[3],
        )
        interaction = prey * predator
        return [
            growth * prey - predation * interaction,
            reproduction * interaction - mortality * predator,
        ]

    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_hudson_lynx_hare.json.zip"
        ) as archive,
        archive.open("hudson_lynx_hare.json") as stream,
    ):
        data = json.load(stream)
    times = np.asarray(data["ts"], dtype=float)
    initial_observed = np.asarray(data["y_init"], dtype=float)
    observed = np.asarray(data["y"], dtype=float)
    # Stan uses RK45 with rtol=1e-5, atol=1e-3 and max_num_steps=500.
    # PyMC 6.4 DifferentialEquation uses SciPy odeint (LSODA) defaults.
    # The equations are unchanged, but solver and tolerance differences can
    # change evaluated logp and gradients, not just a normalization constant.
    dynamics = DifferentialEquation(
        func=_lotka_volterra_dynamics,
        times=times,
        n_states=2,
        n_theta=4,
        t0=0,
    )
    coords = {
        "time": times,
        "species": ["hare", "lynx"],
        "rate": ["growth", "predation", "mortality", "reproduction"],
    }
    with pm.Model(coords=coords) as model:
        # Stan evaluates Normal densities on positive-constrained theta.
        # TruncatedNormal additionally normalizes that restriction; fixed
        # prior means and scales make this only a constant logp offset.
        theta = pm.TruncatedNormal(
            "theta",
            mu=np.array([1.0, 0.05, 1.0, 0.05]),
            sigma=np.array([0.5, 0.05, 0.5, 0.05]),
            lower=0,
            dims="rate",
            initval=np.array([0.55, 0.028, 0.84, 0.026]),
        )
        z_init = pm.LogNormal(
            "z_init", mu=np.log(10.0), sigma=1, dims="species", initval=initial_observed
        )
        sigma = pm.LogNormal(
            "sigma", mu=-1, sigma=1, dims="species", initval=np.full(2, np.exp(-1.0))
        )
        populations = pm.Deterministic(
            "z", dynamics(y0=z_init, theta=theta), dims=("time", "species")
        )
        pm.LogNormal(
            "y_init",
            mu=pt.log(z_init),
            sigma=sigma,
            observed=initial_observed,
            dims="species",
        )
        pm.LogNormal(
            "y",
            mu=pt.log(populations),
            sigma=sigma,
            observed=observed,
            dims=("time", "species"),
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
