"""
Model: A one compartment model
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/one_comp_mm_elim_abs.stan
Authors: Not specified in upstream model metadata.
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/one_comp_mm_elim_abs.py
Upstream catalogue contributor: Mans Magnusson
Description: Note that the model in the database currently hardcodes C0.

Data: one_comp_mm_elim_abs (upstream posterior: one_comp_mm_elim_abs-one_comp_mm_elim_abs)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/one_comp_mm_elim_abs.json.zip
    Archive SHA-256: 67f9ae02f88a9fb73c7ae38fc9b3e245f776d0bd2d34169cc96fbdfe6e93be68
    Local archive: data/posteriordb_one_comp_mm_elim_abs.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/one_comp_mm_elim_abs-one_comp_mm_elim_abs.json
Reference URL: https://github.com/stan-dev/stat_comp_benchmarks/tree/master/benchmarks/pkpd

Implementation notes:
One-compartment absorption with Michaelis-Menten elimination.

Source: posteriordb one_comp_mm_elim_abs, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4.
The initial concentration is zero, as specified by the published database model.
Native ODE integration separates the concentration dynamics from lognormal measurement errors.

Stan comparison:
- A: Stan = -47.953679678651916; PyMC = -45.18109126320565.
- B: Stan = -44.48394585873322; PyMC = -41.71135715146996.
- Raw delta: Stan = 3.4697338199186945; PyMC = 3.46973411173569.
- Raw comparison: MATCH; residual = -2.918169954568839e-07.
- Tolerance: 1.0346973411173568e-06
- Stan ODE integration and PyMC LSODA differ numerically; no solver-residual correction
  is applied.
- Original Stan SHA-256: dbb2bdf39dd21bdf83f062b59a2d5f210b5ccd372d93631302d723a5b5f57d9c

Benchmark results:
- Original: logp = -44.3001, grad norm = 13.5193, 7444.0 us/call (5 evals).
- Frozen: logp = -44.3001, grad norm = 13.5193, 7692.5 us/call (5 evals).
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
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_one_comp_mm_elim_abs.json.zip"
        ) as archive,
        archive.open("one_comp_mm_elim_abs.json") as stream,
    ):
        data = json.load(stream)
    times = np.asarray(data["times"], dtype=float)
    observed = np.asarray(data["C_hat"], dtype=float)
    dose = float(data["D"])
    volume = float(data["V"])

    def concentration_dynamics(concentration, time, rates):
        absorption, half_saturation, maximum_elimination = rates[0], rates[1], rates[2]
        absorbed_dose = pt.switch(
            time > 0, dose * absorption * pt.exp(-absorption * time) / volume, 0.0
        )
        eliminated_dose = (
            maximum_elimination
            * concentration[0]
            / (volume * (half_saturation + concentration[0]))
        )
        return [absorbed_dose - eliminated_dose]

    # Stan uses integrate_ode_bdf with its default solver controls.
    # PyMC 6.4 DifferentialEquation uses SciPy odeint (LSODA) defaults.
    # The equations are unchanged, but solver and tolerance differences can
    # change evaluated logp and gradients, not just a normalization constant.
    dynamics = DifferentialEquation(
        func=concentration_dynamics,
        times=times,
        n_states=1,
        n_theta=3,
        t0=float(data["t0"]),
    )
    with pm.Model(coords={"time": times}) as model:
        k_a = pm.HalfCauchy("k_a", beta=1, initval=1.0)
        K_m = pm.HalfCauchy("K_m", beta=1, initval=1.0)
        V_m = pm.HalfCauchy("V_m", beta=1, initval=1.0)
        sigma = pm.HalfCauchy("sigma", beta=1, initval=0.2)
        concentration = pm.Deterministic(
            "C",
            dynamics(y0=np.zeros(1), theta=pt.stack([k_a, K_m, V_m]))[:, 0],
            dims="time",
        )
        pm.LogNormal(
            "C_hat",
            mu=pt.log(concentration),
            sigma=sigma,
            observed=observed,
            dims="time",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
