"""
Model: Multiple Predictor Linear Model
Source: posteriordb/posterior_database/models/stan/nes.stan
Authors: Gelman, Andrew and Hill, Jennifer (cited work)
Description: partyid7 ~ real_ideo + race_adj + age30_44 + age45_64 + age65up + educ1 + gender +
    income

Stan comparison:
- A: Stan = -3028.966902487587; PyMC = -3028.9669024875852.
- B: Stan = -9510.142648536486; PyMC = -9510.142648536494.
- Raw delta: Stan = -6481.175746048899; PyMC = -6481.175746048908.
- Raw comparison: MATCH; residual = 9.094947017729282e-12.
- Tolerance: 6.581175746048908e-05

Benchmark results:
- Original:  logp = -1042.3892, grad norm = 169.9735, 15.9 us/call (5 evals).
- Frozen:    logp = -1042.3892, grad norm = 169.9735, 15.0 us/call (5 evals).
- Local NUMBA validation; short timings are indicative.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_nes2000.json.zip"
        ) as archive,
        archive.open("nes2000.json") as stream,
    ):
        data = json.load(stream)
    response = np.asarray(data["partyid7"], dtype=float)
    real_ideo = np.asarray(data["real_ideo"], dtype=float)
    race_adj = np.asarray(data["race_adj"], dtype=float)
    age = np.asarray(data["age_discrete"], dtype=int)
    educ1 = np.asarray(data["educ1"], dtype=float)
    gender = np.asarray(data["gender"], dtype=float)
    income = np.asarray(data["income"], dtype=float)
    design = np.column_stack(
        (
            np.ones(len(response)),
            real_ideo,
            race_adj,
            age == 2,
            age == 3,
            age == 4,
            educ1,
            gender,
            income,
        )
    )
    coords = {
        "obs_id": np.arange(len(response)),
        "coefficient": [
            "intercept",
            "real_ideo",
            "race_adj",
            "age30_44",
            "age45_64",
            "age65up",
            "educ1",
            "gender",
            "income",
        ],
    }
    initial_beta = np.zeros(design.shape[1])
    initial_beta[0] = response.mean()
    with pm.Model(coords=coords) as model:
        x = pm.Data("design", design, dims=("obs_id", "coefficient"))
        beta = pm.Flat("beta", dims="coefficient", initval=initial_beta)
        sigma = pm.HalfFlat("sigma", initval=max(float(response.std()), 1.0))
        pm.Normal(
            "partyid7",
            mu=pm.math.dot(x, beta),
            sigma=sigma,
            observed=response,
            dims="obs_id",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
