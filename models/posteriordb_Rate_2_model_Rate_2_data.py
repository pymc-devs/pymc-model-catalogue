"""Difference in success rates.

Stan comparison:
- A: Stan = -3.079291541514772; PyMC = -3.079291541514775.
- B: Stan = -3.376244044403064; PyMC = -3.3762440444030664.
- Raw delta: Stan = -0.296952502888292; PyMC = -0.29695250288829156.
- Raw comparison: MATCH; residual = -4.440892098500626e-16.
- Tolerance: 1.0029695250288828e-06
"""

import json
from pathlib import Path
from zipfile import ZipFile

import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_Rate_2_data.json.zip"
        ) as archive,
        archive.open("Rate_2_data.json") as stream,
    ):
        data = json.load(stream)
    with pm.Model() as model:
        theta1 = pm.Beta("theta1", alpha=1, beta=1)
        theta2 = pm.Beta("theta2", alpha=1, beta=1)
        pm.Binomial("k1", n=data["n1"], p=theta1, observed=data["k1"])
        pm.Binomial("k2", n=data["n2"], p=theta2, observed=data["k2"])
        pm.Deterministic("delta", theta1 - theta2)
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
