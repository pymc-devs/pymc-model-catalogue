"""
Model: Inferring population size with constant detection probability
Source: posteriordb/posterior_database/models/pymc/M0_model_discrete.py
Authors: M Kery, M Schaub (cited work)
Description: Capture-recapture model with constant detection probability and explicit population-
    inclusion indicators.

Stan comparison:
- Not applicable: no corresponding Stan model for the explicit discrete latents.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_M0_data.json.zip"
        ) as archive,
        archive.open("M0_data.json") as stream,
    ):
        data = json.load(stream)
    history = np.asarray(data["y"], dtype="int64")
    coords = {
        "individual": np.arange(data["M"]),
        "capture_period": np.arange(data["T"]),
    }
    with pm.Model(coords=coords) as model:
        omega = pm.Uniform("omega", 0, 1)
        p = pm.Uniform("p", 0, 1)
        z = pm.Bernoulli(
            "z",
            p=omega,
            dims="individual",
            initval=np.ones(data["M"], dtype="int64"),
        )
        pm.Bernoulli(
            "y",
            p=z[:, None] * p,
            observed=history,
            dims=("individual", "capture_period"),
        )
        pm.Deterministic("N", z.sum())
        no_capture = (1 - p) ** int(data["T"])
        pm.Deterministic(
            "omega_nd", omega * no_capture / (omega * no_capture + 1 - omega)
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model, discrete=True)
