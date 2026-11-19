"""Autoregressive Moving Average.

Stan comparison: MATCH.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt


def build_model():
    def _arma11_error_step(current, previous, previous_error, mu, phi, theta):
        return current - mu - phi * previous - theta * previous_error

    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_arma.json.zip"
        ) as archive,
        archive.open("arma.json") as stream,
    ):
        data = json.load(stream)
    y = np.asarray(data["y"], dtype=float)
    with pm.Model(coords={"time": np.arange(int(data["T"]))}) as model:
        mu = pm.Normal("mu", mu=0, sigma=10)
        phi = pm.Normal("phi", mu=0, sigma=2)
        theta = pm.Normal("theta", mu=0, sigma=2)
        sigma = pm.HalfCauchy("sigma", beta=2.5)
        observations = pm.Data("observations", y, dims="time")
        first_prediction = mu + phi * mu
        first_error = observations[0] - first_prediction
        if len(y) > 1:
            later_errors, _ = pytensor.scan(
                fn=_arma11_error_step,
                sequences=[observations[1:], observations[:-1]],
                outputs_info=first_error,
                non_sequences=[mu, phi, theta],
                strict=True,
                name="arma11_errors",
            )
            errors = pt.concatenate([first_error[None], later_errors])
            prediction = pt.concatenate(
                [
                    first_prediction[None],
                    mu + phi * observations[:-1] + theta * errors[:-1],
                ]
            )
        else:
            prediction = first_prediction[None]
        pm.Normal("y", mu=prediction, sigma=sigma, observed=y, dims="time")
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
