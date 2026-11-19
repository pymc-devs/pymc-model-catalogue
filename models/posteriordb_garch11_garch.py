"""Generalized Autoregressive Conditional Heteroscedastic Model.

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
    def _garch11_variance_step(
        previous_y, previous_variance, mu, alpha0, alpha1, beta1
    ):
        return alpha0 + alpha1 * pt.square(previous_y - mu) + beta1 * previous_variance

    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_garch.json.zip"
        ) as archive,
        archive.open("garch.json") as stream,
    ):
        data = json.load(stream)
    y = np.asarray(data["y"], dtype=float)
    with pm.Model(coords={"time": np.arange(int(data["T"]))}) as model:
        mu = pm.Flat("mu", initval=float(y.mean()))
        alpha0 = pm.HalfFlat("alpha0", initval=1)
        alpha1 = pm.Uniform("alpha1", lower=0, upper=1, initval=0.2)
        beta1 = pm.Uniform("beta1", lower=0, upper=1 - alpha1, initval=0.5)
        # Remove the conditional uniform normalizer to retain the flat joint density.
        pm.Potential("coefficient_density", pt.log1p(-alpha1))
        observations = pm.Data("observations", y, dims="time")
        first_variance = pt.as_tensor_variable(np.float64(data["sigma1"]) ** 2)
        if len(y) > 1:
            later_variance, _ = pytensor.scan(
                fn=_garch11_variance_step,
                sequences=observations[:-1],
                outputs_info=first_variance,
                non_sequences=[mu, alpha0, alpha1, beta1],
                strict=True,
                name="garch11_variance",
            )
            variance = pt.concatenate([first_variance[None], later_variance])
        else:
            variance = first_variance[None]
        pm.Normal("y", mu=mu, sigma=pt.sqrt(variance), observed=y, dims="time")
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
