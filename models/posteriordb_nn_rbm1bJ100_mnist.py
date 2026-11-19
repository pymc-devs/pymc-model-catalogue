"""A One Layer Restricted Boltzmann Machine Neural Network (100 hidden units).

Stan comparison: MATCH.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_mnist.json.zip"
        ) as archive,
        archive.open("mnist.json") as stream,
    ):
        data = json.load(stream)
    hidden_units = 100
    predictors = data["M"]
    categories = data["K"]
    variance_shape = 0.25
    alpha_variance_scale = variance_shape * (0.05 / predictors**2) ** 2
    beta_variance_scale = variance_shape * (0.05 / hidden_units**2) ** 2
    coords = {
        "observation": range(data["N"]),
        "predictor": range(predictors),
        "hidden_unit": range(hidden_units),
        "category": range(1, categories + 1),
        "nonbaseline_category": range(2, categories + 1),
    }
    with pm.Model(coords=coords) as model:
        x = pm.Data(
            "x",
            np.asarray(data["x"], dtype="float64"),
            dims=("observation", "predictor"),
        )
        # The variance densities have shape one quarter and no finite mean.
        # Use their modes as positive initial values.
        sigma2_alpha = pm.InverseGamma(
            "sigma2_alpha",
            alpha=variance_shape,
            beta=alpha_variance_scale,
            initval=alpha_variance_scale / (variance_shape + 1),
        )
        sigma2_beta = pm.InverseGamma(
            "sigma2_beta",
            alpha=variance_shape,
            beta=beta_variance_scale,
            initval=beta_variance_scale / (variance_shape + 1),
        )
        alpha = pm.Normal(
            "alpha",
            mu=0,
            sigma=pt.sqrt(sigma2_alpha),
            dims=("predictor", "hidden_unit"),
            initval=np.zeros((predictors, hidden_units)),
        )
        beta = pm.Normal(
            "beta",
            mu=0,
            sigma=pt.sqrt(sigma2_beta),
            dims=("hidden_unit", "nonbaseline_category"),
            initval=np.zeros((hidden_units, categories - 1)),
        )
        alpha1 = pm.Normal(
            "alpha1", mu=0, sigma=1, dims="hidden_unit", initval=np.zeros(hidden_units)
        )
        beta1 = pm.Normal(
            "beta1",
            mu=0,
            sigma=1,
            dims="nonbaseline_category",
            initval=np.zeros(categories - 1),
        )
        hidden = pt.tanh(pt.dot(x, alpha) + alpha1)
        nonbaseline_logits = pt.dot(hidden, beta) + beta1
        # The first category logit equals one, not zero.
        logits = pt.concatenate((pt.ones((data["N"], 1)), nonbaseline_logits), axis=1)
        pm.Categorical(
            "y",
            logit_p=logits,
            observed=np.asarray(data["y"], dtype="int64") - 1,
            dims="observation",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
