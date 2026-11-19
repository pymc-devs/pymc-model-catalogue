"""
Model: A One Layer Restricted Boltzmann Machine Neural Network
Source: posteriordb/posterior_database/models/stan/nn_rbm1bJ10.stan
Authors: Jouko Lampinen, Aki Vehtari
Description: A One Layer Restricted Boltzmann Machine Neural Network with added bias and 10 hidden
    units. The implementation follows that of Lampainen and Vehtari (2001).

Stan comparison:
- A: Stan = -6984.3954375624235; PyMC = -6984.395437549347.
- B: Stan = -6277.298990779531; PyMC = -6277.298990766442.
- Raw delta: Stan = 707.0964467828926; PyMC = 707.0964467829044.
- Raw comparison: MATCH; residual = -1.1823431123048067e-11.
- Tolerance: 8.070964467829044e-06

Benchmark results:
- Original:  logp = 127507.2551, grad norm = 3919.2566, 249.2 us/call (5 evals).
- Frozen:    logp = 127507.2551, grad norm = 3919.2566, 272.1 us/call (5 evals).
- Local NUMBA validation; short timings are indicative.
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
            Path(__file__).parent / "data" / "posteriordb_mnist_100.json.zip"
        ) as archive,
        archive.open("mnist_100.json") as stream,
    ):
        data = json.load(stream)
    hidden_units = 10
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
        logits = pt.concatenate(
            (pt.ones_like(nonbaseline_logits[:, :1]), nonbaseline_logits), axis=1
        )
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
