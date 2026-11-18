"""
Model: A One Layer Restricted Boltzman Machine Neural Network (100 hidden units)
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/nn_rbm1bJ100.stan
Authors: Lampinen, Jouko and Vehtari, Aki (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/nn_rbm1bJ100.py
Upstream catalogue contributor: Mans Magnusson
Description: A One Layer Restricted Boltzman Machine Neural Network with added bias and 100 hidden
    units. The implementation follows that of Lampainen and Vehtari (2001).

Data: mnist (upstream posterior: mnist-nn_rbm1bJ100)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/mnist.json.zip
    Archive SHA-256: 9248453fb3d7bb927a21a3088dc74e9af6e2764083065a6b1bf799163bfc2968
    Local archive: data/posteriordb_mnist.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/mnist-nn_rbm1bJ100.json
References: lampinen2001bayesian (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/example-models/tree/master/knitr/neural-nets

Implementation notes:
One-hidden-layer classification network with one hundred hidden units.

Source: Lampinen and Vehtari (2001), as specified by posteriordb.

Stan comparison:
- A: Stan = -271514.2967306021; PyMC = -271514.29673058644.
- B: Stan = -274624.68105277163; PyMC = -274624.6810527595.
- Raw delta: Stan = -3110.384322169528; PyMC = -3110.3843221730785.
- Raw comparison: MATCH; residual = 3.550667315721512e-09.
- Tolerance: 3.210384322173078e-05
- Observed labels shift from 1..K to 0..K-1.
- Original Stan SHA-256: 810f47fa99f782219c6b16fdb23e719e126dfb678997c0ae0f4b3498a0cdc801

Benchmark results:
- Not measured.
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
