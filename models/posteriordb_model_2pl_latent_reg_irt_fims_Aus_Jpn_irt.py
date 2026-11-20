"""
Model: Two-parameter logistic item theory model with latent regression
Source: posteriordb/posterior_database/models/stan/2pl_latent_reg_irt.stan
Description: Two-parameter logistic item response model with item difficulty, discrimination and
    latent regression on person-level covariates.

Changes from original:
- Scaled binary covariates by their range rather than twice their standard deviation, changing the
  induced coefficient priors.

Stan comparison:
- A: Stan = -5447.38105313543; PyMC = -5445.415477402245.
- B: Stan = -5549.40077511055; PyMC = -5547.401838847258.
- Raw delta: Stan = -102.01972197511986; PyMC = -101.98636144501234.
- Raw comparison: MISMATCH; residual = -0.033360530107529485.
- Additive Stan adjustment: A = -0.05197484520454432; B = -0.01861431509717404.
- Adjusted Stan delta = -101.98636144501249.
- Adjusted comparison: MATCH; residual = -1.5631940186722204e-13.
- Tolerance: 2.0198636144501246e-06

Benchmark results:
- Original:  logp = -5361.5964, grad norm = 468.4698, 349.7 us/call (5 evals).
- Frozen:    logp = -5361.5964, grad norm = 468.4698, 282.4 us/call (5 evals).
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    def _item_response_design(data):
        """Apply the documented binary-range and continuous two-SD scaling."""
        design = np.asarray(data["W"], dtype=float)
        center = np.zeros(design.shape[1])
        scale = np.ones(design.shape[1])
        if design.shape[1] > 1:
            center[1:] = design[:, 1:].mean(axis=0)
            # Stan's minmax_count expression assigns a Boolean instead of
            # accumulating endpoint counts, so binary columns use two SDs.
            # Range scaling fixes this, but retaining the lambda_adj prior
            # changes the induced priors on original-scale coefficients.
            scale[1:] = [
                np.ptp(column)
                if np.unique(column).size == 2
                else 2 * column.std(ddof=1)
                for column in design[:, 1:].T
            ]
        return (design - center) / scale, center, scale

    def _item_response_latent_regression(data):
        design, center, scale = _item_response_design(data)
        w = pm.Data("W_adj", design, dims=("person", "covariate"))
        coefficient = pm.StudentT("lambda_adj", nu=3, mu=0, sigma=1, dims="covariate")
        theta = pm.Normal("theta", mu=w @ coefficient, sigma=1, dims="person")
        slope = coefficient[1:] / scale[1:]
        intercept = coefficient[0] - pt.dot(center[1:], slope)
        pm.Deterministic(
            "lambda", pt.concatenate([intercept[None], slope]), dims="covariate"
        )
        alpha = pm.LogNormal(
            "alpha", mu=1, sigma=1, dims="item", initval=np.ones(data["I"])
        )
        return theta, alpha

    def _item_response_coords(data):
        return {
            "item": np.arange(1, data["I"] + 1),
            "person": np.arange(1, data["J"] + 1),
            "response": np.arange(data["N"]),
            "covariate": np.arange(data["K"]),
        }

    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_fims_Aus_Jpn_irt.json.zip"
        ) as archive,
        archive.open("fims_Aus_Jpn_irt.json") as stream,
    ):
        data = json.load(stream)
    item = np.asarray(data["ii"], dtype="int64") - 1
    person = np.asarray(data["jj"], dtype="int64") - 1
    with pm.Model(coords=_item_response_coords(data)) as model:
        theta, alpha = _item_response_latent_regression(data)
        # Stan sets the last effect to minus the sum of the free effects,
        # then applies Normal(0, 3) to the full vector. ZeroSumNormal uses
        # different coordinates for that same constrained density; the
        # volume factor is constant because the prior scale is fixed.
        beta = pm.ZeroSumNormal("beta", sigma=3, dims="item")
        pm.Bernoulli(
            "y",
            logit_p=alpha[item] * theta[person] - beta[item],
            observed=np.asarray(data["y"], dtype="int64"),
            dims="response",
        )
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
