"""Partial credit and generalized partial credit models with latent regression.

Stan comparison: MISMATCH; MATCH after prior adjustment.
Different scaling of binary covariates changes the coefficient priors.
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
            Path(__file__).parent / "data" / "posteriordb_timssAusTwn_irt.json.zip"
        ) as archive,
        archive.open("timssAusTwn_irt.json") as stream,
    ):
        data = json.load(stream)
    item = np.asarray(data["ii"], dtype="int64") - 1
    person = np.asarray(data["jj"], dtype="int64") - 1
    response = np.asarray(data["y"], dtype="int64")
    steps = np.zeros(data["I"], dtype="int64")
    np.maximum.at(steps, item, response)
    offsets = np.concatenate(([0], np.cumsum(steps[:-1])))
    n_steps = int(steps.sum())
    max_steps = int(steps.max())
    category = np.arange(max_steps + 1)
    threshold_index = np.minimum(
        offsets[:, None] + np.arange(max_steps)[None, :],
        n_steps - 1,
    )
    coords = _item_response_coords(data)
    coords["item_step"] = np.arange(n_steps)
    with pm.Model(coords=coords) as model:
        theta, alpha = _item_response_latent_regression(data)
        # Stan sets the last effect to minus the sum of the free effects,
        # then applies Normal(0, 3) to the full vector. ZeroSumNormal uses
        # different coordinates for that same constrained density; the
        # volume factor is constant because the prior scale is fixed.
        beta = pm.ZeroSumNormal("beta", sigma=3, dims="item_step")
        threshold_sum = pt.concatenate(
            [
                pt.zeros((data["I"], 1)),
                pt.cumsum(beta[threshold_index], axis=1),
            ],
            axis=1,
        )
        # Category logits sum the adjacent-category log odds.
        logits = (alpha[item] * theta[person])[:, None] * category[
            None, :
        ] - threshold_sum[item]
        logits = pt.where(category[None, :] <= steps[item, None], logits, -np.inf)
        pm.Categorical("y", logit_p=logits, observed=response, dims="response")
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
