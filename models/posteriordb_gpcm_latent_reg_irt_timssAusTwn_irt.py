"""
Model: Partial credit and generalized partial credit models with latent regression
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/gpcm_latent_reg_irt.stan
Authors: Not specified in upstream model metadata.
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/gpcm_latent_reg_irt.py
Upstream catalogue contributor: Keane Nguyen
Description: The PCM (Masters 1982) is used for item response data that features more than two
    ordered response categories for some or all items. This extends the PCM by including a
    discrimination term.

Data: timssAusTwn_irt (upstream posterior: timssAusTwn_irt-gpcm_latent_reg_irt)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/timssAusTwn_irt.json.zip
    Archive SHA-256: c816b0dd6bf45a8522e5930efbfd49dac0f90c129db5232a055bb2ffba9fdcaf
    Local archive: data/posteriordb_timssAusTwn_irt.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/timssAusTwn_irt-gpcm_latent_reg_irt.json
Reference URL: https://github.com/danielcfurr/edstan
Reference URL: https://mc-stan.org/users/documentation/case-studies/pcm_and_gpcm.html

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native generalized partial-credit model with latent regression from edstan.
Native item-response models from the edstan probability definitions.

Covariate scaling follows Furr (2017), sections 1.1 and 1.3.
Source: https://mc-stan.org/learn-stan/case-studies/rasch_and_2pl.html

Binary columns use their range.
Continuous columns use twice their sample standard deviation, and the intercept stays unchanged.

This corrects the published binary-column counter defect.
It changes the induced coefficient priors for fims_Aus_Jpn_irt-2pl_latent_reg_irt and timssAusTwn_irt-gpcm_latent_reg_irt.
The intercept-only science_irt-grsm_latent_reg_irt pair does not change.

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
    def _item_response_design(data):
        """Apply the documented binary-range and continuous two-SD scaling."""
        design = np.asarray(data["W"], dtype=float)
        center = np.zeros(design.shape[1])
        scale = np.ones(design.shape[1])
        if design.shape[1] > 1:
            center[1:] = design[:, 1:].mean(axis=0)
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
