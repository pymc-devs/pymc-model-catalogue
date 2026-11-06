"""Native item-response models from the edstan probability definitions.

Covariate scaling follows Furr (2017), sections 1.1 and 1.3.
Source: https://mc-stan.org/learn-stan/case-studies/rasch_and_2pl.html

Binary columns use their range.
Continuous columns use twice their sample standard deviation, and the intercept stays unchanged.

This corrects the published binary-column counter defect.
It changes the induced coefficient priors for fims_Aus_Jpn_irt-2pl_latent_reg_irt and timssAusTwn_irt-gpcm_latent_reg_irt.
The intercept-only science_irt-grsm_latent_reg_irt pair does not change.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def _item_response_design(data):
    """Apply the documented binary-range and continuous two-SD scaling."""
    design = np.asarray(data["W"], dtype=float)
    center = np.zeros(design.shape[1])
    scale = np.ones(design.shape[1])
    if design.shape[1] > 1:
        center[1:] = design[:, 1:].mean(axis=0)
        scale[1:] = [
            np.ptp(column) if np.unique(column).size == 2 else 2 * column.std(ddof=1)
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
