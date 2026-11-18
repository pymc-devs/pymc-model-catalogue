"""
Model: Two-parameter logistic item theory model with latent regression
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/2pl_latent_reg_irt.stan
Authors: Not specified in upstream model metadata.
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/model_2pl_latent_reg_irt.py
Upstream catalogue contributor: Keane Nguyen
Description: The Rasch model (Rasch 1960) is an item response theory model for dichotomous items.
    The two-parameter logistic model (2PL) (Swaminathan and Gifford 1985) is an item response theory
    model that includes parameters for both the difficulty and discrimination of dichotomous items.
    The version presented includes a latent regression. However, the latent regression part of the
    model may be restricted to an intercept only, resulting in a regular 2PL.

Data: fims_Aus_Jpn_irt (upstream posterior: fims_Aus_Jpn_irt-2pl_latent_reg_irt)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/fims_Aus_Jpn_irt.json.zip
    Archive SHA-256: c6860be3eb744cd9479a5ff4c653a6aeeb4a1dfd15b6b4dbb75ae7e50e06b425
    Local archive: data/posteriordb_fims_Aus_Jpn_irt.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/fims_Aus_Jpn_irt-2pl_latent_reg_irt.json
Reference URL: https://github.com/danielcfurr/edstan
Reference URL: https://mc-stan.org/users/documentation/case-studies/rasch_and_2pl.html

Changes from original:
- Scale binary covariates by their range, changing the induced coefficient priors
  on the original covariate scale; see the scaling notes below.
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native two-parameter logistic model with latent regression from edstan.
Native item-response models from the edstan probability definitions.

Covariate scaling follows Furr (2017), sections 1.1 and 1.3.
Source: https://mc-stan.org/learn-stan/case-studies/rasch_and_2pl.html

Binary columns use their range.
Continuous columns use twice their sample standard deviation, and the intercept stays unchanged.

This corrects the published binary-column counter defect.
It changes the induced coefficient priors for fims_Aus_Jpn_irt-2pl_latent_reg_irt and timssAusTwn_irt-gpcm_latent_reg_irt.
The intercept-only science_irt-grsm_latent_reg_irt pair does not change.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -5447.38105313543; PyMC = -5445.415477402245.
- B: Stan = -5549.40077511055; PyMC = -5547.401838847258.
- Raw delta: Stan = -102.01972197511986; PyMC = -101.98636144501234.
- Raw comparison: MISMATCH; residual = -0.033360530107529485.
- Additive Stan adjustment changed_student_t_coefficient_prior: A =
  -0.05197484520454432; B = -0.01861431509717404.
- Adjusted Stan delta = -101.98636144501249.
- Adjusted comparison: MATCH; residual = -1.5631940186722204e-13.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 2.0198636144501246e-06
  using adjusted delta_Stan when an adjustment is present.
- Append minus the sum to beta_free (and kappa_free when present) for ZeroSumNormal.
  Fixed prior scale 3 makes the subspace-volume and normalization offsets constant.
- Original targets differ: binary covariate scaling changes coefficient priors. Map
  lambda_PyMC=lambda_Stan*scale_PyMC/scale_Stan to preserve physical slopes and latent
  means; add the StudentT(3) prior difference. The Jacobian is data-only.
- Original Stan SHA-256: 96c8f931ea6ff2569863cf1afa8b255aefe441516ab06705519b16ef1dc461b8
- Saved points: models/posteriordb_logp/points/fims_Aus_Jpn_irt-2pl_latent_reg_irt.npz
- Full evidence: models/posteriordb_logp/records/fims_Aus_Jpn_irt-2pl_latent_reg_irt.json
- Two-point evidence only; no global equivalence, support or inference claim.

Benchmark results:
- Original: logp = -5361.5964, grad norm = 468.4698, 349.7 us/call (5 evals).
- Frozen: logp = -5361.5964, grad norm = 468.4698, 282.4 us/call (5 evals).
- Local NUMBA validation; short timings are indicative.
- Environment: PyMC 6.4.0+9.gaf565aa28, PyTensor 3.3.8+30.g9a7f24ce0, Numba 0.68.0.
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
