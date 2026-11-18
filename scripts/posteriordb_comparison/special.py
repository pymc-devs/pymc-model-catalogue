"""Source-derived mappings; corrections never receive measured engine logps."""

import numpy as np
from scipy.special import log_softmax, logsumexp
from scipy.stats import norm, t

MODEL_NAMES = frozenset(
    {
        "2pl_latent_reg_irt",
        "dogs_nonhierarchical",
        "gpcm_latent_reg_irt",
        "grsm_latent_reg_irt",
        "hmm_gaussian",
        "iohmm_reg",
        "lotka_volterra",
        "one_comp_mm_elim_abs",
        "sir",
        "soil_incubation",
    }
)


def stan_center(name, data):
    """ODE centers follow native explicit initial values, not agreement searches."""
    if name == "lotka_volterra":
        return {
            "theta": np.array([0.55, 0.028, 0.84, 0.026]),
            "z_init": np.asarray(data["y_init"], dtype=float),
            "sigma": np.full(2, np.exp(-1)),
        }
    if name == "one_comp_mm_elim_abs":
        return {"k_a": 1.0, "K_m": 1.0, "V_m": 1.0, "sigma": 0.2}
    if name == "sir":
        return {"beta": 0.2, "gamma": 0.1, "xi": 10.0, "delta": 0.1}
    if name == "soil_incubation":
        return {
            "gamma": 0.9,
            "k1": 0.1,
            "k2": 0.01,
            "alpha21": 0.1,
            "alpha12": 0.1,
            "sigma": 1.0,
        }
    return {}


def map_parameters(name, data, stan):
    values = dict(stan)
    adjustments = {}
    notes = []
    if name == "dogs_nonhierarchical":
        factor = np.diag(stan["sigma_logit_ab"]) @ stan["L_logit_ab"]
        # Preserve dog effects: z_Stan @ A == z_PyMC @ A.T. The block
        # determinant is det(A)/det(A.T)=1, also in the joint transformation.
        values["z"] = np.linalg.solve(factor, (stan["z"] @ factor).T).T
        values["rho_unit"] = (stan["L_logit_ab"][1, 0] + 1) / 2
        adjustments["changed_standard_normal_effect_prior"] = float(
            -0.5 * (np.square(values["z"]).sum() - np.square(stan["z"]).sum())
        )
        notes.extend(
            [
                "Original targets differ: Stan uses z @ A, PyMC z @ A.T. "
                "Map z_PyMC=solve(A,(z_Stan@A).T).T to preserve dog effects; "
                "add the independently computed standard-Normal prior difference.",
                "This z mapping has unit Jacobian. rho_unit=(L[1,0]+1)/2 "
                "has a constant Jacobian; LKJ(2) and Beta(2,2) then agree "
                "up to constants, as do Normal and HalfNormal factors.",
            ]
        )
    elif name in {"2pl_latent_reg_irt", "gpcm_latent_reg_irt", "grsm_latent_reg_irt"}:
        design = np.asarray(data["W"], dtype=float)
        stan_scale = np.ones(design.shape[1])
        native_scale = np.ones_like(stan_scale)
        if design.shape[1] > 1:
            # The original minmax_count Boolean-assignment defect makes every
            # nonintercept column use two sample SDs, including binary columns.
            stan_scale[1:] = 2 * design[:, 1:].std(axis=0, ddof=1)
            native_scale[1:] = [
                np.ptp(column)
                if np.unique(column).size == 2
                else 2 * column.std(ddof=1)
                for column in design[:, 1:].T
            ]
        values["lambda_adj"] = stan["lambda_adj"] * native_scale / stan_scale
        values["beta"] = np.append(stan["beta_free"], -stan["beta_free"].sum())
        if "kappa_free" in stan:
            values["kappa"] = np.append(stan["kappa_free"], -stan["kappa_free"].sum())
        notes.append(
            "Append minus the sum to beta_free (and kappa_free when present) "
            "for ZeroSumNormal. Fixed prior scale 3 makes the subspace-volume "
            "and normalization offsets constant."
        )
        if not np.array_equal(native_scale, stan_scale):
            adjustments["changed_student_t_coefficient_prior"] = float(
                (
                    t.logpdf(values["lambda_adj"], 3) - t.logpdf(stan["lambda_adj"], 3)
                ).sum()
            )
            notes.append(
                "Original targets differ: binary covariate scaling changes "
                "coefficient priors. Map lambda_PyMC=lambda_Stan*scale_PyMC/"
                "scale_Stan to preserve physical slopes and latent means; add "
                "the StudentT(3) prior difference. The Jacobian is data-only."
            )
    elif name == "hmm_gaussian":
        emissions = norm.logpdf(
            np.asarray(data["y"])[:, None], stan["mu"], stan["sigma"]
        )
        # Both forward algorithms share transitions and all later emissions.
        # Row zero reproduces Stan's vectorized scalar emission at time one;
        # row one assigns the state-specific emission used in the native model.
        alpha = np.stack(
            [
                np.log(stan["pi1"]) + emissions[0].sum(),
                np.log(stan["pi1"]) + emissions[0],
            ]
        )
        log_transition = np.log(stan["A"])
        for emission in emissions[1:]:
            alpha = logsumexp(alpha[:, :, None] + log_transition, axis=1) + emission
        likelihoods = logsumexp(alpha, axis=1)
        adjustments["changed_initial_emission_likelihood"] = float(
            likelihoods[1] - likelihoods[0]
        )
        notes.append(
            "Original targets differ: Stan adds the sum of first-time state "
            "emissions to every state; PyMC uses state-specific emissions. "
            "Adjustment is the difference of independently computed forward "
            "likelihoods. Ordered means are tested on shared Stan-valid support."
        )
    elif name == "iohmm_reg":
        predictors = np.asarray(data["u"])
        emissions = norm.logpdf(
            np.asarray(data["y"])[:, None], predictors @ stan["b"].T, stan["sigma"]
        )
        log_gate = log_softmax(predictors @ stan["w"].T, axis=1)
        alpha = np.log(stan["pi1"]) + emissions[0]
        for gate, emission in zip(log_gate[1:], emissions[1:], strict=True):
            alpha = logsumexp(alpha + gate) + emission
        original = logsumexp(alpha)
        corrected = (
            logsumexp(np.log(stan["pi1"]) + emissions[0])
            + logsumexp(log_gate[1:] + emissions[1:], axis=1).sum()
        )
        adjustments["changed_transition_gating_likelihood"] = float(
            corrected - original
        )
        notes.append(
            "Original targets differ: Stan gates source states; PyMC gates "
            "destination states. Adjustment is the independently computed "
            "destination-mixture likelihood minus the original forward likelihood."
        )
    else:
        notes.append(
            "Same constrained parameters and probability factors up to "
            "normalization constants. Stan ODE integration and PyMC LSODA "
            "differ numerically; no solver-residual correction is applied."
        )
    return values, adjustments, notes
