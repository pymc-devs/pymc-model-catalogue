"""Source-derived constrained-coordinate maps for dynamic posteriordb models.

These maps use original Stan parameters, never transformed parameters or measured
engine residuals. Normalizing constants are described in notes rather than added
to adjustments, because the comparison uses two-point log-density differences.
"""

import numpy as np


_PARAMETER_NAMES = {
    "accel_gp": (
        "Intercept",
        "sdgp_1",
        "lscale_1",
        "zgp_1",
        "Intercept_sigma",
        "sdgp_sigma_1",
        "lscale_sigma_1",
        "zgp_sigma_1",
    ),
    "accel_splines": (
        "Intercept",
        "bs",
        "zs_1_1",
        "sds_1_1",
        "Intercept_sigma",
        "bs_sigma",
        "zs_sigma_1_1",
        "sds_sigma_1_1",
    ),
    "arK": ("alpha", "beta", "sigma"),
    "arma11": ("mu", "phi", "theta", "sigma"),
    "bym2_offset_only": ("beta0", "sigma", "rho", "theta", "phi"),
    "covid19imperial_v2": (
        "mu",
        "alpha_hier",
        "kappa",
        "y",
        "phi",
        "tau",
        "ifr_noise",
    ),
    "covid19imperial_v3": (
        "mu",
        "alpha_hier",
        "kappa",
        "y",
        "phi",
        "tau",
        "ifr_noise",
    ),
    "election88_full": (
        "a",
        "b",
        "c",
        "d",
        "e",
        "beta",
        "sigma_a",
        "sigma_b",
        "sigma_c",
        "sigma_d",
        "sigma_e",
    ),
    "garch11": ("mu", "alpha0", "alpha1", "beta1"),
    "gp_pois_regr": ("rho", "alpha"),
    "gp_regr": ("rho", "alpha", "sigma"),
    "hier_2pl": ("theta", "mu", "tau"),
    "hierarchical_gp": (
        "GP_region_std",
        "GP_state_std",
        "year_std",
        "state_std",
        "region_std",
        "tot_var",
        "prop_var",
        "mu",
        "length_GP_region_long",
        "length_GP_state_long",
        "length_GP_region_short",
        "length_GP_state_short",
    ),
    "hmm_drive_0": ("theta1", "theta2", "phi", "lambda"),
    "hmm_drive_1": ("theta1", "theta2", "phi", "lambda"),
    "hmm_example": ("theta1", "theta2", "mu"),
    "irt_2pl": ("sigma_theta", "theta", "sigma_a", "a", "mu_b", "sigma_b", "b"),
    "kilpisjarvi": ("alpha", "beta", "sigma"),
    "kronecker_gp": ("var1", "bw1", "L", "sigma1"),
    "ldaK2": ("theta", "phi"),
    "ldaK5": ("theta", "phi"),
    "losscurve_sislob": ("omega", "theta", "LR", "mu_LR", "sd_LR", "loss_sd"),
    "low_dim_gauss_mix_collapse": ("mu", "sigma", "theta"),
    "low_dim_gauss_mix": ("mu", "sigma", "theta"),
    "nn_rbm1bJ100": ("sigma2_alpha", "sigma2_beta", "alpha", "beta", "alpha1", "beta1"),
    "nn_rbm1bJ10": ("sigma2_alpha", "sigma2_beta", "alpha", "beta", "alpha1", "beta1"),
    "normal_mixture_k": ("theta", "mu", "sigma"),
    "normal_mixture": ("theta", "mu"),
    "prophet": ("k", "m", "delta", "sigma_obs", "beta"),
    "state_space_stochastic_level_stochastic_seasonal": (
        "mu",
        "seasonal",
        "beta",
        "lambda",
        "sigma",
    ),
}

MODEL_NAMES = frozenset(_PARAMETER_NAMES)


_NOTES = {
    "accel_gp": (
        "In the measured installed PyMC/PyTensor, integer sigma=36 "
        "in both HalfStudentT GP amplitude priors has dtype int8: sigma**2 "
        "overflows to 16. Their native density is therefore HalfStudentT(3,4), "
        "not Stan's normalized positive StudentT(3,0,36). The named adjustment "
        "adds the two source-derived prior log-density differences to Stan. "
        "Original targets differ; this runtime discrepancy is not a "
        "best-practice reparameterization. Recheck this adjustment after engine upgrades.",
    ),
    "accel_splines": (
        "In the measured installed PyMC/PyTensor, integer sigma=36 in both "
        "HalfStudentT spline-scale priors has dtype int8: sigma**2 overflows "
        "to 16. Their native density is therefore HalfStudentT(3,4), not "
        "Stan's normalized positive StudentT(3,0,36). The named adjustment "
        "adds the two source-derived prior log-density differences to Stan. "
        "Original targets differ; this runtime discrepancy is not a "
        "best-practice reparameterization. Recheck this adjustment after "
        "engine upgrades.",
    ),
    "arK": (
        "HalfCauchy versus a Cauchy density on positive sigma contributes only log(2).",
    ),
    "arma11": (
        "HalfCauchy versus a Cauchy density on positive sigma contributes only log(2).",
    ),
    "bym2_offset_only": (
        "HalfNormal versus a Normal density on positive sigma contributes only log(2).",
    ),
    "election88_full": (
        "Categorical indices are shifted from Stan one-based to NumPy zero-based.",
        "The five Uniform(0,100) scale normalizers are parameter-independent "
        "relative to the original Stan bounded scales.",
    ),
    "garch11": (
        "The PyMC log1p(-alpha1) potential cancels its conditional Uniform "
        "beta1 normalizer, leaving the flat joint density on the triangle "
        "alpha1>0, beta1>0, alpha1+beta1<1 that Stan declares.",
    ),
    "gp_pois_regr": (
        "Set f_rotated_=f_tilde: PyMC gp/gp.py Latent._build_prior names the "
        "free whitened vector f_rotated_. Neither density is evaluated in "
        "centered f coordinates, so there is no Cholesky determinant correction.",
        "HalfNormal versus a Normal density on positive alpha contributes only log(2).",
    ),
    "gp_regr": (
        "The two HalfNormal versus positive Normal density normalizers are constants.",
    ),
    "hier_2pl": (
        "Set xi[i,:]=[xi1[i],xi2[i]] and correlation_unit=(L_Omega[1,0]+1)/2.",
        "In dimension two LKJ(eta=4) has density proportional to (1-rho^2)^3; "
        "Beta(4,4) on u=(rho+1)/2 is its exact coordinate rewrite. "
        "log|d rho/d u|=log(2) is constant and omitted from delta adjustments; "
        "stacking the item vectors is a permutation with unit absolute determinant.",
    ),
    "hierarchical_gp": (
        "State-region and observation indices shift from one-based to zero-based.",
    ),
    "hmm_example": (
        "Explicit uniform simplex densities in PyMC add only constant "
        "normalizers to Stan's implicit flat simplex densities.",
    ),
    "irt_2pl": (
        "The three HalfCauchy versus positive Cauchy normalizers are constants.",
    ),
    "kronecker_gp": (
        "PyMC flattens observations as y.T.reshape(-1) and passes "
        "sigma=sqrt(sigma1) to KroneckerNormal, matching Stan's to_vector(y) "
        "ordering and additive noise variance sigma1.",
        "Stan's handwritten Gaussian factor omits -n1*n2/2*log(2*pi); "
        "PyMC includes it. Positive Cauchy and fixed-threshold truncated "
        "lognormal normalization differences are also parameter-independent.",
    ),
    "ldaK2": ("Document and word indices shift from one-based to zero-based.",),
    "ldaK5": ("Document and word indices shift from one-based to zero-based.",),
    "low_dim_gauss_mix_collapse": (
        "HalfNormal scale normalizers differ from positive Normal densities "
        "only by two log(2) constants.",
    ),
    "low_dim_gauss_mix": (
        "HalfNormal scale normalizers differ from positive Normal densities "
        "only by two log(2) constants.",
    ),
    "normal_mixture_k": (
        "PyMC's Dirichlet(ones) and Uniform(0,10) normalizers differ from "
        "Stan's implicit bounded flat densities only by constants.",
    ),
    "prophet": (
        "For the logistic trend, PyMC uses the telescoped form "
        "logit(trend/cap)=k*(t-m)+sum(delta_s*max(t-t_change_s,0)) in place "
        "of Stan's continuity recursion; m remains a time location, not a "
        "log-odds intercept.",
        "HalfNormal versus positive Normal sigma_obs contributes only log(2).",
    ),
    "state_space_stochastic_level_stochastic_seasonal": (
        "Native AR rho=-ones(11) with flat initial eleven states replaces "
        "Stan's seasonal[t]~Normal(-sum(previous eleven),sigma[0]); level "
        "transitions remain unnormalized Normal factors inside the fixed "
        "mean(y)+/-3*sample_sd(y) level bounds.",
        "Uniform level normalizers and three HalfStudentT versus positive "
        "Student-t scale normalizers are parameter-independent.",
    ),
}

_COVID_NOTES = (
    "Stan X[country,day,feature] is transposed only as data to PyMC "
    "(day,country,feature); delay convolutions use lag-1 array indices.",
    "The random-kappa truncation normalizer for positive mu is explicitly "
    "canceled by the PyMC mu_constraint_normalization potential. "
    "ifr_noise truncation and HalfNormal phi/kappa change only constants.",
)

_NETWORK_NOTES = ("Observed labels shift from 1..K to 0..K-1.",)

_ORDERED_MODELS = frozenset(
    {
        "hmm_drive_0",
        "hmm_drive_1",
        "hmm_example",
        "low_dim_gauss_mix",
        "state_space_stochastic_level_stochastic_seasonal",
    }
)


def map_parameters(
    name: str, data: dict, stan: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, float], list[str]]:
    """Map one original constrained Stan point and derive density corrections."""
    if name not in MODEL_NAMES:
        raise KeyError(f"Unknown dynamic Stan model: {name}")
    values = {key: np.asarray(stan[key]) for key in _PARAMETER_NAMES[name]}
    adjustments = {}
    notes = [
        f"Sources: pinned posteriordb Stan {name}.stan and the assigned native "
        f"models/posteriordb_{name}_*.py builder(s)."
    ]
    if name in {"covid19imperial_v2", "covid19imperial_v3"}:
        notes.extend(_COVID_NOTES)
    elif name in {"nn_rbm1bJ100", "nn_rbm1bJ10"}:
        notes.extend(_NETWORK_NOTES)
    else:
        notes.extend(_NOTES.get(name, ()))

    if name == "gp_pois_regr":
        values["f_rotated_"] = np.asarray(stan["f_tilde"])
    elif name == "hier_2pl":
        values["xi"] = np.column_stack((stan["xi1"], stan["xi2"]))
        values["correlation_unit"] = np.asarray((stan["L_Omega"][1, 0] + 1.0) / 2.0)
    elif name in {"accel_gp", "accel_splines"}:
        scales = (
            ("sdgp_1", "sdgp_sigma_1")
            if name == "accel_gp"
            else ("sds_1_1", "sds_sigma_1_1")
        )
        # Installed HalfStudentT squares the int8 constant 36 to 16.
        # This is log HalfT_3(scale=4) - log HalfT_3(scale=36), derived
        # from that kernel, not from either engine's measured joint logp.
        adjustments["changed_integer_scale_half_student_t_prior"] = float(
            sum(
                np.sum(
                    np.log(9.0)
                    - 2 * np.log1p(values[key] ** 2 / 48.0)
                    + 2 * np.log1p(values[key] ** 2 / 3888.0)
                )
                for key in scales
            )
        )

    if name in _ORDERED_MODELS:
        notes.append(
            "Shared-support limitation: original Stan points enforce all ordered "
            "or positive-ordered constraints. PyMC ordering supplied only by "
            "a transform is not separately enforced after transforms are removed. "
            "These evaluations compare points on the shared Stan-valid support, "
            "not global support or model equivalence."
        )
    return values, adjustments, notes
