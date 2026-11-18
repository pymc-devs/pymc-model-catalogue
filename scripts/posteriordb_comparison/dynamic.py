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
        "Identity of both whitened GP coefficient vectors: for dimension D, "
        "spectral SD_m = sdgp*(sqrt(2*pi)*lscale)^(D/2)"
        "*exp(-lscale^2*sum(slambda_m^2)/4); both predictors use the supplied "
        "Xgp matrices times spectral_SD*zgp, with the scale predictor exponentiated.",
        "Student-t intercepts, inverse-gamma lengths and the prior_only switch "
        "match. In the measured installed PyMC/PyTensor, integer sigma=36 "
        "in both HalfStudentT GP amplitude priors has dtype int8: sigma**2 "
        "overflows to 16. Their native density is therefore HalfStudentT(3,4), "
        "not Stan's normalized positive StudentT(3,0,36). The named adjustment "
        "adds the two source-derived prior log-density differences to Stan. "
        "Original targets differ; this runtime discrepancy is not a "
        "best-practice reparameterization. Recheck this adjustment after engine upgrades.",
    ),
    "accel_splines": (
        "Identity of unpenalized bs/bs_sigma and standardized zs vectors; "
        "spline coefficients are sds*zs in both predictors, and sigma is exp "
        "of its linear predictor. The unpenalized coefficients have flat densities.",
        "Student-t intercepts and prior_only agree. In the measured installed "
        "PyMC/PyTensor, integer sigma=36 in both HalfStudentT spline-scale "
        "priors has dtype int8: sigma**2 overflows to 16. Their native density "
        "is therefore HalfStudentT(3,4), not Stan's normalized positive "
        "StudentT(3,0,36). The named adjustment adds the two source-derived "
        "prior log-density differences to Stan. Original targets differ; "
        "this runtime discrepancy is not a best-practice reparameterization. "
        "Recheck this adjustment after engine upgrades.",
    ),
    "arK": (
        "Identity of alpha, lag-ordered beta[1:K], and sigma; row t uses "
        "y[t-1],...,y[t-K]. The first K observations only condition the likelihood.",
        "HalfCauchy versus a Cauchy density on positive sigma contributes only log(2).",
    ),
    "arma11": (
        "Identity of mu, phi, theta, sigma. Both recursions start with "
        "nu[1]=mu+phi*mu and zero pre-series error; later nu[t]="
        "mu+phi*y[t-1]+theta*(y[t-1]-nu[t-1]). Normal errors equal the "
        "conditional Normal observation likelihood, with a unit triangular Jacobian.",
        "HalfCauchy versus a Cauchy density on positive sigma contributes only log(2).",
    ),
    "bym2_offset_only": (
        "Identity of spatial phi and unstructured theta; convolved effect="
        "sqrt(1-rho)*theta+sqrt(rho/scaling_factor)*phi. The ICAR factor uses "
        "one copy per undirected edge and Normal(sum(phi);0,0.001*N), matching "
        "the original soft sum constraint without centering or projection.",
        "HalfNormal versus a Normal density on positive sigma contributes only log(2).",
    ),
    "election88_full": (
        "Identity of the five centered effect vectors and their scales. beta "
        "order is intercept, black, female, previous_vote, female*black; all "
        "categorical indices are shifted from Stan one-based to NumPy zero-based.",
        "The five Uniform(0,100) scale normalizers are parameter-independent; "
        "the original Stan bounded scales have flat constrained densities.",
    ),
    "garch11": (
        "Identity on the triangle alpha1>0, beta1>0, alpha1+beta1<1. The PyMC "
        "log1p(-alpha1) potential cancels its conditional Uniform beta1 normalizer; "
        "the joint constrained coefficient density is flat as in Stan.",
        "Both recursions initialize variance=sigma1^2, then add alpha0 + "
        "alpha1*(previous_y-mu)^2 + beta1*previous_variance. No scale squaring "
        "or observation reordering is introduced by this map.",
    ),
    "gp_pois_regr": (
        "Set f_rotated_=f_tilde. Both whitened standard Normal vectors generate "
        "f=chol(K+1e-10*I)*z with K_ij=alpha^2*exp(-(x_i-x_j)^2/(2*rho^2)); "
        "neither density is evaluated in centered f coordinates, so there is "
        "no Cholesky determinant correction. PyMC gp/gp.py Latent._build_prior "
        "names the free whitened vector f_rotated_.",
        "HalfNormal versus a Normal density on positive alpha contributes only log(2).",
    ),
    "gp_regr": (
        "Identity of rho, alpha and sigma; both marginalized Normal covariances "
        "are alpha^2*ExpQuad(rho)+sigma*I. sigma is additive variance, not an SD "
        "to square. Observed y retains its supplied order.",
        "The two HalfNormal versus positive Normal density normalizers are constants.",
    ),
    "hier_2pl": (
        "Set xi[i,:]=[xi1[i],xi2[i]] and correlation_unit=(L_Omega[1,0]+1)/2. "
        "rho=L_Omega[1,0]; diag(tau)*[[1,0],[rho,sqrt(1-rho^2)]] is the same "
        "bivariate covariance Cholesky factor. Item discrimination=exp(xi[:,0]), "
        "difficulty=xi[:,1], and abilities theta remain centered.",
        "In dimension two LKJ(eta=4) has density proportional to (1-rho^2)^3; "
        "Beta(4,4) on u=(rho+1)/2 is its exact coordinate rewrite. "
        "log|d rho/d u|=log(2) is constant and omitted from delta adjustments; "
        "stacking the item vectors is a permutation with unit absolute determinant.",
    ),
    "hierarchical_gp": (
        "Identity of whitened GP matrices with shape (year,region) and "
        "(year,state), never transposed. Both left-multiply each temporal "
        "Cholesky factor into these columns. Region covariance=sum over long/short "
        "kernels of 17*tot_var*prop_var[j]*ExpQuad(length_j)+1e-6*I; state "
        "covariance uses the corresponding state components.",
        "The 17-simplex prop_var retains its component order: year, region, "
        "ten state-region scales, region_long, state_long, region_short, "
        "state_short, error. State-region and observation indices shift only "
        "from one-based to zero-based. All priors match in these coordinates.",
    ),
    "hmm_drive_0": (
        "Identity of two transition simplex rows and positive ordered emission "
        "rates phi/lambda. Both likelihoods sum all hidden paths by the same "
        "log-space forward recursion, with initial state weights one rather "
        "than a normalized uniform distribution. Each emission is the sum "
        "of Exponential(u;phi) and Exponential(v;lambda) log densities.",
    ),
    "hmm_drive_1": (
        "Identity of two transition simplex rows and ordered emission means "
        "phi/lambda. Both forward algorithms marginalize the same hidden paths "
        "with initial state weights one; u and v Normal emission SDs are the "
        "fixed data tau and rho, respectively.",
    ),
    "hmm_example": (
        "Identity of transition simplex rows and positive ordered means. "
        "The same Gaussian forward recursion marginalizes hidden states with "
        "initial weights one. Explicit uniform simplex densities in PyMC add "
        "only constant normalizers to Stan's implicit flat simplex densities.",
    ),
    "irt_2pl": (
        "Identity of abilities theta, positive lognormal discrimination a, "
        "difficulty b, mu_b and the three hierarchical SDs. The response "
        "matrix has shape (item,person), with logits a[:,None] * "
        "(theta[None,:]-b[:,None]); no transpose, log(a) rewrite or "
        "noncentering is needed.",
        "The three HalfCauchy versus positive Cauchy normalizers are constants.",
    ),
    "kilpisjarvi": (
        "Identity of alpha, beta and sigma, preserving the supplied uncentered "
        "year predictor x and its adjustable prior parameters. HalfFlat sigma "
        "matches the original positive flat constrained density.",
    ),
    "kronecker_gp": (
        "Identity of variance var1, bandwidth bw1, lower correlation Cholesky "
        "L and additive noise variance sigma1>1e-5. Omega=L@L.T. The temporal "
        "covariance is var1*exp(-bw1*distance^2)+1e-5*I. The flattened PyMC "
        "observations y.T.reshape(-1) put category fastest, matching Stan "
        "to_vector(y) and temporal_covariance tensor-product Omega. "
        "KroneckerNormal sigma=sqrt(sigma1) preserves additive noise variance.",
        "The installed LKJCorr has lower-Cholesky support and logp "
        "C+sum_i(2*eta-2+K-i)*log(L_ii), with one-based i. This includes "
        "the same Cholesky-to-correlation measure term as Stan's "
        "lkj_corr_cholesky; no parameter-dependent correction is needed.",
        "Stan's handwritten Gaussian factor omits -n1*n2/2*log(2*pi); "
        "PyMC includes it. Positive Cauchy and fixed-threshold truncated "
        "lognormal normalization differences are also parameter-independent.",
    ),
    "ldaK2": (
        "Identity of theta with shape (M,2) and phi with shape (2,V), each "
        "row a simplex. Priors are Dirichlet(ones). Each token contributes "
        "logsumexp_k(log(theta[doc,k])+log(phi[k,word])); topic indicators "
        "are marginalized identically, retaining token order and shifting "
        "only document/word indices to zero-based.",
    ),
    "ldaK5": (
        "Identity of theta with shape (M,5) and phi with shape (5,V), each "
        "row a simplex with the supplied alpha/beta concentration vectors. "
        "Each token contributes logsumexp over its five topics; no "
        "categorical latent variable or document-word count normalization "
        "is introduced. Only document/word indices shift to zero-based.",
    ),
    "losscurve_sislob": (
        "Identity of omega, theta, LR, mu_LR, sd_LR and loss_sd. The supplied "
        "growthmodel_id selects the same Weibull 1-exp(-(t/theta)^omega) "
        "or log-logistic sigmoid(omega*(log(t)-log(theta))) growth. "
        "Loss mean=LR[cohort]*premium[cohort]*growth[t_idx] and observation "
        "SD=loss_sd*premium[cohort], preserving observation order and all priors.",
    ),
    "low_dim_gauss_mix_collapse": (
        "Identity of unrestricted two-component mu, positive sigma and scalar "
        "theta; no relabeling or sorting. Both likelihoods marginalize with "
        "weights [theta,1-theta] per observation. HalfNormal scale normalizers "
        "differ from positive Normal densities only by two log(2) constants.",
    ),
    "low_dim_gauss_mix": (
        "Identity of ordered mu, positive sigma and scalar theta; no relabeling. "
        "Both likelihoods marginalize with weights [theta,1-theta] per "
        "observation. HalfNormal scale normalizers differ from positive "
        "Normal densities only by two log(2) constants.",
    ),
    "normal_mixture_k": (
        "Identity of K-simplex theta, unrestricted mu and bounded sigma "
        "vectors, retaining component labels. The likelihood is the same "
        "per-observation logsumexp over weighted Normal components. "
        "PyMC's Dirichlet(ones) and Uniform(0,10) normalizers differ from "
        "Stan's implicit bounded flat densities only by constants.",
    ),
    "normal_mixture": (
        "Identity of scalar theta and mu[2]; both marginalized Normal "
        "likelihoods use weights [theta,1-theta] and known unit SD. "
        "No sorting or label permutation is performed.",
    ),
    "prophet": (
        "Identity of k, m, delta, sigma_obs and feature beta. Linear trend "
        "uses k*t+m+sum(delta_s*max(t-t_change_s,0)). For logistic trend, "
        "Stan's continuity recursion telescopes to logit(trend/cap)="
        "k*(t-m)+sum(delta_s*max(t-t_change_s,0)); thus m remains a time "
        "location, not a log-odds intercept. Additive/multiplicative feature "
        "masks and observation order are unchanged.",
        "HalfNormal versus positive Normal sigma_obs contributes only log(2).",
    ),
    "state_space_stochastic_level_stochastic_seasonal": (
        "Identity of bounded level mu, seasonal states, flat beta/lambda and "
        "three positive ordered SDs (seasonal,level,observation). Native AR "
        "rho=-ones(11) with flat initial eleven states equals Stan's "
        "seasonal[t]~Normal(-sum(previous eleven),sigma[0]); level "
        "transitions remain unnormalized Normal factors inside the fixed "
        "mean(y)+/-3*sample_sd(y) level bounds.",
        "Uniform level normalizers and three HalfStudentT versus positive "
        "Student-t scale normalizers are parameter-independent.",
    ),
}

_COVID_NOTES = (
    "Identity of all seven parameter blocks; alpha=alpha_hier-log(1.05)/6 "
    "is the same deterministic intervention effect. Stan X[country,day,feature] "
    "is transposed only as data to PyMC (day,country,feature). Both renewal "
    "recursions seed N0 days with y and use N0*y cumulative infections at "
    "the first post-seed day; delay convolutions use lag-1 array indices.",
    "Both likelihoods use negative-binomial mean E_deaths and shape phi, "
    "selecting each country's EpidemicStart..N inclusive in country-major "
    "order. The random-kappa truncation normalizer for positive mu is "
    "explicitly canceled by the PyMC mu_constraint_normalization potential. "
    "ifr_noise truncation and HalfNormal phi/kappa change only constants.",
)

_NETWORK_NOTES = (
    "Identity of alpha (M,J), beta (J,K-1), row-vector alpha1 (J,), beta1 "
    "(K-1,) and their variances. No transpose or Fortran/C flattening "
    "permutation is applied to matrices. The hidden layer is tanh(x@alpha+alpha1), "
    "and nonbaseline logits are hidden@beta+beta1. The first category logit "
    "is one, not zero; observed labels shift from 1..K to 0..K-1.",
    "The variance priors have inverse-gamma shape 0.25 and scales "
    "0.25*(0.05/M^2)^2 and 0.25*(0.05/J^2)^2. Conditional weight SDs "
    "are square roots of these variances, matching the original priors.",
)

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
        f"models/posteriordb_{name}_*.py builder(s). Original constrained "
        "parameter shapes/index order are preserved unless explicitly mapped below; "
        "automatic support-transform Jacobians are disabled in both engines."
    ]
    if name in {"covid19imperial_v2", "covid19imperial_v3"}:
        notes.extend(_COVID_NOTES)
    elif name in {"nn_rbm1bJ100", "nn_rbm1bJ10"}:
        notes.extend(_NETWORK_NOTES)
    else:
        notes.extend(_NOTES[name])

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
