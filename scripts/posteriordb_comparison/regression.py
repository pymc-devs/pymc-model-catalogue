"""Source-derived constrained-coordinate maps for the regression inventory.

All 78 assigned pairs (57 Stan definitions) retain the original free parameter
names and coordinates. The runner filters the pass-through dictionary against
PyMC free RV names; no Stan transformed parameter is required. Only constant
normalization differences occur in this group, so adjustments are empty. This
static derivation makes no claim about numerical agreement at evaluated points.
"""

import numpy as np

MODEL_NAMES = frozenset(
    {
        "GLMM1_model",
        "GLMM_Poisson_model",
        "GLM_Binomial_model",
        "GLM_Poisson_model",
        "Survey_model",
        "blr",
        "diamonds",
        "earn_height",
        "eight_schools_centered",
        "eight_schools_noncentered",
        "kidscore_interaction_c2",
        "kidscore_interaction_c",
        "kidscore_interaction",
        "kidscore_interaction_z",
        "kidscore_mom_work",
        "kidscore_momhs",
        "kidscore_momhsiq",
        "kidscore_momiq",
        "log10earn_height",
        "logearn_height",
        "logearn_height_male",
        "logearn_interaction",
        "logearn_interaction_z",
        "logearn_logheight_male",
        "logistic_regression_rhs",
        "logmesquite_logva",
        "logmesquite_logvas",
        "logmesquite_logvash",
        "logmesquite_logvolume",
        "logmesquite",
        "mesquite",
        "nes_logit_model",
        "nes",
        "pilots",
        "radon_county_intercept",
        "radon_county",
        "radon_hierarchical_intercept_centered",
        "radon_hierarchical_intercept_noncentered",
        "radon_partially_pooled_centered",
        "radon_partially_pooled_noncentered",
        "radon_pooled",
        "radon_variable_intercept_centered",
        "radon_variable_intercept_noncentered",
        "radon_variable_intercept_slope_centered",
        "radon_variable_intercept_slope_noncentered",
        "radon_variable_slope_centered",
        "radon_variable_slope_noncentered",
        "sesame_one_pred_a",
        "wells_daae_c_model",
        "wells_dae_c_model",
        "wells_dae_inter_model",
        "wells_dae_model",
        "wells_dist100_model",
        "wells_dist100ars_model",
        "wells_dist",
        "wells_interaction_c_model",
        "wells_interaction_model",
    }
)

_PREPROCESSING = {
    "kidscore_interaction": "[1, mom_hs, mom_iq, mom_hs*mom_iq]",
    "kidscore_interaction_c": "[1, hs-mean(hs), iq-mean(iq), centered_hs*centered_iq]",
    "kidscore_interaction_c2": "[1, hs-0.5, iq-100, (hs-0.5)*(iq-100)]",
    "kidscore_interaction_z": (
        "[1, z_hs, z_iq, z_hs*z_iq], with each z=(x-mean(x))/(2*sd(x)); "
        "Stan sd and NumPy std(ddof=1) both use sample SD"
    ),
    "kidscore_mom_work": "[1, mom_work==2, mom_work==3, mom_work==4]",
    "kidscore_momhs": "[1, mom_hs]",
    "kidscore_momhsiq": "[1, mom_hs, mom_iq]",
    "kidscore_momiq": "[1, mom_iq]",
    "earn_height": "earn as response; [1, height]",
    "log10earn_height": "log10(earn) as response; [1, height]",
    "logearn_height": "log(earn) as response; [1, height]",
    "logearn_height_male": "log(earn) as response; [1, height, male]",
    "logearn_interaction": "log(earn) as response; [1, height, male, height*male]",
    "logearn_interaction_z": (
        "log(earn) as response; [1, z_height, male, z_height*male], where "
        "z_height=(height-mean(height))/sd(height), using sample SD (ddof=1)"
    ),
    "logearn_logheight_male": "log(earn) as response; [1, log(height), male]",
    "mesquite": (
        "weight as response; [1, diam1, diam2, canopy_height, total_height, density, group]"
    ),
    "logmesquite": (
        "log(weight) as response; [1, log(diam1), log(diam2), log(canopy_height), "
        "log(total_height), log(density), group]"
    ),
    "logmesquite_logvolume": "log(weight) as response; [1, log(diam1*diam2*canopy_height)]",
    "logmesquite_logva": (
        "log(weight) as response; [1, log(diam1*diam2*canopy_height), log(diam1*diam2), group]"
    ),
    "logmesquite_logvas": (
        "log(weight) as response; [1, log(diam1*diam2*canopy_height), log(diam1*diam2), "
        "log(diam1/diam2), log(total_height), log(density), group]"
    ),
    "logmesquite_logvash": (
        "log(weight) as response; [1, log(diam1*diam2*canopy_height), log(diam1*diam2), "
        "log(diam1/diam2), log(total_height), group]"
    ),
    "wells_dist": "[1, dist], with the intercept stored in beta[0]",
    "wells_dist100_model": "[dist/100]",
    "wells_dist100ars_model": "[dist/100, arsenic]",
    "wells_interaction_model": "[dist/100, arsenic, (dist/100)*arsenic]",
    "wells_interaction_c_model": "[cd, ca, cd*ca]",
    "wells_dae_model": "[dist/100, arsenic, educ/4]",
    "wells_dae_c_model": "[cd, ca, cd*ca, educ/4]",
    "wells_daae_c_model": "[cd, ca, cd*ca, assoc, educ/4]",
    "wells_dae_inter_model": "[cd, ca, ce, cd*ca, cd*ce, ca*ce]",
}


def map_parameters(
    name: str, data: dict, stan: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, float], list[str]]:
    """Return same-name coordinates and independently derived constant-only notes."""
    if name not in MODEL_NAMES:
        raise KeyError(name)
    notes = [
        f"Sources: pinned stan/{name}.stan parameters/model blocks and "
        f"models/posteriordb_{name}_*.py build_model (each dataset variant inspected).",
        "Identity on all constrained free parameters, including original array order. "
        "No coordinate-change Jacobian or parameter-dependent factor correction. "
        "Stan support-transform and PyMC value-transform Jacobians are disabled.",
    ]
    if name.startswith("kidscore_"):
        notes.append(
            "Both predictors use " + _PREPROCESSING[name] + "; beta order is unchanged."
        )
        if name in {
            "kidscore_interaction",
            "kidscore_momhs",
            "kidscore_momhsiq",
            "kidscore_momiq",
        }:
            notes.append(
                "Stan sigma>0 with Cauchy(0,2.5), PyMC HalfCauchy(2.5): "
                "PyMC-Stan is the constant log(2). Beta has a flat prior in both."
            )
        else:
            notes.append("Stan implicit flat beta/sigma>0 match PyMC Flat/HalfFlat.")
    elif name.startswith("wells_"):
        notes.append(
            "Both use predictors "
            + _PREPROCESSING[name]
            + "; cd=(dist-mean(dist))/100, "
            "ca=arsenic-mean(arsenic), ce=(educ-mean(educ))/4 where present. "
            "Original observation and coefficient order are retained; all coefficients "
            "are flat. Except wells_dist, alpha is the separate intercept."
        )
    elif name in _PREPROCESSING:
        notes.append(
            "Both use " + _PREPROCESSING[name] + ". Beta order is unchanged, with "
            "Flat beta and HalfFlat sigma matching Stan's implicit priors. "
            "Response transformations are identical fixed-data operations, not "
            "parameter transforms requiring a Jacobian correction."
        )
    elif name.startswith("eight_schools_"):
        notes.append(
            "mu, tau and the native school coordinates are unchanged. "
            + (
                "Both use theta_trans~Normal(0,1), theta=mu+tau*theta_trans; "
                "do not substitute centered theta as the density coordinate. "
                if name.endswith("noncentered")
                else "Both use centered theta~Normal(mu,tau). "
            )
            + "Stan tau>0 with Cauchy(0,5) versus PyMC HalfCauchy(5) differs "
            "only by the constant log(2)."
        )
    elif name.startswith("radon_"):
        notes.append(
            "County labels are converted from Stan one-based to NumPy zero-based "
            "indices without reordering observations or county parameters. "
            "Residual SD and all varying-effect SDs retain their names and units."
        )
        if name == "radon_county":
            notes.append(
                "a, mu_a, sigma_a, sigma_y are unchanged. Stan implicit SD priors "
                "on (0,100) versus PyMC Uniform(0,100) contribute only -2*log(100)."
            )
        else:
            notes.append(
                "Each Stan positive-constrained Normal(0,1) SD becomes HalfNormal(1); "
                "each contributes the constant log(2), with no changed prior shape."
            )
            if name.endswith("noncentered"):
                notes.append(
                    "Keep alpha_raw and/or beta_raw as density coordinates: both "
                    "sources form alpha=mu_alpha+sigma_alpha*alpha_raw and/or "
                    "beta=mu_beta+sigma_beta*beta_raw. No centered/noncentered "
                    "conversion or scale Jacobian is applied between these paired sources."
                )
            elif name.endswith("centered"):
                notes.append("Both sources retain centered county alpha and/or beta.")
            if "hierarchical_intercept" in name:
                notes.append("beta order is [log_uppm, floor_measure] in both sources.")
    elif name == "GLMM1_model":
        notes.append(
            "alpha[obssite-1] matches Stan's replicated alpha row at each obsyear. "
            "mu_alpha, sd_alpha and centered alpha are unchanged. The implicit "
            "Stan sd_alpha prior on (0,5) versus Uniform(0,5) adds only -log(5)."
        )
    elif name in {"GLMM_Poisson_model", "GLM_Poisson_model", "GLM_Binomial_model"}:
        notes.append(
            "Identical year, year^2 and (for Poisson) year^3 predictors; PyMC's "
            "Horner expression is algebraically the same linear predictor. "
            "All coefficient names, units and bounds are unchanged."
        )
        if name == "GLMM_Poisson_model":
            notes.append(
                "Both retain centered eps~Normal(0,sigma) and identical explicit Uniform priors."
            )
        elif name == "GLM_Poisson_model":
            notes.append(
                "Stan implicit bounded coefficients versus PyMC Uniform priors "
                "add only -log(40)-3*log(20)."
            )
        else:
            notes.append(
                "Both coefficient priors are Normal(0,100), with the same binomial totals."
            )
    elif name == "Survey_model":
        notes.append(
            "theta is unchanged. Both sum the joint binomial likelihood over "
            "n=max(1,max(k)),...,nmax with weight 1/nmax; omitted lower n have "
            "zero likelihood. Uniform(0,1) adds zero log-normalization."
        )
    elif name == "blr":
        notes.append(
            "beta, sigma and X are unchanged. Stan sigma>0 with Normal(0,10) "
            "versus PyMC HalfNormal(10) adds the constant log(2)."
        )
    elif name == "diamonds":
        notes.append(
            "Both discard X's intercept column and center each remaining column "
            "by its mean. Use the original centered Intercept, not generated "
            "b_Intercept=Intercept-means_X@b. b and sigma are unchanged. "
            "Stan explicitly subtracts StudentT(3,0,10)'s log survival at zero, "
            "so its sigma normalization already matches HalfStudentT(3,10). "
            "Both honor prior_only."
        )
    elif name == "logistic_regression_rhs":
        notes.append(
            "beta0, z, tau, lambda and caux are unchanged. Both set c^2="
            "slab_scale^2*caux and beta=z*tau*sqrt(c^2*lambda^2/"
            "(c^2+tau^2*lambda^2)); beta is not a density coordinate. "
            "Both use tau scale 2*scale_global and caux~InvGamma(slab_df/2,slab_df/2). "
            "PyMC's positive half-t normalizations for tau and lambda add only "
            f"(d+1)*log(2) with d={data['d']}."
        )
    elif name == "nes":
        notes.append(
            "Identical beta order [intercept, real_ideo, race_adj, age==2, age==3, "
            "age==4, educ1, gender, income], partyid7 response and flat beta/sigma>0 priors."
        )
    elif name == "nes_logit_model":
        notes.append(
            "Identical flat alpha and length-one beta; logit=alpha+beta[0]*income."
        )
    elif name == "pilots":
        notes.append(
            "a, b, mu_a, mu_b and all SDs are unchanged; a~Normal(10*mu_a,sigma_a), "
            "b~Normal(10*mu_b,sigma_b). One-based group/scenario indices become "
            "zero-based without row reordering. Implicit Stan (0,100) priors "
            "versus three PyMC Uniforms add only -3*log(100)."
        )
    elif name == "sesame_one_pred_a":
        notes.append(
            "Identical watched response, beta order [intercept, encouraged], "
            "and Flat beta/HalfFlat sigma matching the implicit Stan priors."
        )
    return dict(stan), {}, notes
