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


def map_parameters(
    name: str, data: dict, stan: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, float], list[str]]:
    """Return same-name coordinates and independently derived constant-only notes."""
    if name not in MODEL_NAMES:
        raise KeyError(name)
    notes = [
        f"Sources: pinned stan/{name}.stan parameters/model blocks and "
        f"models/posteriordb_{name}_*.py build_model (each dataset variant inspected)."
    ]
    if name.startswith("kidscore_"):
        if name in {
            "kidscore_interaction",
            "kidscore_momhs",
            "kidscore_momhsiq",
            "kidscore_momiq",
        }:
            notes.append(
                "Stan sigma>0 with Cauchy(0,2.5) versus PyMC HalfCauchy(2.5): "
                "PyMC-Stan is the constant log(2)."
            )
    elif name.startswith("eight_schools_"):
        notes.append(
            "Stan tau>0 with Cauchy(0,5) versus PyMC HalfCauchy(5) differs "
            "only by the constant log(2)."
        )
    elif name.startswith("radon_"):
        notes.append(
            "County labels are converted from Stan one-based to NumPy zero-based indices."
        )
        if name == "radon_county":
            notes.append(
                "Stan implicit SD priors on (0,100) versus PyMC Uniform(0,100) "
                "contribute only -2*log(100)."
            )
        else:
            notes.append(
                "Each Stan positive-constrained Normal(0,1) SD becomes HalfNormal(1); "
                "each contributes the constant log(2)."
            )
    elif name == "GLMM1_model":
        notes.append(
            "PyMC indexes alpha[obssite-1] where Stan replicates the alpha row at "
            "each obsyear. The implicit Stan sd_alpha prior on (0,5) versus "
            "Uniform(0,5) adds only -log(5)."
        )
    elif name == "GLM_Poisson_model":
        notes.append(
            "Stan implicit bounded coefficients versus PyMC Uniform priors "
            "add only -log(40)-3*log(20)."
        )
    elif name == "blr":
        notes.append(
            "Stan sigma>0 with Normal(0,10) versus PyMC HalfNormal(10) adds "
            "the constant log(2)."
        )
    elif name == "logistic_regression_rhs":
        notes.append(
            "PyMC's positive half-t normalizations for tau and lambda add only "
            f"(d+1)*log(2) with d={data['d']}."
        )
    elif name == "pilots":
        notes.append(
            "One-based group/scenario indices become zero-based. Implicit Stan "
            "(0,100) priors versus three PyMC Uniforms add only -3*log(100)."
        )
    return dict(stan), {}, notes
