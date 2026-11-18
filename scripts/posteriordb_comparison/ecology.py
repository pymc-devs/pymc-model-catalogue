"""Source-derived constrained-coordinate mappings for the ecology comparisons.

The original Stan parameters, not generated quantities or deterministics, define
these points. None of these 23 source pairs has a nonconstant changed probability
factor. Parameter-independent normalization and coordinate constants are disclosed
in the notes and deliberately omitted from the delta adjustments.
"""

import numpy as np


# Each entry lists PyMC free RVs, the exact source module, and the source-derived
# correspondence. The Stan source is stan/<model name>.stan in the pinned bundle.
_MODELS = {
    "M0_model": (
        ("omega", "p"),
        "models/posteriordb_M0_model_M0_data.py",
        "Both sources marginalize one inclusion indicator over each individual's "
        "complete history; Stan's binomial uses the same row sums as PyMC's "
        "ZeroInflatedBinomial. Generated population size N is not a free RV.",
    ),
    "Mb_model": (
        ("omega", "p", "c"),
        "models/posteriordb_Mb_model_Mb_data.py",
        "Both sources use p at the first occasion and p+(c-p)*previous_capture "
        "afterward, marginalizing inclusion over the complete history. No latent "
        "inclusion indicators or generated population size are added to the point.",
    ),
    "Mh_model": (
        ("omega", "mean_p", "sigma", "eps_raw"),
        "models/posteriordb_Mh_model_Mh_data.py",
        "Retain all M eps_raw values, including never-detected individuals. Both "
        "sources use logit(mean_p)+sigma*eps_raw and the same marginalized "
        "Binomial likelihood. PyMC Uniform(0,5) for sigma adds only -log(5) "
        "relative to Stan's implicit bounded-flat prior.",
    ),
    "Mt_model": (
        ("omega", "p"),
        "models/posteriordb_Mt_model_Mt_data.py",
        "The T detection probabilities retain occasion order. Both sources "
        "marginalize inclusion over each complete Bernoulli capture history, "
        "not independently at each occasion.",
    ),
    "Mtbh_model": (
        ("omega", "mean_p", "gamma", "sigma", "eps_raw"),
        "models/posteriordb_Mtbh_model_Mtbh_data.py",
        "Keep all M noncentered eps_raw values. Both sources use "
        "logit(mean_p[j])+sigma*eps_raw[i]+gamma*y[i,j-1], with zero previous "
        "capture at the first occasion, and marginalize individual inclusion. "
        "PyMC's sigma Uniform(0,3) contributes only the constant -log(3).",
    ),
    "Mth_model": (
        ("omega", "mean_p", "sigma", "eps_raw"),
        "models/posteriordb_Mth_model_Mth_data.py",
        "The T mean_p probabilities and all M eps_raw effects retain original "
        "order. Both sources use logit(mean_p[j])+sigma*eps_raw[i] and "
        "marginalize inclusion over complete histories. PyMC's sigma Uniform(0,5) "
        "contributes only the constant -log(5).",
    ),
    "Rate_1_model": (
        ("theta",),
        "models/posteriordb_Rate_1_model_Rate_1_data.py",
        "Identical Beta(1,1) prior and Binomial(n,theta) observed count.",
    ),
    "Rate_2_model": (
        ("theta1", "theta2"),
        "models/posteriordb_Rate_2_model_Rate_2_data.py",
        "Identical independent Beta(1,1) rates and their respective Binomial "
        "counts. delta=theta1-theta2 is deterministic, not a density coordinate.",
    ),
    "Rate_3_model": (
        ("theta",),
        "models/posteriordb_Rate_3_model_Rate_3_data.py",
        "Identical Beta(1,1) common rate and two Binomial observation factors.",
    ),
    "Rate_4_model": (
        ("theta", "thetaprior"),
        "models/posteriordb_Rate_4_model_Rate_4_data.py",
        "Both independent Beta(1,1) free rates are retained, including the "
        "unobserved thetaprior. Only theta enters the observed Binomial. "
        "Generated prior/posterior predictive counts are not free parameters.",
    ),
    "Rate_5_model": (
        ("theta",),
        "models/posteriordb_Rate_5_model_Rate_5_data.py",
        "Identical Beta(1,1) common rate and two Binomial counts; generated "
        "posterior-predictive counts do not enter either density.",
    ),
    "bones_model": (
        ("theta",),
        "models/posteriordb_bones_model_bones_data.py",
        "theta retains child order and Normal(0,36) SD. Stan's categories "
        "1..ncat become PyMC categories 0..ncat-1; missing grade=-1 entries "
        "are omitted in both. OrderedLogistic eta=delta*theta and cutpoints "
        "delta*gamma reproduce Stan's differences of inv_logit(delta*(theta-gamma)). "
        "The bundled discriminations are positive and active thresholds ordered.",
    ),
    "dogs": (
        ("beta",),
        "models/posteriordb_dogs_dogs.py",
        "beta order is intercept, avoidance, shock. Both sources count observed "
        "avoidances and shocks strictly before each trial and use the same "
        "Normal(0,100) coefficients and Bernoulli-logit likelihood. There are "
        "no dog-specific independent or correlated random effects in this source.",
    ),
    "dogs_hierarchical": (
        ("a", "b"),
        "models/posteriordb_dogs_hierarchical_dogs.py",
        "Despite its name, the original source has only shared probabilities "
        "a and b, with implicit Uniform(0,1) priors and no dog random effects. "
        "Both likelihoods use a**previous_shock*b**previous_avoid. The first "
        "trial has probability one; every bundled first response is one. "
        "PyMC preserves that support factor separately as first_shock.",
    ),
    "dogs_log": (
        ("beta",),
        "models/posteriordb_dogs_log_dogs.py",
        "beta order is avoidance, shock, with identical Uniform(-100,0) and "
        "Uniform(0,100) factors. Stan declares beta unconstrained, but these "
        "prior supports still apply. PyMC's logit_p equals Stan's inv_logit "
        "predictor; first-trial probability is 1/2 in both. The feasible center "
        "[-0.5,0.5] is chosen solely from support, not engine agreement. "
        "The constrained beta-to-beta map is identity with log Jacobian zero; "
        "automatic Stan/PyMC support-transform Jacobians remain disabled.",
    ),
    "dugongs_model": (
        ("alpha", "beta", "lambda", "tau"),
        "models/posteriordb_dugongs_model_dugongs_data.py",
        "Keep the original precision tau, with Gamma(shape=0.0001,rate=0.0001) "
        "in both sources. sigma=tau**(-1/2) is deterministic, not a new free "
        "coordinate, so no tau-to-sigma Jacobian applies. The lambda interval "
        "(0.5,1), Normal coefficient priors, and growth likelihood are unchanged.",
    ),
    "lsat_model": (
        ("alpha", "theta", "beta"),
        "models/posteriordb_lsat_model_lsat_data.py",
        "Expand response patterns in cumulative-count order; Stan's item/person "
        "matrix is transposed relative to PyMC's person/item likelihood, without "
        "permuting theta. Both logits are beta*theta-alpha. PyMC HalfNormal(100) "
        "for positive beta differs from Stan's constrained Normal(0,100) only "
        "by the constant +log(2).",
    ),
    "multi_occupancy": (
        ("alpha", "beta", "Omega", "sigma_uv", "correlation_unit", "uv"),
        "models/posteriordb_multi_occupancy_butterfly.py",
        "Map correlation_unit=(rho_uv+1)/2 and uv=column_stack(uv1,uv2), "
        "retaining all S species, including S-n never-observed species. The "
        "PyMC Cholesky factor induces the same covariance with diagonals "
        "sigma_uv**2 and off-diagonal sigma_uv[0]*sigma_uv[1]*rho_uv. "
        "Stan already evaluates Beta(2,2) at (rho_uv+1)/2 without a Jacobian; "
        "its value equals PyMC's Beta density at correlation_unit. The "
        "coordinate determinant |d rho_uv/d correlation_unit|=2 is constant "
        "(log(2)); normalizing Stan's rho prior removes the same constant. "
        "The two positive Cauchy-to-HalfCauchy normalizations add only 2*log(2) "
        "to raw PyMC logp. Observed and never-observed inclusion/occupancy "
        "mixtures match, including the J-fold zero-site likelihood for each "
        "never-observed species; no latent species are discarded.",
    ),
    "rats_model": (
        (
            "alpha",
            "beta",
            "mu_alpha",
            "mu_beta",
            "sigma_y",
            "sigma_alpha",
            "sigma_beta",
        ),
        "models/posteriordb_rats_model_rats_data.py",
        "Keep centered alpha and beta in rat order. PyMC rat_index=rat-1 and "
        "centered_x=x-xbar reproduce the Stan observation predictor. Positive "
        "sigma_y, sigma_alpha and sigma_beta have the same improper flat "
        "kernels as HalfFlat, not proper half-Normal priors. Independent "
        "Normal alpha/beta hierarchy and hypermeans are unchanged.",
    ),
    "seeds_centered_model": (
        ("alpha0", "alpha1", "alpha12", "alpha2", "c", "sigma"),
        "models/posteriordb_seeds_centered_model_seeds_data.py",
        "Retain all I unconstrained c values. In both sources b=c-mean(c) "
        "is deterministic; it does not replace c with a lower-dimensional "
        "zero-sum random variable or introduce an intrinsic improper prior. "
        "Coefficient Normal(0,1) priors and Binomial logits match. PyMC "
        "HalfCauchy(1) differs from Stan's positive Cauchy only by +log(2).",
    ),
    "seeds_model": (
        ("alpha0", "alpha1", "alpha12", "alpha2", "tau", "b"),
        "models/posteriordb_seeds_model_seeds_data.py",
        "Keep tau as precision with Gamma(shape=0.001,rate=0.001) and all I "
        "centered b effects. Both sources set sigma=tau**(-1/2) deterministically, "
        "so no precision/SD coordinate Jacobian applies. Normal(0,1000) "
        "coefficient priors and Binomial logits use identical x1,x2 interactions.",
    ),
    "seeds_stanified_model": (
        ("alpha0", "alpha1", "alpha12", "alpha2", "b", "sigma"),
        "models/posteriordb_seeds_stanified_model_seeds_data.py",
        "All I b effects remain unconstrained centered Normal(0,sigma) draws, "
        "without subtracting their mean. Normal(0,1) coefficient priors and "
        "Binomial logits match. Positive Cauchy versus HalfCauchy(1) changes "
        "only the constant +log(2), not the sigma-dependent kernel.",
    ),
    "surgical_model": (
        ("mu", "sigmasq", "b"),
        "models/posteriordb_surgical_model_surgical_data.py",
        "Keep variance sigmasq with InverseGamma(shape=0.001,scale=0.001). "
        "sigma=sqrt(sigmasq) and p=sigmoid(b) are deterministics, so no "
        "variance/SD or logit/probability Jacobian applies. Both sources use "
        "the same Normal(mu,sigma) b effects and Binomial-logit observations.",
    ),
}

MODEL_NAMES: frozenset[str] = frozenset(_MODELS)


def stan_center(name: str, data: dict[str, object]) -> dict[str, np.ndarray]:
    """Provide a support-feasible center for Stan's unbounded dogs_log beta."""
    if name == "dogs_log":
        return {"beta": np.array([-0.5, 0.5])}
    return {}


def map_parameters(
    name: str, data: dict[str, object], stan: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, float], list[str]]:
    """Map original constrained Stan parameters without using engine residuals."""
    parameters, module, correspondence = _MODELS[name]
    if name == "multi_occupancy":
        values = {
            parameter: stan[parameter]
            for parameter in parameters
            if parameter not in {"correlation_unit", "uv"}
        }
        values["correlation_unit"] = np.asarray((stan["rho_uv"] + 1) / 2)
        values["uv"] = np.column_stack((stan["uv1"], stan["uv2"]))
    else:
        values = {parameter: stan[parameter] for parameter in parameters}
    notes = [
        f"Sources: pinned stan/{name}.stan and {module}::build_model.",
        correspondence,
        "No nonconstant density adjustment is required by these source definitions; "
        "parameter-independent constants are omitted from two-point deltas.",
    ]
    return values, {}, notes
