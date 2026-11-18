"""Source-derived constrained-coordinate mappings for the ecology comparisons.

The original Stan parameters, not generated quantities or deterministics, define
these points. None of these 23 source pairs has a nonconstant changed probability
factor. Parameter-independent normalization and coordinate constants are disclosed
in the notes and deliberately omitted from the delta adjustments.
"""

import numpy as np


# Each entry lists PyMC free RVs, the exact source module, and the notes on genuine
# differences from Stan (empty when none). The Stan source is stan/<model name>.stan
# in the pinned bundle.
_MODELS = {
    "M0_model": (
        ("omega", "p"),
        "models/posteriordb_M0_model_M0_data.py",
        (
            "PyMC marginalizes inclusion with ZeroInflatedBinomial over each "
            "individual's row sum, where Stan evaluates the same row sums in "
            "its binomial.",
        ),
    ),
    "Mb_model": (
        ("omega", "p", "c"),
        "models/posteriordb_Mb_model_Mb_data.py",
        (),
    ),
    "Mh_model": (
        ("omega", "mean_p", "sigma", "eps_raw"),
        "models/posteriordb_Mh_model_Mh_data.py",
        (
            "PyMC Uniform(0,5) for sigma adds only -log(5) relative to Stan's "
            "implicit bounded-flat prior.",
        ),
    ),
    "Mt_model": (
        ("omega", "p"),
        "models/posteriordb_Mt_model_Mt_data.py",
        (),
    ),
    "Mtbh_model": (
        ("omega", "mean_p", "gamma", "sigma", "eps_raw"),
        "models/posteriordb_Mtbh_model_Mtbh_data.py",
        ("PyMC's sigma Uniform(0,3) contributes only the constant -log(3).",),
    ),
    "Mth_model": (
        ("omega", "mean_p", "sigma", "eps_raw"),
        "models/posteriordb_Mth_model_Mth_data.py",
        ("PyMC's sigma Uniform(0,5) contributes only the constant -log(5).",),
    ),
    "Rate_1_model": (
        ("theta",),
        "models/posteriordb_Rate_1_model_Rate_1_data.py",
        (),
    ),
    "Rate_2_model": (
        ("theta1", "theta2"),
        "models/posteriordb_Rate_2_model_Rate_2_data.py",
        (),
    ),
    "Rate_3_model": (
        ("theta",),
        "models/posteriordb_Rate_3_model_Rate_3_data.py",
        (),
    ),
    "Rate_4_model": (
        ("theta", "thetaprior"),
        "models/posteriordb_Rate_4_model_Rate_4_data.py",
        (),
    ),
    "Rate_5_model": (
        ("theta",),
        "models/posteriordb_Rate_5_model_Rate_5_data.py",
        (),
    ),
    "bones_model": (
        ("theta",),
        "models/posteriordb_bones_model_bones_data.py",
        (
            "Stan's categories 1..ncat become PyMC categories 0..ncat-1. PyMC "
            "OrderedLogistic eta=delta*theta and cutpoints delta*gamma "
            "reproduce Stan's differences of inv_logit(delta*(theta-gamma)).",
        ),
    ),
    "dogs": (
        ("beta",),
        "models/posteriordb_dogs_dogs.py",
        (),
    ),
    "dogs_hierarchical": (
        ("a", "b"),
        "models/posteriordb_dogs_hierarchical_dogs.py",
        (
            "The first trial has probability one; every bundled first response "
            "is one. PyMC preserves that support factor separately as first_shock.",
        ),
    ),
    "dogs_log": (
        ("beta",),
        "models/posteriordb_dogs_log_dogs.py",
        (
            "Stan declares beta unconstrained, but the Uniform(-100,0) and "
            "Uniform(0,100) prior supports still apply. The feasible center "
            "[-0.5,0.5] is chosen solely from support, not engine agreement.",
        ),
    ),
    "dugongs_model": (
        ("alpha", "beta", "lambda", "tau"),
        "models/posteriordb_dugongs_model_dugongs_data.py",
        (),
    ),
    "lsat_model": (
        ("alpha", "theta", "beta"),
        "models/posteriordb_lsat_model_lsat_data.py",
        (
            "Stan's item/person matrix is transposed relative to PyMC's "
            "person/item likelihood, without permuting theta. PyMC "
            "HalfNormal(100) for positive beta differs from Stan's constrained "
            "Normal(0,100) only by the constant +log(2).",
        ),
    ),
    "multi_occupancy": (
        ("alpha", "beta", "Omega", "sigma_uv", "correlation_unit", "uv"),
        "models/posteriordb_multi_occupancy_butterfly.py",
        (
            "Map correlation_unit=(rho_uv+1)/2 and uv=column_stack(uv1,uv2). "
            "The PyMC Cholesky factor induces the covariance with diagonals "
            "sigma_uv**2 and off-diagonal sigma_uv[0]*sigma_uv[1]*rho_uv. "
            "Stan already evaluates Beta(2,2) at (rho_uv+1)/2 without a Jacobian; "
            "its value equals PyMC's Beta density at correlation_unit. The "
            "coordinate determinant |d rho_uv/d correlation_unit|=2 is constant "
            "(log(2)); normalizing Stan's rho prior removes the same constant. "
            "The two positive Cauchy-to-HalfCauchy normalizations add only 2*log(2) "
            "to raw PyMC logp.",
        ),
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
        ("PyMC indexes rats with rat_index=rat-1 (zero-based).",),
    ),
    "seeds_centered_model": (
        ("alpha0", "alpha1", "alpha12", "alpha2", "c", "sigma"),
        "models/posteriordb_seeds_centered_model_seeds_data.py",
        ("PyMC HalfCauchy(1) differs from Stan's positive Cauchy only by +log(2).",),
    ),
    "seeds_model": (
        ("alpha0", "alpha1", "alpha12", "alpha2", "tau", "b"),
        "models/posteriordb_seeds_model_seeds_data.py",
        (),
    ),
    "seeds_stanified_model": (
        ("alpha0", "alpha1", "alpha12", "alpha2", "b", "sigma"),
        "models/posteriordb_seeds_stanified_model_seeds_data.py",
        (
            "Positive Cauchy versus HalfCauchy(1) changes only the constant "
            "+log(2), not the sigma-dependent kernel.",
        ),
    ),
    "surgical_model": (
        ("mu", "sigmasq", "b"),
        "models/posteriordb_surgical_model_surgical_data.py",
        (),
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
    parameters, module, differences = _MODELS[name]
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
        *differences,
    ]
    return values, {}, notes
