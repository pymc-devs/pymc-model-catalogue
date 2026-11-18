"""
Model: Difference in success rates
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/Rate_2_model.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/Rate_2_model.py
Authors: M D Lee, E Wagenmakers (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/Rate_2_model.py
Upstream catalogue contributor: Kane Lindsay
Description: Predicting the difference between the success rates of two trials

Data: Rate_2_data (upstream posterior: Rate_2_data-Rate_2_model)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/Rate_2_data.json.zip
    Archive SHA-256: 92db117ef105d42727deead55593dbc8f0d34bed3640ec4bc51e4f45710ab1c7
    Local archive: data/posteriordb_Rate_2_data.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/Rate_2_data-Rate_2_model.json
References: lee2013cognitive (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/example-models/tree/master/Bayesian_Cognitive_Modeling/ParameterEstimation/Binomial

Implementation notes:
Two independent success rates and their difference.

Source: Lee and Wagenmakers (2013), binomial rate examples.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -3.079291541514772; PyMC = -3.079291541514775.
- B: Stan = -3.376244044403064; PyMC = -3.3762440444030664.
- Raw delta: Stan = -0.296952502888292; PyMC = -0.29695250288829156.
- Raw comparison: MATCH; residual = -4.440892098500626e-16.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 1.0029695250288828e-06
- Original Stan SHA-256: 2556299694d2e3420d6d100b5439376bed7f78454816d89cd189d09df0b5877d
- Saved points: models/posteriordb_logp/points/Rate_2_data-Rate_2_model.npz
- Two-point evidence only; no global equivalence, support or inference claim.

Benchmark results:
- Not measured.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_Rate_2_data.json.zip"
        ) as archive,
        archive.open("Rate_2_data.json") as stream,
    ):
        data = json.load(stream)
    with pm.Model() as model:
        theta1 = pm.Beta("theta1", alpha=1, beta=1)
        theta2 = pm.Beta("theta2", alpha=1, beta=1)
        pm.Binomial("k1", n=data["n1"], p=theta1, observed=data["k1"])
        pm.Binomial("k2", n=data["n2"], p=theta2, observed=data["k2"])
        pm.Deterministic("delta", theta1 - theta2)
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
