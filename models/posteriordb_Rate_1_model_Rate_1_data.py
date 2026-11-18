"""
Model: Predicting the rate of correct test question answers
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/Rate_1_model.stan
Authors: M D Lee, E Wagenmakers (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/Rate_1_model.py
Upstream catalogue contributor: Kane Lindsay
Description: Predicting the rate of correct answers for one subject over questions with the same
    difficulty.

Data: Rate_1_data (upstream posterior: Rate_1_data-Rate_1_model)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/Rate_1_data.json.zip
    Archive SHA-256: de0330f774e9ee9bbe30d743202d2928292ce0cd8a40601b90b99c536221b4f5
    Local archive: data/posteriordb_Rate_1_data.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/Rate_1_data-Rate_1_model.json
References: lee2013cognitive (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/example-models/tree/master/Bayesian_Cognitive_Modeling/GettingStarted

Implementation notes:
One success rate with a uniform Beta distribution.

Source: Lee and Wagenmakers (2013), binomial rate examples.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -1.5191739432523104; PyMC = -1.519173943252311.
- B: Stan = -1.4755165919307762; PyMC = -1.4755165919307773.
- Raw delta: Stan = 0.043657351321534144; PyMC = 0.0436573513215337.
- Raw comparison: MATCH; residual = 4.440892098500626e-16.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 1.0004365735132153e-06
- Original Stan SHA-256: ae95dcc6c2e2953e202fb257181ee24e1cb1f7780819e95d94663dde4fddc795
- Saved points: models/posteriordb_logp/points/Rate_1_data-Rate_1_model.npz
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
            Path(__file__).parent / "data" / "posteriordb_Rate_1_data.json.zip"
        ) as archive,
        archive.open("Rate_1_data.json") as stream,
    ):
        data = json.load(stream)
    with pm.Model() as model:
        theta = pm.Beta("theta", alpha=1, beta=1)
        pm.Binomial("k", n=data["n"], p=theta, observed=data["k"])
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
