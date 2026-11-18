"""
Model: Common Rate of Success From Two Trials with Posterior Predictives
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/Rate_5_model.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/Rate_5_model.py
Authors: M D Lee, E Wagenmakers (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/Rate_5_model.py
Upstream catalogue contributor: Kane Lindsay
Description: Inference of a common rate of success from two trials with posterior predictive for
    each trial for checking the adequacy of the model.

Data: Rate_5_data (upstream posterior: Rate_5_data-Rate_5_model)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/Rate_5_data.json.zip
    Archive SHA-256: ad52821b8361c34b147784137eddc949dd27e65a677b81b276fc6145c592a48d
    Local archive: data/posteriordb_Rate_5_data.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/Rate_5_data-Rate_5_model.json
References: lee2013cognitive (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/example-models/tree/master/Bayesian_Cognitive_Modeling/ParameterEstimation/Binomial

Implementation notes:
Two count observations share one success rate.

Source: Lee and Wagenmakers (2013), binomial rate examples.
Predictive counts do not add free variables to the probability model.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -13.865169993120485; PyMC = -13.865169993120485.
- B: Stan = -13.87246475492653; PyMC = -13.87246475492653.
- Raw delta: Stan = -0.007294761806043937; PyMC = -0.007294761806043937.
- Raw comparison: MATCH; residual = 0.0.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 1.0000729476180604e-06
- Original Stan SHA-256: b86503076ff82e0c5a4e183d6274aa1f3f08b867a5de854adfada0d368f5c544
- Saved points: models/posteriordb_logp/points/Rate_5_data-Rate_5_model.npz
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
            Path(__file__).parent / "data" / "posteriordb_Rate_5_data.json.zip"
        ) as archive,
        archive.open("Rate_5_data.json") as stream,
    ):
        data = json.load(stream)
    with pm.Model() as model:
        theta = pm.Beta("theta", alpha=1, beta=1)
        pm.Binomial("k1", n=data["n1"], p=theta, observed=data["k1"])
        pm.Binomial("k2", n=data["n2"], p=theta, observed=data["k2"])
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
