"""
Model: Success rate of a trial with prior and posterior inference
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/Rate_4_model.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/Rate_4_model.py
Authors: M D Lee, E Wagenmakers (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/Rate_4_model.py
Upstream catalogue contributor: Kane Lindsay
Description: Inference of the success rate of a trial with prior and posterior inference

Data: Rate_4_data (upstream posterior: Rate_4_data-Rate_4_model)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/Rate_4_data.json.zip
    Archive SHA-256: d3a1b76b173776148bec6f56f20dda0e14aeb89ea2383548b7e0041fe2b74c41
    Local archive: data/posteriordb_Rate_4_data.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/Rate_4_data-Rate_4_model.json
References: lee2013cognitive (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/example-models/tree/master/Bayesian_Cognitive_Modeling/ParameterEstimation/Binomial

Implementation notes:
The independent comparison rate retains its unconditioned Beta distribution.

Source: Lee and Wagenmakers (2013), binomial rate examples.
Predictive counts do not add free variables to the probability model.

Stan comparison:
- A: Stan = -7.016003499815746; PyMC = -7.016003499815746.
- B: Stan = -7.579023772270681; PyMC = -7.57902377227068.
- Raw delta: Stan = -0.5630202724549349; PyMC = -0.563020272454934.
- Raw comparison: MATCH; residual = -8.881784197001252e-16.
- Tolerance: 1.0056302027245493e-06
- Original Stan SHA-256: 7aaf395c9ce53028fceaf8719538c3e8c98eba9884e1b3b843b33d636487d88c

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
            Path(__file__).parent / "data" / "posteriordb_Rate_4_data.json.zip"
        ) as archive,
        archive.open("Rate_4_data.json") as stream,
    ):
        data = json.load(stream)
    with pm.Model() as model:
        theta = pm.Beta("theta", alpha=1, beta=1)
        pm.Beta("thetaprior", alpha=1, beta=1)
        pm.Binomial("k", n=data["n"], p=theta, observed=data["k"])
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
