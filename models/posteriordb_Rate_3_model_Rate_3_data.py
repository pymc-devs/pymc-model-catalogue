"""
Model: Common rate of success from two trials
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/Rate_3_model.stan
Authors: M D Lee, E Wagenmakers (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/Rate_3_model.py
Upstream catalogue contributor: Kane Lindsay
Description: Inference of a common rate of success from two trials

Data: Rate_3_data (upstream posterior: Rate_3_data-Rate_3_model)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/Rate_3_data.json.zip
    Archive SHA-256: e5d36f46170ad99b0df4b63d3f71e7c2bf082b92b1b879c8c8abe5dcc53d86a3
    Local archive: data/posteriordb_Rate_3_data.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/Rate_3_data-Rate_3_model.json
References: lee2013cognitive (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/example-models/tree/master/Bayesian_Cognitive_Modeling/ParameterEstimation/Binomial

Implementation notes:
Two count observations share one success rate.

Source: Lee and Wagenmakers (2013), binomial rate examples.

Stan comparison:
- A: Stan = -3.756980106360462; PyMC = -3.756980106360465.
- B: Stan = -3.556700839794817; PyMC = -3.5567008397948205.
- Raw delta: Stan = 0.2002792665656452; PyMC = 0.20027926656564432.
- Raw comparison: MATCH; residual = 8.881784197001252e-16.
- Tolerance: 1.0020027926656564e-06
- Original Stan SHA-256: 416ea0e377796973a67eddfa956d93e4e5f2204d96889b7b55432275b5e4e0d2

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
            Path(__file__).parent / "data" / "posteriordb_Rate_3_data.json.zip"
        ) as archive,
        archive.open("Rate_3_data.json") as stream,
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
