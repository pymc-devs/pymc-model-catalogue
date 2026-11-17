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

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
One success rate with a uniform Beta distribution.

Source: Lee and Wagenmakers (2013), binomial rate examples.

Stan comparison:
- Not recorded.

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
