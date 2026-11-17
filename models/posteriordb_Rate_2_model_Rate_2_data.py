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

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Two independent success rates and their difference.

Source: Lee and Wagenmakers (2013), binomial rate examples.

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
