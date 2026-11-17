"""
Model: Random Effects Model to Rank Hospitals on True Failure Probability
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/surgical_model.stan
Authors: MRC Biostatistics Unit (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/surgical_model.py
Upstream catalogue contributor: Kane Lindsay
Description: Random effects model assuming the probabilities of failure over hospitals are similar
    in some way.

Data: surgical_data (upstream posterior: surgical_data-surgical_model)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/surgical_data.json.zip
    Archive SHA-256: cc197d58349660d879519eab3a065d37d50412c1cb362978bba5d6d7124636a8
    Local archive: data/posteriordb_surgical_data.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/surgical_data-surgical_model.json
References: mrc1winbugs (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/example-models/tree/master/bugs_examples/vol1/surgical

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Native hospital mortality model from posteriordb surgical_model.

Benchmark results:
- Not measured.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_surgical_data.json.zip"
        ) as archive,
        archive.open("surgical_data.json") as stream,
    ):
        data = json.load(stream)
    with pm.Model(coords={"hospital": np.arange(1, data["N"] + 1)}) as model:
        mu = pm.Normal("mu", 0, 1000)
        sigmasq = pm.InverseGamma("sigmasq", alpha=0.001, beta=0.001, initval=1.0)
        sigma = pm.Deterministic("sigma", pm.math.sqrt(sigmasq))
        b = pm.Normal("b", mu, sigma, dims="hospital")
        pm.Binomial("r", n=data["n"], logit_p=b, observed=data["r"], dims="hospital")
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
