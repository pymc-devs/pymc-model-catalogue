"""
Model: Random Effects (Rasch) Model for True Difficulty of LSAT Questions
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/lsat_model.stan
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/pymc/lsat_model.py
Authors: MRC Biostatistics Unit (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/lsat_model.py
Upstream catalogue contributor: Kane Lindsay
Description: One-parameter 'Rasch Model' where the probability that a student responds correctly to
    a question is assumed to follow a logistic function parameterized by 'item difficulty' and a
    latent variable representing the students underlying ability.

Data: lsat_data (upstream posterior: lsat_data-lsat_model)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/lsat_data.json.zip
    Archive SHA-256: 63a2ba05000665a5adf348c009a069a3a9914a394a1637a589fbe18889800060
    Local archive: data/posteriordb_lsat_data.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/lsat_data-lsat_model.json
References: mrc1winbugs (see posteriordb/source/bibliography/references.bib)
Reference URL: https://github.com/stan-dev/example-models/tree/master/bugs_examples/vol1/lsat

Implementation notes:
Native LSAT item-response model from the BUGS example.

Cumulative pattern counts expand to individual student responses.

Stan comparison:
- A: Stan = -4649.540355460236; PyMC = -4648.847208663369.
- B: Stan = -4287.590745592309; PyMC = -4286.897598795441.
- Raw delta: Stan = 361.9496098679274; PyMC = 361.9496098679283.
- Raw comparison: MATCH; residual = -9.094947017729282e-13.
- Tolerance: 4.619496098679283e-06
- Stan's item/person matrix is transposed relative to PyMC's person/item likelihood,
  without permuting theta. PyMC HalfNormal(100) for positive beta differs from Stan's
  constrained Normal(0,100) only by the constant +log(2).
- Original Stan SHA-256: 9f3d0abccfb55fcab634d81a19d74507409b836c549224b923b237ecf9f2c817

Benchmark results:
- Not measured.
"""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model():
    with (
        ZipFile(
            Path(__file__).parent / "data" / "posteriordb_lsat_data.json.zip"
        ) as archive,
        archive.open("lsat_data.json") as stream,
    ):
        data = json.load(stream)
    cumulative = np.asarray(data["culm"], dtype="int64")
    counts = np.diff(np.concatenate(([0], cumulative)))
    responses = np.repeat(np.asarray(data["response"], dtype="int64"), counts, axis=0)
    coords = {
        "item": np.arange(1, data["T"] + 1),
        "person": np.arange(1, data["N"] + 1),
    }
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", mu=0, sigma=100, dims="item")
        theta = pm.Normal("theta", mu=0, sigma=1, dims="person")
        beta = pm.HalfNormal("beta", sigma=100, initval=1)
        pm.Bernoulli(
            "response",
            logit_p=beta * theta[:, None] - alpha[None, :],
            observed=responses,
            dims=("person", "item"),
        )
        pm.Deterministic("a", alpha - pt.mean(alpha), dims="item")
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
