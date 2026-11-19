"""
Model: LDA with 2 topics
Source: posteriordb/posterior_database/models/stan/ldaK2.stan
Authors: Blei, David M and Ng, Andrew Y and Jordan, Michael I (cited work)
Description: Latent Dirichlet Allocation (LDA) with token indicators marginalized out and 2 topics.

Stan comparison:
- A: Stan = -1316.1619373814558; PyMC = -1316.161937381456.
- B: Stan = -1315.32528153365; PyMC = -1315.32528153365.
- Raw delta: Stan = 0.8366558478057868; PyMC = 0.8366558478060142.
- Raw comparison: MATCH; residual = -2.2737367544323206e-13.
- Tolerance: 1.00836655847806e-06

Benchmark results:
- Original:  logp = -1323.4223, grad norm = 40.0000, 86.0 us/call (5 evals).
- Frozen:    logp = -1323.4223, grad norm = 40.0000, 71.3 us/call (5 evals).
- Local NUMBA validation; short timings are indicative.
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
            Path(__file__).parent / "data" / "posteriordb_three_docs1200.json.zip"
        ) as archive,
        archive.open("three_docs1200.json") as stream,
    ):
        data = json.load(stream)
    topics = 2
    alpha = np.ones(2)
    beta = np.ones(data["V"])
    words = np.asarray(data["w"], dtype="int64") - 1
    documents = np.asarray(data["doc"], dtype="int64") - 1
    coords = {
        "document": range(data["M"]),
        "topic": range(topics),
        "word": range(data["V"]),
        "token": range(data["N"]),
    }
    with pm.Model(coords=coords) as model:
        word = pm.Data("word_index", words, dims="token")
        document = pm.Data("document_index", documents, dims="token")
        theta = pm.Dirichlet(
            "theta", a=np.asarray(alpha, dtype="float64"), dims=("document", "topic")
        )
        phi = pm.Dirichlet(
            "phi", a=np.asarray(beta, dtype="float64"), dims=("topic", "word")
        )
        # Sum over topics without a token-by-vocabulary probability tensor.
        token_topic_logp = pt.log(theta[document]) + pt.log(phi[:, word].T)
        pm.Potential("word_likelihood", pt.logsumexp(token_topic_logp, axis=-1).sum())
    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
