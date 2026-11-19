"""
Model: LDA with 2 topics
Source: posteriordb/posterior_database/models/stan/ldaK2.stan
Authors: David M Blei, Andrew Y Ng, Michael I Jordan
Description: Latent Dirichlet Allocation (LDA) with token indicators marginalized out and 2 topics.

Stan comparison:
- A: Stan = -25362.424083896924; PyMC = -25362.42408389695.
- B: Stan = -25345.140012056734; PyMC = -25345.14001205677.
- Raw delta: Stan = 17.284071840189426; PyMC = 17.284071840178513.
- Raw comparison: MATCH; residual = 1.0913936421275139e-11.
- Tolerance: 1.1728407184018942e-06
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
            Path(__file__).parent / "data" / "posteriordb_three_men1.json.zip"
        ) as archive,
        archive.open("three_men1.json") as stream,
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
