"""
Model: LDA with 5 topics
Source: posteriordb/posterior_database/models/stan/ldaK5.stan
Authors: David M Blei, Andrew Y Ng, Michael I Jordan
Description: Latent Dirichlet Allocation (LDA) with token indicators marginalized out.

Stan comparison:
- A: Stan = -205189.15496980018; PyMC = -205189.15496980195.
- B: Stan = -205072.78648196885; PyMC = -205072.78648197.
- Raw delta: Stan = 116.36848783132154; PyMC = 116.36848783196183.
- Raw comparison: MATCH; residual = -6.402842700481415e-10.
- Tolerance: 2.1636848783196183e-06
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
            Path(__file__).parent
            / "data"
            / "posteriordb_prideprejudice_chapter.json.zip"
        ) as archive,
        archive.open("prideprejudice_chapter.json") as stream,
    ):
        data = json.load(stream)
    topics = 5
    alpha = data["alpha"]
    beta = data["beta"]
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
