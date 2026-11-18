"""
Model: LDA with 2 topics
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/ldaK2.stan
Authors: Blei, David M and Ng, Andrew Y and Jordan, Michael I (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/ldaK2.py
Upstream catalogue contributor: Mans Magnusson
Description: Latent Dirichlet Allocation (LDA) with token indicators marginalized out and 2 topics.

Data: three_docs1200 (upstream posterior: three_docs1200-ldaK2)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/three_docs1200.json.zip
    Archive SHA-256: 1ac0f7efdc3143f6c11bb9daac6fccdaa8b4057edc40ddf5b7a70170bc1e6fd5
    Local archive: data/posteriordb_three_docs1200.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/three_docs1200-ldaK2.json
References: blei2003latent (see posteriordb/source/bibliography/references.bib)
Reference URL: https://mc-stan.org/docs/2_18/stan-users-guide/latent-dirichlet-allocation.html

Implementation notes:
Latent Dirichlet allocation with two topics and uniform simplex densities.

Source: Blei, Ng, and Jordan (2003), as specified by posteriordb.

Stan comparison:
- A: Stan = -1316.1619373814558; PyMC = -1316.161937381456.
- B: Stan = -1315.32528153365; PyMC = -1315.32528153365.
- Raw delta: Stan = 0.8366558478057868; PyMC = 0.8366558478060142.
- Raw comparison: MATCH; residual = -2.2737367544323206e-13.
- Tolerance: 1.00836655847806e-06
- Document and word indices shift from one-based to zero-based.
- Original Stan SHA-256: 2b943b45446d68148c94c9089ee17f45597eb839e0cd4f94410a817f787327d5

Benchmark results:
- Original: logp = -1323.4223, grad norm = 40.0000, 86.0 us/call (5 evals).
- Frozen: logp = -1323.4223, grad norm = 40.0000, 71.3 us/call (5 evals).
- Local NUMBA validation; short timings are indicative.
- Environment: PyMC 6.4.0+9.gaf565aa28, PyTensor 3.3.8+30.g9a7f24ce0, Numba 0.68.0.
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
