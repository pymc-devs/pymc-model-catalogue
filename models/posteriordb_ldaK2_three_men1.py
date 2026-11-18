"""
Model: LDA with 2 topics
Source: posteriordb, revision 91027f9cdd13b7e76283fec3f2898f59d4b83ec4
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/models/stan/ldaK2.stan
Authors: Blei, David M and Ng, Andrew Y and Jordan, Michael I (cited work)
PyMC port source: pymc-model-catalogue commit a481aaf8be79760a413e1a7cc6fd9c04b55aad5e
    models/posteriordb/ldaK2.py
Upstream catalogue contributor: Mans Magnusson
Description: Latent Dirichlet Allocation (LDA) with token indicators marginalized out and 2 topics.

Data: three_men1 (upstream posterior: three_men1-ldaK2)
    https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/data/data/three_men1.json.zip
    Archive SHA-256: ffbd94deb93edfb92291a5681749b12345fa398e6a599cded70c1d9dbc684d5a
    Local archive: data/posteriordb_three_men1.json.zip
Metadata: https://github.com/stan-dev/posteriordb/blob/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database/posteriors/three_men1-ldaK2.json
References: blei2003latent (see posteriordb/source/bibliography/references.bib)
Reference URL: https://mc-stan.org/docs/2_18/stan-users-guide/latent-dirichlet-allocation.html

Changes from original:
- Native PyMC port; the upstream implementation paths above identify the source definitions.
- Bundled the processed source JSON archive unchanged.
- Captured the initial point before clearing model initial-value overrides.

Implementation notes:
Latent Dirichlet allocation with two topics and uniform simplex densities.

Source: Blei, Ng, and Jordan (2003), as specified by posteriordb.

Stan comparison:
- BridgeStan 2.9.0, PyMC 6.4.0; delta = logp(B) - logp(A).
- Constrained densities; automatic support-transform Jacobians disabled in both.
- A: Stan = -25362.424083896924; PyMC = -25362.42408389695.
- B: Stan = -25345.140012056734; PyMC = -25345.14001205677.
- Raw delta: Stan = 17.284071840189426; PyMC = 17.284071840178513.
- Raw comparison: MATCH; residual = 1.0913936421275139e-11.
- No parameter-dependent adjustment; only constant offsets may cancel.
- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = 1.1728407184018942e-06
  using adjusted delta_Stan when an adjustment is present.
- Identity of theta with shape (M,2) and phi with shape (2,V), each row a simplex.
  Priors are Dirichlet(ones). Each token contributes
  logsumexp_k(log(theta[doc,k])+log(phi[k,word])); topic indicators are marginalized
  identically, retaining token order and shifting only document/word indices to zero-
  based.
- Original Stan SHA-256: 2b943b45446d68148c94c9089ee17f45597eb839e0cd4f94410a817f787327d5
- Saved points: models/posteriordb_logp/points/three_men1-ldaK2.npz
- Full evidence: models/posteriordb_logp/records/three_men1-ldaK2.json
- Two-point evidence only; no global equivalence, support or inference claim.

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
