"""Latent Dirichlet allocation with five topics and dataset concentration parameters.

Source: Blei, Ng, and Jordan (2003), as specified by posteriordb.
"""

from ._data import load_data
from ._topic_neural_lda import build_topic_neural_lda


def build_model(posterior: str | None = None):
    """Build every published model and data pair with five topics."""
    data = load_data("ldaK5", posterior)
    return build_topic_neural_lda(data, 5, data["alpha"], data["beta"])


if __name__ == "__main__":
    from ._benchmark import main

    main("ldaK5")
