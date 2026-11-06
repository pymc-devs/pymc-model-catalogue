"""One-hidden-layer classification network with ten hidden units.

Source: Lampinen and Vehtari (2001), as specified by posteriordb.
"""

from ._data import load_data
from ._topic_neural_network import build_topic_neural_network


def build_model(posterior: str | None = None):
    """Build the centered ten-unit neural classification model."""
    data = load_data("nn_rbm1bJ10", posterior)
    return build_topic_neural_network(data, 10)


if __name__ == "__main__":
    from ._benchmark import main

    main("nn_rbm1bJ10")
