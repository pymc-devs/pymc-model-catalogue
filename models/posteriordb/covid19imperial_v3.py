"""Flaxman et al. (2020) intervention renewal model, version 3."""

from ._data import load_data
from ._epidemic_renewal import _build_epidemic_renewal


def build_model(posterior: str | None = None):
    """Build the renewal model from one canonical posterior pair."""
    data = load_data("covid19imperial_v3", posterior)
    return _build_epidemic_renewal(data)


if __name__ == "__main__":
    from ._benchmark import main

    main("covid19imperial_v3")
