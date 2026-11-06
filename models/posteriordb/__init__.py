"""Native PyMC models with offline posteriordb data and source metadata."""

from importlib import import_module

from ._data import (
    benchmark_mode,
    catalogue_metadata,
    get_dataset,
    get_model,
    get_posterior,
    list_datasets,
    list_models,
    list_posteriors,
    load_data,
)

__all__ = [
    "benchmark_mode",
    "build_model",
    "catalogue_metadata",
    "get_dataset",
    "get_model",
    "get_posterior",
    "list_datasets",
    "list_models",
    "list_posteriors",
    "load_data",
]


def build_model(posterior: str):
    """Build one native PyMC model through a lazy module import."""
    record = get_posterior(posterior)
    module = import_module(record["module"])
    return module.build_model(posterior=posterior)
