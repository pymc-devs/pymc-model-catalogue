"""Offline data and source metadata for native PyMC models."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from importlib.resources import files
import json
from typing import Any
from zipfile import ZipFile


@lru_cache(maxsize=1)
def _catalogue() -> dict[str, Any]:
    """Return the private source registry without model imports."""
    resource = files(__package__).joinpath("catalogue.json")
    with resource.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _model_record(model_name: str) -> dict[str, Any]:
    try:
        return _catalogue()["models"][model_name]
    except KeyError:
        raise KeyError(f"Unknown posteriordb model: {model_name!r}") from None


def _posterior_record(posterior: str) -> dict[str, Any]:
    try:
        return _catalogue()["posteriors"][posterior]
    except KeyError:
        raise KeyError(f"Unknown posteriordb posterior: {posterior!r}") from None


def list_models() -> tuple[str, ...]:
    """List canonical model identifiers."""
    return tuple(sorted(_catalogue()["models"]))


def list_datasets() -> tuple[str, ...]:
    """List canonical dataset identifiers."""
    return tuple(sorted(_catalogue()["datasets"]))


def list_posteriors(model_name: str | None = None) -> tuple[str, ...]:
    """List canonical posterior identifiers for one model or all models."""
    if model_name is None:
        return tuple(sorted(_catalogue()["posteriors"]))
    return tuple(_model_record(model_name)["posteriors"])


def get_model(model_name: str) -> dict[str, Any]:
    """Return independent source metadata for a model."""
    return deepcopy(_model_record(model_name))


def get_dataset(data_name: str) -> dict[str, Any]:
    """Return independent source metadata for a dataset."""
    try:
        return deepcopy(_catalogue()["datasets"][data_name])
    except KeyError:
        raise KeyError(f"Unknown posteriordb dataset: {data_name!r}") from None


def get_posterior(posterior: str) -> dict[str, Any]:
    """Return posterior metadata and its authoritative workload identity.

    The archive path is relative to ``models/posteriordb``.
    """
    record = _posterior_record(posterior)
    model = _model_record(record["model_name"])
    dataset = _catalogue()["datasets"][record["data_name"]]
    result = deepcopy(record)
    result.update(
        module=model["module"],
        benchmark_mode=model["benchmark_mode"],
        benchmark_notes=model["benchmark_notes"],
        data_archive=dataset["archive"],
        data_sha256=dataset["sha256"],
    )
    return result


def benchmark_mode(model_name: str) -> str:
    """Return the declared PyTensor benchmark mode."""
    return _model_record(model_name)["benchmark_mode"]


def catalogue_metadata() -> dict[str, Any]:
    """Return the complete source registry without model imports."""
    return deepcopy(_catalogue())


def load_data(model_name: str, posterior: str | None = None) -> dict[str, Any]:
    """Load an original processed JSON dictionary from a bundled archive."""
    model = _model_record(model_name)
    selected = model["default_posterior"] if posterior is None else posterior
    record = _posterior_record(selected)
    if record["model_name"] != model_name:
        raise ValueError(
            f"Posterior {selected!r} belongs to model {record['model_name']!r}, "
            f"not {model_name!r}. Select a posterior from list_posteriors({model_name!r})."
        )
    dataset = _catalogue()["datasets"][record["data_name"]]
    resource = files(__package__).joinpath(dataset["archive"])
    with resource.open("rb") as stream, ZipFile(stream) as archive:
        members = [
            member
            for member in archive.infolist()
            if not member.is_dir() and member.filename.endswith(".json")
        ]
        if len(members) != 1:
            raise ValueError(
                f"Bundled dataset {record['data_name']!r} must contain one JSON member."
            )
        with archive.open(members[0]) as data_stream:
            data = json.load(data_stream)
    if not isinstance(data, dict):
        raise ValueError(
            f"Bundled dataset {record['data_name']!r} is not a JSON object."
        )
    return data
