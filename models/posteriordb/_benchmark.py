"""Run one model with the shared benchmark helper."""

from __future__ import annotations

import argparse
from importlib import import_module

from ._data import get_model, get_posterior, list_posteriors


def main(model_name: str) -> None:
    """Select a posterior and use its declared benchmark mode."""
    metadata = get_model(model_name)
    parser = argparse.ArgumentParser(
        description=f"Benchmark the native PyMC {model_name} model with bundled data."
    )
    parser.add_argument(
        "--posterior",
        default=metadata["default_posterior"],
        help="Canonical posterior identifier. The default uses the smallest archive.",
    )
    parser.add_argument(
        "--list-posteriors",
        action="store_true",
        help="List posterior identifiers for this model without model construction.",
    )
    args = parser.parse_args()
    if args.list_posteriors:
        print("\n".join(list_posteriors(model_name)))
        return
    try:
        record = get_posterior(args.posterior)
    except KeyError as error:
        parser.error(str(error))
    if record["model_name"] != model_name:
        parser.error(
            f"Posterior {args.posterior!r} belongs to model {record['model_name']!r}, "
            f"not {model_name!r}. Use --list-posteriors for valid identifiers."
        )

    from models._benchmark import run_benchmark

    builder = import_module(record["module"]).build_model
    prepared = builder(posterior=record["name"])
    model, _ = prepared
    discrete = any(
        variable.dtype.startswith(("int", "uint", "bool"))
        for variable in model.value_vars
    )

    def build_selected():
        nonlocal prepared
        if prepared is not None:
            result = prepared
            prepared = None
            return result
        return builder(posterior=record["name"])

    mode = record["benchmark_mode"]
    print(f"Posterior: {record['name']}")
    print(f"Dataset: {record['data_name']}")
    print(f"Backend: {mode}")
    print(
        f"Gradients: {'disabled for discrete free variables' if discrete else 'enabled'}"
    )
    if mode == "NUMBA":
        print("NUMBA first evaluation includes JIT compilation.")
    else:
        print(
            "FAST_RUN first evaluation measures initial evaluation, not JIT compilation."
        )
    if record["benchmark_notes"]:
        print(record["benchmark_notes"])
    run_benchmark(build_selected, discrete=discrete, mode=mode)
