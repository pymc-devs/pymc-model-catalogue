"""Measure original Stan and unchanged native PyMC densities at saved points.

Run from the repository root with ``python -m scripts.posteriordb_comparison``.
Compilation and evaluation are separate phases; neither performs sampling.
"""

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import traceback
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
ATOL = 1e-6
RTOL = 1e-8
SEED = 20261117


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def inventory(root):
    records = []
    for path in sorted(root.glob("models*/posteriordb_*.py")):
        tree = ast.parse(path.read_text())
        doc = ast.get_docstring(tree)
        posterior = re.search(r"upstream posterior: ([^)]+)", doc).group(1)
        archive = path.parent / re.search(r"Local archive: (\S+)", doc).group(1)
        source = re.search(
            r"https://github.com/stan-dev/posteriordb/blob/([^/]+)/"
            r"posterior_database/models/stan/([^\s]+)\.stan",
            doc,
        )
        tree.body = tree.body[1:]
        record = {
            "module": str(path.relative_to(root)),
            "posterior": posterior,
            "archive": str(archive.relative_to(root)),
            "archive_sha256": _sha256(archive),
            "builder_ast_sha256": hashlib.sha256(
                ast.dump(tree, include_attributes=False).encode()
            ).hexdigest(),
            "stan_model": source.group(2) if source else None,
            "stan_url": source.group(0).replace(
                "github.com/stan-dev/posteriordb/blob/",
                "raw.githubusercontent.com/stan-dev/posteriordb/",
            )
            if source
            else None,
        }
        records.append(record)
    return records


def _compile_one(record, cache, bridge, make_args, output):
    name = record["stan_model"]
    original = cache / "stan" / f"{name}.stan"
    original.parent.mkdir(parents=True, exist_ok=True)
    if not original.exists():
        with urlopen(record["stan_url"], timeout=60) as response:
            original.write_bytes(response.read())
    build = cache / "build"
    build.mkdir(parents=True, exist_ok=True)
    source = build / f"pdb_{name}.stan"
    if not source.exists():
        shutil.copyfile(original, source)
    library = build / f"pdb_{name}_model.so"
    command = ["make", "-j1", str(library), *make_args]
    result = {
        "stan_model": name,
        "stan_url": record["stan_url"],
        "stan_sha256": _sha256(original),
        "compiled_source_sha256": _sha256(source),
        "command": command,
        "library": str(library),
    }
    try:
        if result["stan_sha256"] != result["compiled_source_sha256"]:
            raise ValueError(
                "Compiled source differs from pinned original; explicit disclosure required"
            )
        process = subprocess.run(
            command, cwd=bridge, text=True, capture_output=True, timeout=900
        )
        log = process.stdout + process.stderr
        (output / "compile_logs" / f"{name}.txt").write_text(log)
        if process.returncode:
            raise RuntimeError(
                f"make exited {process.returncode}; see compile_logs/{name}.txt"
            )
        result.update(status="compiled", library_sha256=_sha256(library))
    except Exception:
        result.update(status="error", error=traceback.format_exc())
    return result


def _unpack(names, flat):
    """Recover logical indices, independently of Stan's serialization order."""
    import numpy as np

    entries = {}
    for name, value in zip(names, flat, strict=True):
        head, *tail = name.split(".")
        index = tuple(int(part) - 1 for part in tail)
        entries.setdefault(head, []).append((index, value))
    values = {}
    for name, items in entries.items():
        if not items[0][0]:
            values[name] = np.asarray(items[0][1])
        else:
            shape = tuple(
                max(index[axis] for index, _ in items) + 1
                for axis in range(len(items[0][0]))
            )
            array = np.empty(shape)
            if math.prod(shape) != len(items):
                raise ValueError(f"Incomplete serialized parameter {name}")
            for index, value in items:
                array[index] = value
            values[name] = array
    return values


def _mapping(name):
    from . import dynamic, ecology, regression, special

    matches = [
        module
        for module in (dynamic, ecology, regression, special)
        if name in module.MODEL_NAMES
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one mapping for {name}, got {len(matches)}")
    return matches[0]


def _evaluate(record, cache, output, cxx):
    # Set before importing either numerical engine, including in spawned workers.
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = "1"
    result = dict(record)
    try:
        import bridgestan
        import numpy as np
        import pytensor
        from pymc.model.transform.conditioning import remove_value_transforms
        from pytensor.compile.mode import Mode

        name = record["stan_model"]
        if name is None:
            result.update(
                status="no-stan-reference",
                notes=["PyMC-only discrete M0 model; no Stan values exist."],
            )
            return result
        original = cache / "stan" / f"{name}.stan"
        source = cache / "build" / f"pdb_{name}.stan"
        library = cache / "build" / f"pdb_{name}_model.so"
        if _sha256(original) != _sha256(source):
            raise ValueError("Compiled source differs from pinned original")
        result.update(stan_sha256=_sha256(original), library_sha256=_sha256(library))
        with ZipFile(ROOT / record["archive"]) as archive:
            members = [
                member for member in archive.namelist() if member.endswith(".json")
            ]
            if len(members) != 1:
                raise ValueError("Expected exactly one JSON data member")
            data = json.loads(archive.read(members[0]))
        stan_model = bridgestan.StanModel(library, data=data, seed=SEED)
        result["stan_model_info"] = stan_model.model_info()
        mapping = _mapping(name)
        names = stan_model.param_names()
        q_center = np.zeros(stan_model.param_unc_num())
        center = _unpack(names, stan_model.param_constrain(q_center))
        overrides = (
            mapping.stan_center(name, data) if hasattr(mapping, "stan_center") else {}
        )
        if overrides:
            center.update(overrides)
            q_center = stan_model.param_unconstrain_json(center)
        spec = importlib.util.spec_from_file_location(
            "comparison_model", ROOT / record["module"]
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        mode = Mode(linker="cvm", optimizer="fast_run")
        flags = {"mode": "FAST_RUN", "linker": "cvm"}
        if cxx:
            flags["cxx"] = cxx
        with pytensor.config.change_flags(**flags):
            native, _ = module.build_model()
            native = remove_value_transforms(native)
            native_logp = native.compile_logp(jacobian=False, mode=mode)
            result["stan_parameter_names"] = names
            result["stan_unconstrained_parameter_names"] = stan_model.param_unc_names()
            result["pymc_parameter_names"] = [rv.name for rv in native.free_RVs]
            result["pymc_value_names"] = {
                rv.name: native.rvs_to_values[rv].name for rv in native.free_RVs
            }
            result["center_overrides"] = {
                key: np.asarray(value).tolist() for key, value in overrides.items()
            }
            result["points"] = {}
            rejected = []
            arrays = {}
            for label in ("a", "b"):
                for attempt in range(8):
                    seed = int.from_bytes(
                        hashlib.sha256(
                            f"{SEED}/{record['posterior']}/{label}/{attempt}".encode()
                        ).digest()[:8],
                        "little",
                    )
                    q = q_center + np.random.default_rng(seed).normal(
                        0, 0.15, q_center.shape
                    )
                    try:
                        stan_values = _unpack(names, stan_model.param_constrain(q))
                        mapped, adjustments, notes = mapping.map_parameters(
                            name, data, stan_values
                        )
                        native_values = {
                            native.rvs_to_values[rv].name: np.asarray(
                                mapped[rv.name], dtype=native.rvs_to_values[rv].dtype
                            )
                            for rv in native.free_RVs
                        }
                        stan_lp = float(
                            stan_model.log_density(q, propto=False, jacobian=False)
                        )
                        native_lp = float(native_logp(native_values))
                        if not all(
                            math.isfinite(value)
                            for value in [stan_lp, native_lp, *adjustments.values()]
                        ):
                            raise ValueError(
                                f"Nonfinite density/adjustment: Stan={stan_lp}, PyMC={native_lp}, adjustments={adjustments}"
                            )
                    except (RuntimeError, ValueError, FloatingPointError) as error:
                        rejected.append(
                            {
                                "point": label,
                                "attempt": attempt,
                                "seed": seed,
                                "reason": str(error),
                            }
                        )
                        continue
                    result["points"][label] = {
                        "seed": seed,
                        "attempt": attempt,
                        "stan_logp": stan_lp,
                        "pymc_logp": native_lp,
                        "adjustments": adjustments,
                    }
                    arrays[f"{label}__stan_unconstrained"] = q
                    arrays.update(
                        {
                            f"{label}__stan__{key}": value
                            for key, value in stan_values.items()
                        }
                    )
                    arrays.update(
                        {
                            f"{label}__pymc__{key}": value
                            for key, value in native_values.items()
                        }
                    )
                    result["notes"] = notes
                    break
                else:
                    result["rejected_candidates"] = rejected
                    raise RuntimeError(
                        f"No finite candidate for point {label} after eight attempts"
                    )
            if np.any(
                arrays["a__stan_unconstrained"] == arrays["b__stan_unconstrained"]
            ):
                raise ValueError(
                    "The two selected points did not vary every unconstrained coordinate"
                )
            result["rejected_candidates"] = rejected
            point_path = output / "points" / f"{record['posterior']}.npz"
            np.savez_compressed(point_path, **arrays)
            result.update(
                point_file=str(point_path.relative_to(output)),
                point_sha256=_sha256(point_path),
            )
            a, b = result["points"]["a"], result["points"]["b"]
            delta_stan = b["stan_logp"] - a["stan_logp"]
            delta_pymc = b["pymc_logp"] - a["pymc_logp"]
            delta_adjustment = math.fsum(
                [
                    *b["adjustments"].values(),
                    *(-value for value in a["adjustments"].values()),
                ]
            )
            adjusted = delta_stan + delta_adjustment
            tolerance = ATOL + RTOL * max(abs(adjusted), abs(delta_pymc))
            raw_tolerance = ATOL + RTOL * max(abs(delta_stan), abs(delta_pymc))
            result.update(
                status="match"
                if abs(adjusted - delta_pymc) <= tolerance
                else "mismatch",
                raw_status="match"
                if abs(delta_stan - delta_pymc) <= raw_tolerance
                else "mismatch",
                delta_stan=delta_stan,
                delta_pymc=delta_pymc,
                delta_adjustment=delta_adjustment,
                adjusted_delta_stan=adjusted,
                residual=adjusted - delta_pymc,
                tolerance=tolerance,
                raw_residual=delta_stan - delta_pymc,
                raw_tolerance=raw_tolerance,
            )
    except Exception:
        result.update(status="error", error=traceback.format_exc())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("compile", "evaluate"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bridgestan", type=Path)
    parser.add_argument("--cxx")
    parser.add_argument("--cc")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument(
        "--model",
        action="append",
        help="Exact Stan model name; repeat to select several",
    )
    parser.add_argument(
        "--posterior",
        action="append",
        help="Exact posterior ID; repeat to select several",
    )
    args = parser.parse_args()
    args.cache = args.cache.resolve()
    args.output = args.output.resolve()
    records = inventory(ROOT)
    if args.model:
        records = [record for record in records if record["stan_model"] in args.model]
    if args.posterior:
        records = [
            record for record in records if record["posterior"] in args.posterior
        ]
    if not records:
        parser.error("No matching posterior pairs")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "compile":
        if args.bridgestan is None:
            parser.error("--bridgestan source directory is required for compilation")
        (args.output / "compile_logs").mkdir(exist_ok=True)
        make_args = ["O=1", "CXX_TYPE=gcc", "TBB_CXX_TYPE=gcc"]
        if args.cxx:
            make_args.append(f"CXX={args.cxx}")
        if args.cc:
            make_args.append(f"CC={args.cc}")
        unique = {
            record["stan_model"]: record for record in records if record["stan_model"]
        }
        work = list(unique.values())
        results = []
        # Initialize shared BridgeStan/TBB/Sundials targets before concurrent make.
        if work:
            results.append(
                _compile_one(
                    work.pop(0), args.cache, args.bridgestan, make_args, args.output
                )
            )
            print(results[-1]["stan_model"], results[-1]["status"], flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    _compile_one,
                    record,
                    args.cache,
                    args.bridgestan,
                    make_args,
                    args.output,
                )
                for record in work
            ]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(result["stan_model"], result["status"], flush=True)
        _write_json(args.output / "compilation.json", results)
    else:
        import pytensor

        pymc_directory = Path(importlib.util.find_spec("pymc").origin).parent
        (args.output / "points").mkdir(exist_ok=True)
        metadata = {
            "protocol": "Constrained densities; propto=False; both automatic Jacobians disabled; delta=B-A",
            "point_rule": "PCG64 seeded by SHA256 of 20261117/posterior/label/attempt; q=center+Normal(0,0.15); at most eight feasibility-only candidates",
            "atol": ATOL,
            "rtol": RTOL,
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("pymc", "pytensor", "numpy", "scipy", "bridgestan")
            },
            "pytensor_config": {
                "cast_policy": pytensor.config.cast_policy,
                "floatX": pytensor.config.floatX,
            },
            "engine_sources_sha256": {
                f"pymc/{relative}": _sha256(pymc_directory / relative)
                for relative in (
                    "distributions/continuous.py",
                    "distributions/multivariate.py",
                )
            },
            "harness_sha256": {
                str(path.relative_to(ROOT)): _sha256(path)
                for path in sorted(Path(__file__).parent.glob("*.py"))
            },
            "workers": args.workers,
            "numerical_threads": 1,
            "cxx": args.cxx,
        }
        _write_json(args.output / "metadata.json", metadata)
        results = []
        with ProcessPoolExecutor(
            max_workers=args.workers, max_tasks_per_child=1
        ) as pool:
            futures = [
                pool.submit(_evaluate, record, args.cache, args.output, args.cxx)
                for record in records
            ]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    result["posterior"],
                    result["status"],
                    result.get("residual", ""),
                    flush=True,
                )
        _write_json(
            args.output / "results.json",
            sorted(results, key=lambda result: result["posterior"]),
        )
    if any(result["status"] == "error" for result in results):
        raise SystemExit(1)
