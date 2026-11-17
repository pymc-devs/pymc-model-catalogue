"""Warm the compiler, then build each model once and collect all ASV metrics.

Pass the model-class filter with --bench and ordinary `asv run` arguments
after it. For timeline runs, --commit SHA reuses each saved Python environment;
new releases use Python 3.12 before PyMC 6 and Python 3.14 thereafter.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asv_bench._cache import MODEL_BENCHMARK_VERSION, cache_environment  # noqa: E402
from asv_bench._measure import metric_names  # noqa: E402
from asv_bench._model_results import import_reports, rows, snapshot  # noqa: E402
from asv_bench._rerun import RerunCheckpoint  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SKIP_FLAGS = {"--skip-existing-successful", "--skip-existing-failed",
              "--skip-existing-commits", "--skip-existing", "-k"}


def option(arguments, name):
    for index, argument in enumerate(arguments):
        if argument == name:
            return arguments[index + 1]
        if argument.startswith(name + "="):
            return argument.split("=", 1)[1]
    return None


def model_selection(regex):
    for directory in ("models", "models_discrete"):
        for path in sorted((ROOT / directory).glob("*.py")):
            if path.name.startswith("_"):
                continue
            model = f"{directory}.{path.stem}"
            selected = {name for name in metric_names(model).values() if re.search(regex, name)}
            if selected:
                yield model, selected


def successful(row):
    values = row.get("result")
    return (row.get("version") == MODEL_BENCHMARK_VERSION and values
            and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values))


def saved_pythons(commit, results_dir):
    pythons = set()
    for path in (results_dir / "ci").glob(f"{commit[:8]}-*.json"):
        data = json.loads(path.read_text())
        if data.get("commit_hash") != commit:
            continue
        column = data["result_columns"].index("result")
        if not any(
            isinstance(value, (int, float)) and math.isfinite(value)
            for row in data["results"].values()
            for value in (row[column] or [])
        ):
            # Failed attempts on an incompatible Python are not history to replay.
            continue
        python = data["params"]["python"]
        if python not in ("3.12", "3.14") or data["env_name"] != f"virtualenv-py{python}":
            raise ValueError(f"Timeline workflow needs support for environment in {path}")
        pythons.add(python)
    if not pythons:
        ledger = json.loads(subprocess.check_output([
            "git", "show", f"{commit}:versions/tracked_versions.json",
        ], text=True))
        if not ledger["minor_releases"]:
            return []
        version = max(tuple(map(int, row["pymc"].split(".")))
                      for row in ledger["minor_releases"])
        pythons.add("3.12" if version < (6, 0, 0) else "3.14")
    return sorted(pythons)


def run(bench, arguments, *, results_dir, checkpoint=None, commit=None, python=None, publish=None):
    if not commit:
        reference = option(arguments, "--set-commit-hash")
        if not reference:
            raise ValueError("Use --commit or --set-commit-hash to identify saved results")
        commit = subprocess.check_output(["git", "rev-parse", reference], text=True).strip()
    machine = option(arguments, "--machine")
    dry_run = "--dry-run" in arguments or "-n" in arguments
    if dry_run and (publish or (checkpoint and checkpoint.state is not None)):
        raise ValueError("Dry runs cannot publish or update rerun checkpoints")
    failed = False
    models = list(model_selection(bench))
    if not models:
        raise ValueError(f"No model metrics match {bench!r}")
    for model, selected in models:
        before = snapshot(results_dir)
        regex = "^(?:" + "|".join(re.escape(name) for name in sorted(selected)) + ")$"
        active = checkpoint and checkpoint.state is not None
        if active:
            pending = checkpoint.pending(commit, python, regex)
            if pending is not None:
                selected.intersection_update(pending)
        elif SKIP_FLAGS.intersection(arguments) and python:
            records = [data for data in before.values()
                       if data["commit_hash"] == commit and data["params"]["python"] == python
                       and (not machine or data["params"]["machine"] == machine)]
            skip_all = "--skip-existing" in arguments or "-k" in arguments
            skip_success = skip_all or "--skip-existing-successful" in arguments
            skip_failed = skip_all or "--skip-existing-failed" in arguments

            def skip(data, name):
                row = rows(data).get(name, {})
                return (row.get("version") == MODEL_BENCHMARK_VERSION
                        and (skip_success if successful(row) else skip_failed))

            selected = {name for name in selected if not records
                        or not all(skip(data, name) for data in records)}
            if records and "--skip-existing-commits" in arguments:
                selected.clear()
        if not selected:
            print(f"== {model}: already measured", flush=True)
            continue
        regex = "^(?:" + "|".join(re.escape(name) for name in sorted(selected)) + ")$"
        previous = checkpoint.rows(commit, python) if active else None
        with TemporaryDirectory(prefix="pymc-bench-") as directory:
            reports = Path(directory) / "measurements"
            reports.mkdir()
            env = cache_environment(directory) | {
                "CATALOGUE_BUNDLE_DIR": str(reports), "CATALOGUE_COMMIT": commit,
            }
            # ASV runs just this carrier. Its worker measures every selected metric
            # using one target build, even on PyTensor versions without a disk cache.
            carrier = metric_names(model)["compile_time"]
            remaining = [arg for arg in arguments if arg not in SKIP_FLAGS]
            print(f"== {model}: one build, then eval samples", flush=True)
            result = subprocess.run(["asv", "run", *remaining,
                                     "--bench", "^" + re.escape(carrier) + "$"], env=env)
            failed |= result.returncode != 0
            if dry_run:
                continue
            failed |= import_reports(reports, results_dir, model, selected, before, machine)
        if active:
            checkpoint.record(commit, python, regex, previous)
        if publish:
            subprocess.run(["bash", str(publish)], check=True,
                           env=os.environ | {"BENCHMARK_SHA": commit})
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", required=True, help="ASV model-class regex")
    parser.add_argument("--commit", help="Full timeline commit SHA")
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--rerun", action="store_true", help="Start or resume a full rerun")
    parser.add_argument("--rerun-id", help="Optional campaign name; change it to start fresh")
    parser.add_argument("--finish-rerun", action="store_true")
    parser.add_argument("--checkpoint-script", type=Path)
    args, arguments = parser.parse_known_args()
    config = Path(option(arguments, "--config") or "asv.conf.json").resolve()
    if args.results_dir is None:
        settings = json.loads(config.read_text())
        args.results_dir = config.parent / settings.get("results_dir", ".asv/results")
    if not args.commit and not args.finish_rerun:
        if args.rerun or args.rerun_id:
            parser.error("Rerun checkpoints require --commit")
        return run(args.bench, arguments, results_dir=args.results_dir)
    if args.commit and not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        parser.error("--commit must be a full commit SHA")
    checkpoint = RerunCheckpoint(
        args.results_dir, args.bench, Path(__file__).resolve().parents[1],
        rerun=args.rerun or bool(args.rerun_id), rerun_id=args.rerun_id,
    )
    if args.finish_rerun:
        checkpoint.finish()
        return 0
    failed = False
    # Persist a newly started campaign even if this job times out in its first model.
    if checkpoint.state is not None and args.checkpoint_script:
        subprocess.run(["bash", str(args.checkpoint_script)], check=True,
                       env=os.environ | {"BENCHMARK_SHA": args.commit})
    for python in checkpoint.environments(args.commit, lambda: saved_pythons(args.commit, args.results_dir)):
        print(f"== {args.commit[:10]} / Python {python}", flush=True)
        failed |= bool(run(
            args.bench, [*arguments, "--python", python, f"{args.commit}^!"],
            results_dir=args.results_dir, checkpoint=checkpoint, commit=args.commit, python=python,
            publish=args.checkpoint_script,
        ))
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
