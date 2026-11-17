"""Compile once with fresh caches, then reuse them for all other ASV metrics.

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
from asv_bench._cache import cache_environment  # noqa: E402
from asv_bench._rerun import RerunCheckpoint  # noqa: E402


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


def run(bench, arguments, *, checkpoint=None, commit=None, python=None, publish=None):
    failed = False
    with TemporaryDirectory(prefix="pymc-bench-") as directory:
        env = cache_environment(directory)
        phases = [
            ("Cold compilation", rf"(?:{bench})\.track_compile_time$"),
            ("Other metrics (cached)", rf"(?:{bench})\.(?!track_compile_time$)(?:track_\w+|time_\w+)$"),
        ]
        for label, regex in phases:
            if checkpoint and checkpoint.state is not None:
                pending = checkpoint.pending(commit, python, regex)
                if pending == []:
                    print(f"== {label}: already checkpointed", flush=True)
                    continue
                if pending is not None:
                    regex = "^(?:" + "|".join(re.escape(name) for name in pending) + ")$"
                # Old successful results are not evidence of this campaign's progress.
                arguments = [arg for arg in arguments if arg not in {
                    "--skip-existing-successful", "--skip-existing-failed",
                    "--skip-existing-commits", "--skip-existing", "-k",
                }]
                before = checkpoint.rows(commit, python)
            print(f"== {label}", flush=True)
            result = subprocess.run(["asv", "run", *arguments, "--bench", regex], env=env)
            failed |= result.returncode != 0
            if checkpoint and checkpoint.state is not None:
                checkpoint.record(commit, python, regex, before)
            if publish:
                subprocess.run(["bash", str(publish)], check=True,
                               env=os.environ | {"BENCHMARK_SHA": commit})
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", required=True, help="ASV model-class regex")
    parser.add_argument("--commit", help="Full timeline commit SHA")
    parser.add_argument("--results-dir", type=Path, default=Path(".asv/results"))
    parser.add_argument("--rerun", action="store_true", help="Start or resume a full rerun")
    parser.add_argument("--rerun-id", help="Optional campaign name; change it to start fresh")
    parser.add_argument("--finish-rerun", action="store_true")
    parser.add_argument("--checkpoint-script", type=Path)
    args, arguments = parser.parse_known_args()
    if not args.commit and not args.finish_rerun:
        if args.rerun or args.rerun_id:
            parser.error("Rerun checkpoints require --commit")
        return run(args.bench, arguments)
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
    # Persist a newly started campaign even if this job times out in its first phase.
    if checkpoint.state is not None and args.checkpoint_script:
        subprocess.run(["bash", str(args.checkpoint_script)], check=True,
                       env=os.environ | {"BENCHMARK_SHA": args.commit})
    for python in checkpoint.environments(args.commit, lambda: saved_pythons(args.commit, args.results_dir)):
        print(f"== {args.commit[:10]} / Python {python}", flush=True)
        failed |= bool(run(
            args.bench, [*arguments, "--python", python, f"{args.commit}^!"],
            checkpoint=checkpoint, commit=args.commit, python=python,
            publish=args.checkpoint_script,
        ))
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
