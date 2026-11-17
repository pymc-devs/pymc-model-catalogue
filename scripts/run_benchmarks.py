"""Compile once with fresh caches, then reuse them for all other ASV metrics.

Pass the model-class filter with --bench and ordinary `asv run` arguments
after it. For timeline runs, --commit SHA reuses each saved Python environment;
new releases use Python 3.12 before PyMC 6 and Python 3.14 thereafter.
"""

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asv_bench._cache import cache_environment  # noqa: E402


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


def run(bench, arguments):
    failed = False
    with TemporaryDirectory(prefix="pymc-bench-") as directory:
        env = cache_environment(directory)
        phases = [
            ("Cold compilation", rf"(?:{bench})\.track_compile_time$"),
            ("Other metrics (cached)", rf"(?:{bench})\.(?!track_compile_time$)(?:track_\w+|time_\w+)$"),
        ]
        for label, regex in phases:
            print(f"== {label}", flush=True)
            result = subprocess.run(["asv", "run", *arguments, "--bench", regex], env=env)
            failed |= result.returncode != 0
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", required=True, help="ASV model-class regex")
    parser.add_argument("--commit", help="Full timeline commit SHA")
    parser.add_argument("--results-dir", type=Path, default=Path(".asv/results"))
    args, arguments = parser.parse_known_args()
    if not args.commit:
        return run(args.bench, arguments)
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        parser.error("--commit must be a full commit SHA")
    failed = False
    for python in saved_pythons(args.commit, args.results_dir):
        print(f"== {args.commit[:10]} / Python {python}", flush=True)
        failed |= bool(run(args.bench, [*arguments, "--python", python, f"{args.commit}^!"]))
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
