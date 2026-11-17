"""Replace saved single-call core eval timings; preserve all other measurements.

Run after restoring gh-pages/results to .asv/results. --dry-run lists the plan.
Only tagged releases are eligible, so maintenance commits are never measured.
"""

import argparse
import json
import math
from pathlib import Path
import re
import subprocess

from core_models import CORE_MODELS


EVAL_NAMES = {
    f"bench_models.ModelBenchEval_{model.replace('.', '_')}.time_eval"
    for model in CORE_MODELS
}


def rows(data):
    return {
        name: {column: value for column, value in zip(data["result_columns"], row)
               if value is not None}
        for name, row in data["results"].items()
    }


def value_of(row):
    values = row.get("result")
    if not values or len(values) != 1:
        return None
    value = values[0]
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def plan(results_dir, site_dir):
    index = json.loads((site_dir / "index.json").read_text())
    releases = {
        index["revision_to_hash"][str(rev)]: tag.removeprefix("pymc-")
        for tag, rev in index["tags"].items()
        if re.fullmatch(r"pymc-\d+\.\d+\.0", tag)
        and str(rev) in index["revision_to_hash"]
    }
    targets = []
    for path in sorted((results_dir / "ci").glob("*.json")):
        data = json.loads(path.read_text())
        sha = data.get("commit_hash")
        if sha not in releases:
            continue
        names = sorted(
            name for name, row in rows(data).items()
            if name in EVAL_NAMES and row.get("stats_number") == [1]
            and value_of(row) is not None and 0 <= value_of(row) < 0.1
        )
        if not names:
            continue
        python = data["params"]["python"]
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError(f"Invalid commit in {path}")
        if python not in ("3.12", "3.14") or data["env_name"] != f"virtualenv-py{python}":
            raise ValueError(f"Refresh workflow needs support for environment in {path}")
        targets.append((releases[sha], python, sha, path, names))
    return sorted(targets, key=lambda t: (tuple(map(int, t[0].split("."))), t[1]))


def refresh(target):
    version, python, sha, path, names = target
    original = path.read_bytes()
    before = rows(json.loads(original))
    regex = "^(?:" + "|".join(re.escape(name) for name in names) + ")$"
    try:
        subprocess.run([
            "asv", "run", "--show-stderr", "--launch-method=spawn", "--record-samples",
            "--machine", "ci", "--python", python, "--bench", regex, f"{sha}^!",
        ], check=True)
        after = rows(json.loads(path.read_bytes()))
        for name, row in before.items():
            if name not in names and after.get(name) != row:
                raise RuntimeError(f"ASV changed an untargeted measurement: {name}")
        numbers = []
        for name in names:
            row = after.get(name, {})
            count = row.get("stats_number", [0])[0]
            if value_of(row) is None or count <= 1:
                raise RuntimeError(f"No successful batched replacement for {name}")
            numbers.append(count)
        print(f"Verified PyMC {version}, Python {python}: {len(names)} evals; "
              f"{min(numbers)}–{max(numbers)} calls/sample", flush=True)
    except BaseException:
        # A failed ASV run can replace values with null. Keep the old record
        # locally too; the workflow only publishes after all targets succeed.
        path.write_bytes(original)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(".asv/results"))
    parser.add_argument("--site-dir", type=Path, default=Path(".gh-pages"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    targets = plan(args.results_dir, args.site_dir)
    for version, python, _, _, names in targets:
        print(f"PyMC {version}, Python {python}: {len(names)} evals", flush=True)
    print(f"Total: {sum(len(t[4]) for t in targets)} eval timings in {len(targets)} records", flush=True)
    if not args.dry_run:
        for target in targets:
            refresh(target)


if __name__ == "__main__":
    main()
