"""Persistent progress for a full timeline rerun, separate from ASV results."""

import hashlib
import json
import math
import os
import re
import subprocess
import uuid


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def benchmark_sources(root):
    paths = [root / "asv.conf.json", root / "scripts/core_models.py",
             root / "scripts/run_benchmarks.py"]
    for directory in ("asv_bench", "models", "models_discrete"):
        paths.extend((root / directory).rglob("*.py"))
    return [(str(path.relative_to(root)), path.read_text()) for path in sorted(paths)]


def benchmark_fingerprint(root):
    # Updating progress tracking does not change the measurements themselves.
    return digest([(path, source) for path, source in benchmark_sources(root)
                   if path != "asv_bench/_rerun.py"])


def matches_legacy_fingerprint(root, fingerprint):
    """Upgrade an old checkpoint only if all other benchmark files still match.

    Old fingerprints included this module. Recover its historical source from
    Git and compare the complete fingerprint; never waive changes to the model,
    compiler, sampler, or environment configuration.
    """
    path = "asv_bench/_rerun.py"
    sources = benchmark_sources(root)
    try:
        commits = subprocess.check_output(
            ["git", "log", "--all", "--format=%H", "--", path], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
        for commit in commits:
            source = subprocess.check_output(
                ["git", "show", f"{commit}:{path}"], cwd=root, text=True,
                stderr=subprocess.DEVNULL,
            )
            if digest([(name, source if name == path else content)
                       for name, content in sources]) == fingerprint:
                return True
    except subprocess.CalledProcessError:
        pass
    return False


def successful(row):
    values = row.get("result")
    return bool(values) and all(isinstance(v, (int, float)) and math.isfinite(v)
                                for v in values)


class RerunCheckpoint:
    def __init__(self, results_dir, bench, root, *, rerun=False, rerun_id=None):
        self.results_dir = results_dir
        # No .json extension: ASV otherwise tries to read this as a result file.
        self.path = results_dir / "rerun-state"
        self.bench = bench
        self.state = json.loads(self.path.read_text()) if self.path.exists() else None
        fingerprint = benchmark_fingerprint(root)
        compatible = (self.state and self.state["bench"] == bench
                      and self.state["fingerprint"] == fingerprint)
        if (self.state and self.state["bench"] == bench and not compatible
                and matches_legacy_fingerprint(root, self.state["fingerprint"])):
            compatible = True
            self.state["fingerprint"] = fingerprint
            print("Upgrading progress tracking; benchmark code is unchanged", flush=True)
        if self.state and rerun_id == self.state["id"] and not compatible:
            raise RuntimeError("Campaign uses different benchmark code; choose a new rerun_id")
        if self.state and self.state["status"] == "active" and not compatible and not rerun:
            raise RuntimeError("Pending rerun uses different benchmark code; start with rerun=true")
        resume = compatible and (
            self.state["id"] == rerun_id if rerun_id else self.state["status"] == "active"
        )
        if rerun and not resume:
            known_environments = {}
            if self.state:
                known_environments.update(self.state.get("known_environments", {}))
                known_environments.update(self.state["environments"])
            self.state = {
                "version": 1,
                "id": rerun_id or os.environ.get("GITHUB_RUN_ID") or uuid.uuid4().hex,
                "status": "active", "bench": bench, "fingerprint": fingerprint,
                "known_environments": known_environments,
                "environments": {}, "completed": {},
            }
            self.save()
            print(f"Starting rerun {self.state['id']}", flush=True)
        elif resume:
            self.state["status"] = "active"
            self.save()
            print(f"Resuming rerun {self.state['id']}", flush=True)
        else:
            self.state = None

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.path)

    def environments(self, commit, resolve):
        if self.state is None:
            return resolve()
        environments = self.state["environments"]
        if commit not in environments:
            known = self.state.setdefault("known_environments", {})
            environments[commit] = known[commit] if commit in known else resolve()
            known[commit] = environments[commit]
            self.save()
        return environments[commit]

    def rows(self, commit, python):
        path = self.results_dir / "ci" / f"{commit[:8]}-virtualenv-py{python}.json"
        if not path.exists():
            return {}
        data = json.loads(path.read_text())
        return {name: {key: value for key, value in zip(data["result_columns"], row)
                       if value is not None}
                for name, row in data["results"].items()}

    def metadata(self):
        path = self.results_dir / "benchmarks.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def pending(self, commit, python, regex):
        metadata = self.metadata()
        names = sorted(name for name in metadata if re.search(regex, name))
        if not names:
            return None  # Let ASV discover the suite when metadata is unavailable.
        rows = self.rows(commit, python)
        completed = self.state["completed"].get(f"{commit}/{python}", {})
        # Membership proves this campaign measured the metric successfully.
        # ASV rewrites durations/statistics at lower precision when saving other
        # models, so a hash of the entire saved row is not a stable identity.
        return [name for name in names if not completed.get(name)
                or not successful(rows.get(name, {}))
                or rows[name].get("version") != metadata[name].get("version")]

    def record(self, commit, python, regex, before):
        metadata = self.metadata()
        completed = self.state["completed"].setdefault(f"{commit}/{python}", {})
        for name, row in self.rows(commit, python).items():
            if not re.search(regex, name):
                continue
            if (successful(row)
                    and row.get("version") == metadata.get(name, {}).get("version")
                    and digest(row) != digest(before.get(name))):
                completed[name] = True
        self.save()

    def finish(self):
        if self.state is None:
            return
        complete = all(
            self.pending(commit, python, self.bench) == []
            for commit, pythons in self.state["environments"].items()
            for python in pythons
        )
        self.state["status"] = "complete" if complete else "active"
        self.save()
        print(f"Rerun {self.state['id']}: {self.state['status']}", flush=True)
