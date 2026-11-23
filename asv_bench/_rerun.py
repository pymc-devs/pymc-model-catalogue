"""Persistent progress for a full timeline rerun, separate from ASV results."""

import hashlib
import json
import os
import re
import subprocess
import uuid
from pathlib import Path

from asv_bench._measure import metric_names
from asv_bench._model_results import rows, valid_result


# Preserve the active timeline campaign across the metadata-only changes to
# measurement reports and result imports. An exact pair still detects edits
# to any measurement code or model.
DEPENDENCY_METADATA_UPGRADE = (
    "6925d39fa2cfe48035a07028377121f7e7f1fe9d82b6e2315e3d318016aabf9b",
    "9de4f44cc52a6db2752034660210216a0949d2de1a1112e9a7ff08e6790cf534",
)


def benchmark_sources(root):
    paths = [root / "asv.conf.json", root / "scripts/core_models.py",
             root / "scripts/run_benchmarks.py"]
    for directory in ("asv_bench", "models", "models_discrete"):
        paths.extend((root / directory).rglob("*.py"))
    return {path: path.read_text() for path in paths if path != root / "asv_bench/_rerun.py"}


def fingerprint_sources(root, sources):
    inputs = [(str(path.relative_to(root)), sources[path]) for path in sorted(sources)]
    return hashlib.sha256(json.dumps(inputs).encode()).hexdigest()


def benchmark_fingerprint(root, *, legacy=False):
    sources = benchmark_sources(root)
    if legacy:
        # The original format included the checkpoint code in the fingerprint.
        # Use its original revision to verify the rest of the sources exactly.
        previous = subprocess.run(
            ["git", "show", "0678ce47:asv_bench/_rerun.py"],
            cwd=root, capture_output=True, text=True,
        )
        if previous.returncode:
            return None
        sources[root / "asv_bench/_rerun.py"] = previous.stdout
    return fingerprint_sources(root, sources)


def model_module(relative_path):
    path = Path(relative_path)
    if (len(path.parts) == 2 and path.parts[0] in ("models", "models_discrete")
            and path.suffix == ".py" and not path.name.startswith("_")):
        return f"{path.parts[0]}.{path.stem}"
    return None


def selected_metrics(bench, source_hashes):
    return {
        name
        for path in source_hashes
        if (model := model_module(path)) is not None
        for name in metric_names(model).values()
        if re.search(bench, name)
    }


def can_extend(saved, source_hashes, metrics):
    """Keep campaign progress when measurement sources and metrics only grow."""
    previous = saved.get("source_hashes")
    return (
        previous is not None
        and previous.items() <= source_hashes.items()
        and all(model_module(path) for path in source_hashes.keys() - previous.keys())
        and selected_metrics(saved["bench"], previous) <= metrics
    )


class RerunCheckpoint:
    def __init__(self, results_dir, bench, root, *, rerun=False, rerun_id=None):
        self.results_dir = results_dir
        # No .json extension: ASV otherwise tries to read this as a result file.
        self.path = results_dir / "rerun-state"
        self.bench = bench
        saved = json.loads(self.path.read_text()) if self.path.exists() else {}
        sources = benchmark_sources(root)
        fingerprint = fingerprint_sources(root, sources)
        # Selection changes are checked against the actual metric names below.
        source_hashes = {
            str(path.relative_to(root)): hashlib.sha256(text.encode()).hexdigest()
            for path, text in sources.items() if path != root / "scripts/core_models.py"
        }
        self.metrics = selected_metrics(bench, source_hashes)
        resume = saved and (saved["id"] == rerun_id if rerun_id else saved["status"] == "active")
        if resume:
            same_suite = saved["bench"] == bench and (
                saved["fingerprint"] == fingerprint
                or (saved["fingerprint"], fingerprint) == DEPENDENCY_METADATA_UPGRADE
                or saved["fingerprint"] == benchmark_fingerprint(root, legacy=True)
            )
            resume = same_suite or can_extend(saved, source_hashes, self.metrics)
            if not resume and rerun_id:
                raise RuntimeError("Campaign uses different benchmark code; choose a new rerun_id")
        if resume:
            self.state = saved
            self.state.update(status="active", bench=bench, fingerprint=fingerprint,
                              source_hashes=source_hashes)
            action = "Resuming"
        elif rerun:
            self.state = {
                "version": 1,
                "id": rerun_id or os.environ.get("GITHUB_RUN_ID") or uuid.uuid4().hex,
                "status": "active", "bench": bench, "fingerprint": fingerprint,
                "source_hashes": source_hashes,
                "known_environments": saved.get("known_environments", {}) | saved.get("environments", {}),
                "environments": {}, "completed": {},
            }
            action = "Starting"
        else:
            self.state = None
            return
        self.save()
        print(f"{action} rerun {self.state['id']}", flush=True)

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
        return rows(json.loads(path.read_text())) if path.exists() else {}

    def metadata(self):
        path = self.results_dir / "benchmarks.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def pending(self, commit, python, regex):
        metadata = self.metadata()
        expected = self.metrics | metadata.keys()
        names = sorted(name for name in expected if re.search(regex, name))
        if not names:
            return None  # Let ASV discover the suite when metadata is unavailable.
        rows = self.rows(commit, python)
        completed = self.state["completed"].get(f"{commit}/{python}", {})
        # Membership proves this campaign measured the metric successfully.
        # ASV rewrites durations/statistics at lower precision when saving other
        # models, so a hash of the entire saved row is not a stable identity.
        return [name for name in names if not completed.get(name)
                or not valid_result(rows.get(name, {}))
                or name not in metadata
                or rows[name].get("version") != metadata[name].get("version")]

    def record(self, commit, python, regex, before):
        metadata = self.metadata()
        completed = self.state["completed"].setdefault(f"{commit}/{python}", {})
        for name, row in self.rows(commit, python).items():
            if not re.search(regex, name):
                continue
            started_at = row.get("started_at")
            if (valid_result(row)
                    and row.get("version") == metadata.get(name, {}).get("version")
                    and started_at is not None
                    and started_at != before.get(name, {}).get("started_at")):
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
