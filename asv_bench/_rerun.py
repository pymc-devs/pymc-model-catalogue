"""Persistent progress for a full timeline rerun, separate from ASV results."""

import hashlib
import json
import os
import re
import subprocess
import uuid

from asv_bench._model_results import rows, valid_result


def benchmark_fingerprint(root, *, legacy=False):
    paths = [root / "asv.conf.json", root / "scripts/core_models.py",
             root / "scripts/run_benchmarks.py"]
    for directory in ("asv_bench", "models", "models_discrete"):
        paths.extend((root / directory).rglob("*.py"))
    checkpoint_path = root / "asv_bench/_rerun.py"
    sources = {path: path.read_text() for path in paths if path != checkpoint_path}
    if legacy:
        # The original format included the checkpoint code in the fingerprint.
        # Use its original revision to verify the rest of the sources exactly.
        previous = subprocess.run(
            ["git", "show", "0678ce47:asv_bench/_rerun.py"],
            cwd=root, capture_output=True, text=True,
        )
        if previous.returncode:
            return None
        sources[checkpoint_path] = previous.stdout
    inputs = [(str(path.relative_to(root)), sources[path]) for path in sorted(sources)]
    return hashlib.sha256(json.dumps(inputs).encode()).hexdigest()


class RerunCheckpoint:
    def __init__(self, results_dir, bench, root, *, rerun=False, rerun_id=None):
        self.results_dir = results_dir
        # No .json extension: ASV otherwise tries to read this as a result file.
        self.path = results_dir / "rerun-state"
        self.bench = bench
        saved = json.loads(self.path.read_text()) if self.path.exists() else {}
        fingerprint = benchmark_fingerprint(root)
        resume = saved and (saved["id"] == rerun_id if rerun_id else saved["status"] == "active")
        if resume:
            resume = saved["bench"] == bench and (
                saved["fingerprint"] == fingerprint
                or saved["fingerprint"] == benchmark_fingerprint(root, legacy=True)
            )
            if not resume and (rerun_id or not rerun):
                raise RuntimeError("Campaign uses different benchmark code; choose a new rerun_id")
        if resume:
            self.state = saved
            self.state.update(status="active", fingerprint=fingerprint)
            action = "Resuming"
        elif rerun:
            self.state = {
                "version": 1,
                "id": rerun_id or os.environ.get("GITHUB_RUN_ID") or uuid.uuid4().hex,
                "status": "active", "bench": bench, "fingerprint": fingerprint,
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
        names = sorted(name for name in metadata if re.search(regex, name))
        if not names:
            return None  # Let ASV discover the suite when metadata is unavailable.
        rows = self.rows(commit, python)
        completed = self.state["completed"].get(f"{commit}/{python}", {})
        # Membership proves this campaign measured the metric successfully.
        # ASV rewrites durations/statistics at lower precision when saving other
        # models, so a hash of the entire saved row is not a stable identity.
        return [name for name in names if not completed.get(name)
                or not valid_result(rows.get(name, {}))
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
