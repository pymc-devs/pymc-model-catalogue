"""Render measured docstring sections without modifying model source files."""

import argparse
import json
from pathlib import Path
from textwrap import wrap


def comparison_text(record, metadata):
    lines = ["Stan comparison:"]
    if record["status"] == "no-stan-reference":
        lines.append(
            "- Not applicable: this discrete PyMC-only definition has no Stan reference."
        )
        return "\n".join(lines)
    if record["status"] == "error":
        lines.append("- Measurement incomplete; no numerical agreement is claimed.")
        return "\n".join(lines)
    a, b = record["points"]["a"], record["points"]["b"]
    packages = metadata["packages"]
    lines.extend(
        [
            f"- BridgeStan {packages['bridgestan']}, PyMC {packages['pymc']}; delta = logp(B) - logp(A).",
            "- Constrained densities; automatic support-transform Jacobians disabled in both.",
            f"- A: Stan = {a['stan_logp']!r}; PyMC = {a['pymc_logp']!r}.",
            f"- B: Stan = {b['stan_logp']!r}; PyMC = {b['pymc_logp']!r}.",
            f"- Raw delta: Stan = {record['delta_stan']!r}; PyMC = {record['delta_pymc']!r}.",
            f"- Raw comparison: {record['raw_status'].upper()}; residual = {record['raw_residual']!r}.",
        ]
    )
    for name in sorted(a["adjustments"]):
        lines.extend(
            wrap(
                f"- Additive Stan adjustment {name}: A = {a['adjustments'][name]!r}; B = {b['adjustments'][name]!r}.",
                width=88,
                subsequent_indent="  ",
            )
        )
    if a["adjustments"]:
        lines.extend(
            [
                f"- Adjusted Stan delta = {record['adjusted_delta_stan']!r}.",
                f"- Adjusted comparison: {record['status'].upper()}; residual = {record['residual']!r}.",
            ]
        )
    lines.append(
        f"- Tolerance: 1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC)) = {record['tolerance']!r}"
    )
    if a["adjustments"]:
        lines.append("  using adjusted delta_Stan.")
    for note in record["notes"]:
        if not note.startswith("Sources:"):
            lines.extend(wrap("- " + note, width=88, subsequent_indent="  "))
    lines.extend(
        [
            f"- Original Stan SHA-256: {record['stan_sha256']}",
            f"- Saved points: models/posteriordb_logp/{record['point_file']}",
            "- Two-point evidence only; no global equivalence, support or inference claim.",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.evidence / "metadata.json").read_text())
    records = json.loads((args.evidence / "results.json").read_text())
    print(
        json.dumps(
            {record["module"]: comparison_text(record, metadata) for record in records},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
