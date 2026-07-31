"""
Step 1 of the low-threat-collapse diagnosis session (AUDIT.md sec M).

Builds predicted-vs-expected confusion matrices for threat_level, likely_intent,
and recommended_action from the existing evaluation/eval_expanded_{system}.json
files (55-case battery, n_runs=5) -- no new generations.

LIMITATION (stated up front, not buried): per_case only stores majority_threat/
majority_intent/majority_action (the mode across the n_runs replicate generations
for that case), not each individual run's raw prediction -- evaluate_llm never
persisted that. So each case contributes exactly ONE vote (its majority label) to
the confusion matrix, not up to n_runs votes. This under-counts genuine
run-to-run disagreement (a case split 3/5 for "medium" and 2/5 for "high" shows up
as one "medium" vote, the "high" minority is invisible here) but is the most
faithful confusion matrix obtainable without re-running the battery.

Usage:
    python llm_finetuning/report_confusion_matrices.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from swarm_intent.llm.prompts import TEST_CASES  # noqa: E402

SYSTEMS = ["v2", "v3a", "v3a-nomask", "v3b"]
FIELDS = [
    ("threat_level", "expected_threat", "majority_threat", ["low", "medium", "high", "critical"]),
    ("likely_intent", "expected_intent", "majority_intent", None),
    ("recommended_action", "expected_action", "majority_action", None),
]

NAME_TO_CASE = {c["name"]: c for c in TEST_CASES}


def build_matrix(system: str, expected_key: str, predicted_key: str):
    data = json.loads((REPO / "evaluation" / f"eval_expanded_{system}.json").read_text())
    matrix = defaultdict(lambda: defaultdict(int))
    for case in data["per_case"]:
        gt = NAME_TO_CASE.get(case["name"])
        if gt is None or not case["has_ground_truth"]:
            continue
        expected = gt[expected_key]
        predicted = case[predicted_key]
        matrix[expected][predicted] += 1
    return matrix


def print_matrix(matrix, row_order, field_name, system):
    all_predicted = sorted({p for row in matrix.values() for p in row.keys()})
    cols = row_order if row_order else sorted(set(row_order or []) | set(all_predicted))
    if row_order is None:
        cols = sorted(set(matrix.keys()) | set(all_predicted))
        rows = cols
    else:
        rows = row_order
        cols = sorted(set(row_order) | set(all_predicted))
    print(f"\n--- {system}: {field_name} (rows=expected, cols=predicted) ---")
    header = "| expected \\ predicted | " + " | ".join(cols) + " | n |"
    print(header)
    print("|" + "---|" * (len(cols) + 2))
    for r in rows:
        row_counts = matrix.get(r, {})
        n = sum(row_counts.values())
        if n == 0:
            continue
        cells = [str(row_counts.get(c, 0)) for c in cols]
        print(f"| **{r}** | " + " | ".join(cells) + f" | {n} |")
    return rows, cols


def main():
    print("=== threat_level / likely_intent / recommended_action confusion matrices "
          "(majority-vote label per case, 55-case battery, n_runs=5) ===")
    print("LIMITATION: one vote per case (the case's majority label across 5 runs), "
          "not one vote per run -- see module docstring.\n")

    all_matrices = {}
    for system in SYSTEMS:
        all_matrices[system] = {}
        for field_name, expected_key, predicted_key, row_order in FIELDS:
            matrix = build_matrix(system, expected_key, predicted_key)
            print_matrix(matrix, row_order, field_name, system)
            all_matrices[system][field_name] = {k: dict(v) for k, v in matrix.items()}

    # Explicit low-threat-collapse readout: what does "low" actually get predicted as?
    print("\n\n=== low-threat collapse readout: what 'low' cases get predicted AS ===")
    print("| system | low->low | low->medium | low->high | low->critical | n low cases |")
    print("|---|---|---|---|---|---|")
    for system in SYSTEMS:
        m = all_matrices[system]["threat_level"].get("low", {})
        n = sum(m.values())
        print(f"| {system} | {m.get('low', 0)} | {m.get('medium', 0)} | "
              f"{m.get('high', 0)} | {m.get('critical', 0)} | {n} |")

    (REPO / "evaluation" / "confusion_matrices.json").write_text(json.dumps(all_matrices, indent=2))
    print(f"\nsaved: evaluation/confusion_matrices.json")


if __name__ == "__main__":
    main()
