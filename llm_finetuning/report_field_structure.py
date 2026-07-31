"""
Step 4 of the "under-training vs data-diversity" session (AUDIT.md sec S -- see
module docstring note on lettering).

Side-by-side per-field accuracy (intent / threat / action) for every system
evaluated on the 55-case battery so far, to check whether likely_intent is
consistently higher than threat_level/recommended_action across every system --
which would support "lexically-recoverable fields survive low-data fine-tuning,
arbitrary mappings don't" (intent words often echo the input formation names;
threat_level/recommended_action require applying the RULES mapping with no
lexical shortcut). Pure post-processing of existing eval_expanded_*.json, no
model calls.

Usage:
    python llm_finetuning/report_field_structure.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SYSTEMS = ["rules_lookup", "base", "rules_in_prompt", "v2", "v3a", "v3a-nomask", "v3b"]


def field_means(system: str):
    data = json.loads((REPO / "evaluation" / f"eval_expanded_{system}.json").read_text())
    agg = data["aggregate"]
    return agg["mean_intent_accuracy"], agg["mean_threat_accuracy"], agg["mean_action_accuracy"]


def main():
    print("=== per-field accuracy across all systems (55-case battery) ===\n")
    print("| system | intent | threat | action | intent > threat? | intent > action? |")
    print("|---|---|---|---|---|---|")
    rows = []
    for system in SYSTEMS:
        path = REPO / "evaluation" / f"eval_expanded_{system}.json"
        if not path.exists():
            continue
        intent, threat, action = field_means(system)
        gt_threat = intent > threat
        gt_action = intent > action
        rows.append((system, intent, threat, action, gt_threat, gt_action))
        print(f"| {system} | {intent:.1%} | {threat:.1%} | {action:.1%} | "
              f"{'YES' if gt_threat else 'no'} | {'YES' if gt_action else 'no'} |")

    finetuned = [r for r in rows if r[0] not in ("rules_lookup",)]
    n_gt_threat = sum(1 for r in finetuned if r[4])
    n_gt_action = sum(1 for r in finetuned if r[5])
    print(f"\nintent > threat in {n_gt_threat}/{len(finetuned)} non-oracle systems")
    print(f"intent > action in {n_gt_action}/{len(finetuned)} non-oracle systems")
    if n_gt_threat == len(finetuned) and n_gt_action == len(finetuned):
        print("FINDING: intent accuracy is higher than BOTH threat and action for every "
              "non-oracle system, no exceptions -- consistent with lexically-recoverable "
              "fields (intent often echoes input formation vocabulary) surviving low-data "
              "fine-tuning better than fields requiring the full RULES mapping with no "
              "lexical shortcut.")
    else:
        print("FINDING: the pattern does NOT hold universally -- at least one system is an "
              "exception, so 'intent survives, threat/action don't' cannot be stated as a "
              "clean rule across every system.")


if __name__ == "__main__":
    main()
