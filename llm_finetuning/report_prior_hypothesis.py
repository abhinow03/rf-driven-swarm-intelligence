"""
Step 2 of the "resolve the calibration/gap contradiction" session (AUDIT.md
sec AA -- no new generations, reads the existing eval_expanded_rules_in_prompt.json).

Tests: does rules_in_prompt (base Qwen2.5-7B-Instruct + RULES.txt in the system
prompt, no fine-tuning at all) fail on the SAME 8 low-threat pairs v3a fails on
(sec Y), and succeed on the same 4 it succeeds on? If so, the mechanism is a
pretraining semantic prior conflicting with a counterintuitive rule -- present
even with zero fine-tuning -- not a fine-tuning-specific data gap.

Usage:
    python llm_finetuning/report_prior_hypothesis.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

V3A_FAILING = ["encirclement->column", "encirclement->dispersed", "converging->column",
              "v_shape->column", "v_shape->dispersed", "column->dispersed",
              "dispersed->column", "shield->dispersed"]
V3A_SUCCEEDING = ["column->column", "diamond->diamond", "dispersed->dispersed", "converging->dispersed"]


def main():
    data = json.loads((REPO / "evaluation" / "eval_expanded_rules_in_prompt.json").read_text())
    by_name = {c["name"]: c for c in data["per_case"]}

    print("=== rules_in_prompt behavior on v3a's 8 FAILING low-threat pairs ===")
    print("| pair | rules_in_prompt majority_threat | threat_accuracy | matches v3a's failure? |")
    print("|---|---|---|---|")
    n_matches = 0
    for name in V3A_FAILING:
        c = by_name[name]
        fails_too = c["majority_threat"] != "low"
        n_matches += fails_too
        print(f"| {name} | {c['majority_threat']} | {c['threat_accuracy']:.1%} | "
              f"{'YES' if fails_too else 'no'} |")

    print(f"\n=== rules_in_prompt behavior on v3a's 4 SUCCEEDING low-threat pairs ===")
    print("| pair | rules_in_prompt majority_threat | threat_accuracy |")
    print("|---|---|---|")
    for name in V3A_SUCCEEDING:
        c = by_name[name]
        print(f"| {name} | {c['majority_threat']} | {c['threat_accuracy']:.1%} |")

    print(f"\nrules_in_prompt fails (majority != low) on {n_matches}/{len(V3A_FAILING)} of "
          f"v3a's failing pairs.")
    if n_matches >= 7:
        print("HYPOTHESIS HOLDS: rules_in_prompt fails on nearly the same set -- a "
              "pretraining prior conflicting with the rule table, present even with zero "
              "fine-tuning, is the likely mechanism.")
    else:
        print("HYPOTHESIS DOES NOT HOLD: rules_in_prompt succeeds on most of v3a's "
              "failing pairs -- the base model (no fine-tuning) can apply these "
              "'counterintuitive' rules from context most of the time. v3a's failures "
              "are not simply inherited pretraining prior conflict.")


if __name__ == "__main__":
    main()
