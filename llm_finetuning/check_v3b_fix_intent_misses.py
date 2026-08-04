"""
Step 1 of the "does rules_in_prompt change the architecture question" session
(AUDIT.md sec AB).

v3b-fix (AUDIT.md sec AA step 3) lost 18.2pt of intent accuracy relative to
v3b on the clean 55-case battery, concentrated in the low/medium threat
strata the 36 rewritten abstention rows touch. Before treating that as a
real cost, check whether it's actually an accounting artefact: the fix added
threat_level="unknown" as a schema-legal value, so if the retrained model
now emits partial-abstention responses (a real likely_intent alongside
threat_level="unknown", or an near-abstention phrasing is_abstention() 's
exact ABSTENTION_TOKENS match doesn't catch), evaluate_llm would score those
as intent MISSES rather than exclude them as abstentions -- inflating the
apparent regression.

Pure post-processing of evaluation/eval_expanded_v3b-fix_greedy.json (no
model calls) -- majority_intent/majority_threat/n_abstained are already
recorded per case (n_runs=1 in that file, so "majority" IS the single
observed value, no aggregation ambiguity).

Usage:
    python llm_finetuning/check_v3b_fix_intent_misses.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    data = json.loads((REPO / "evaluation" / "eval_expanded_v3b-fix_greedy.json").read_text())
    agg = data["aggregate"]
    print(f"aggregate mean_abstention_rate: {agg['mean_abstention_rate']}")

    misses = [c for c in data["per_case"] if c.get("intent_accuracy") == 0.0]
    print(f"\n{len(misses)} intent misses out of {len(data['per_case'])} cases\n")

    print("| case | predicted likely_intent | predicted threat_level | n_abstained | abstention_rate |")
    print("|---|---|---|---|---|")
    n_actually_abstained = 0
    for c in misses:
        print(f"| {c['name']} | {c['majority_intent']!r} | {c['majority_threat']!r} | "
              f"{c['n_abstained']} | {c['abstention_rate']} |")
        if c["n_abstained"] > 0:
            n_actually_abstained += 1

    print(f"\nmisses that were actually abstentions: {n_actually_abstained}/{len(misses)}")
    if n_actually_abstained == 0:
        print("VERDICT: real regression, not an accounting artefact -- every intent miss "
              "predicted a concrete (non-abstained) likely_intent value.")
    else:
        print("VERDICT: at least some of the regression IS abstention mis-scored as a miss -- "
              "the ledger needs correcting.")


if __name__ == "__main__":
    main()
