"""
AUDIT.md sec AF step 3: sec AE step 2 found `dispersed_converging_ambiguity`
present in 100% of bucket-B cases (191/191) -- because `dispersed` and
`converging` share IDENTICAL base geometry in `data_gen.py`
(`get_formation_offsets`, same `rng.uniform` branch), so any window whose true
formation is near either class is prone to a near-tie read. This is the
number that turns "share identical geometry" into "costs this many
resolvable cases" -- pure post-processing of `evaluation/coverage_measurement.
json` (sec AE step 2's already-saved per-case `guard_reasons` lists), no GPU.

Bucket assignment already runs structural (C) checks before guard (B) checks
(src/swarm_intent/coverage.classify_observation) -- so a bucket-B case with
`dispersed_converging_ambiguity` as its ONLY guard reason has already passed
every structural resolvability test; removing that one guard condition would
move it to bucket A with no other change. A case where the ambiguity
co-occurs with another guard reason (oov_name / dominant_history_contradiction
/ low_confidence) would stay in bucket B regardless, guarded by the other
condition.

Usage:
    python llm_finetuning/quantify_dispersed_converging_defect.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    data = json.loads((REPO / "evaluation" / "coverage_measurement.json").read_text())
    records = data["records"]
    n_total = len(records)

    b_records = [r for r in records if r["bucket"] == "B"]
    n_b = len(b_records)

    combo_counts = Counter(tuple(sorted(r["guard_reasons"])) for r in b_records)

    print(f"=== bucket B guard-reason combinations (n={n_b}) ===")
    print("| combination | n | % of B |")
    print("|---|---|---|")
    for combo, n in combo_counts.most_common():
        print(f"| {' + '.join(combo)} | {n} | {n/n_b:.1%} |")

    ambiguity_only = combo_counts.get(("dispersed_converging_ambiguity",), 0)
    ambiguity_any = sum(n for combo, n in combo_counts.items() if "dispersed_converging_ambiguity" in combo)
    ambiguity_co_occurring = ambiguity_any - ambiguity_only

    print(f"\ndispersed_converging_ambiguity present at all: {ambiguity_any}/{n_b} ({ambiguity_any/n_b:.1%} of B)")
    print(f"  -- as the SOLE guard reason (would flip to A if fixed): "
         f"{ambiguity_only}/{n_b} ({ambiguity_only/n_b:.1%} of B)")
    print(f"  -- co-occurring with another guard reason (would stay B): "
         f"{ambiguity_co_occurring}/{n_b} ({ambiguity_co_occurring/n_b:.1%} of B)")

    new_a = 9 + ambiguity_only  # sec AE step 2: bucket A was 9/500
    print(f"\n=== estimated effect of giving dispersed/converging distinct base geometry ===")
    print(f"bucket A: {9}/{n_total} ({9/n_total:.1%}) -> {new_a}/{n_total} ({new_a/n_total:.1%})")
    print(f"bucket B: {n_b}/{n_total} ({n_b/n_total:.1%}) -> "
         f"{n_b - ambiguity_only}/{n_total} ({(n_b - ambiguity_only)/n_total:.1%})")
    print(f"bucket C: unchanged (dispersed/converging ambiguity is a bucket-B-only guard, "
         f"never a bucket-C structural condition)")
    print(f"\nThis is a LOWER-BOUND estimate: it assumes fixing the geometry collision does "
         f"nothing but remove the ambiguity flag on windows that are ALREADY otherwise clean. "
         f"It does NOT account for a second-order effect the data can't isolate: some fraction "
         f"of {ambiguity_co_occurring} co-occurring cases' OTHER guard triggers (oov_name via "
         f"the model reading 'transitioning' instead of dispersed/converging, or "
         f"dominant_history_contradiction from a near-50/50 window split) may THEMSELVES be "
         f"downstream consequences of the same geometry collision confusing the classifier -- "
         f"if so, the true ceiling is higher than {new_a/n_total:.1%}, not measurable from this "
         f"data alone.")


if __name__ == "__main__":
    main()
