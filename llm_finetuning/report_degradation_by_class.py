"""
Step 4 of the low-threat-collapse diagnosis session (AUDIT.md sec M/N/O/P).

Re-cuts the existing degradation-battery JSONs (evaluation/degradation_{system}.json,
6 ORIGINAL_TEST_CASES perturbed along 5 axes) by the base case's expected_threat, to
check whether the sec M low-threat collapse is specific to the clean in-distribution
battery or shows up under perturbation too. Pure post-processing, no model calls.

Only ORIGINAL_TEST_CASES have threat coverage {low: 2, medium: 2, high: 2, critical: 0}
(Stable Patrol / Breaking Contact = low; Defensive Shield / Area Search = medium;
Converging Attack / Encirclement Behavior = high) -- there is no critical base case in
the degradation battery, unlike the 55-case battery which has 2. n=2 per class here is
already thin; treat all of this as directional, not conclusive.

Usage:
    python llm_finetuning/report_degradation_by_class.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from swarm_intent.llm.prompts import ORIGINAL_TEST_CASES  # noqa: E402

SYSTEMS = ["rules_lookup", "base", "rules_in_prompt", "v2", "v3a", "v3a-nomask", "v3b"]
BASE_CASE_THREAT = {c["name"]: c["expected_threat"] for c in ORIGINAL_TEST_CASES}


def main():
    print("=== degradation battery (perturbed) accuracy_when_answerable, "
          "re-cut by base case's expected_threat ===")
    print(f"base-case threat coverage: "
          f"{ {t: sum(1 for v in BASE_CASE_THREAT.values() if v == t) for t in ('low','medium','high','critical')} } "
          f"-- no critical base case exists in this battery\n")

    for system in SYSTEMS:
        path = REPO / "evaluation" / f"degradation_{system}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        by_threat = defaultdict(lambda: defaultdict(list))
        for axis, sev_blocks in d["axes"].items():
            for sb in sev_blocks:
                for c in sb["per_case"]:
                    if not c["has_ground_truth"]:
                        continue
                    base_name = c["name"].split("__")[0]
                    threat = BASE_CASE_THREAT.get(base_name)
                    if threat is None:
                        continue
                    by_threat[threat]["intent"].append(c["intent_accuracy"])
                    by_threat[threat]["threat"].append(c["threat_accuracy"])

        print(f"--- {system} ---")
        for threat in ("low", "medium", "high"):
            intent_vals = by_threat[threat]["intent"]
            threat_vals = by_threat[threat]["threat"]
            if not intent_vals:
                print(f"  {threat}: no answerable perturbed cases")
                continue
            print(f"  {threat}: n={len(intent_vals)} perturbed cases, "
                  f"intent_accuracy mean={np.mean(intent_vals):.1%} std={np.std(intent_vals):.1%} | "
                  f"threat_accuracy mean={np.mean(threat_vals):.1%} std={np.std(threat_vals):.1%}")
        print()


if __name__ == "__main__":
    main()
