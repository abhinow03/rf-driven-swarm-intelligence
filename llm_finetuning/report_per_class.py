"""
Step 3 of the "is the 55-case battery in-distribution?" session (AUDIT.md sec J/K).

Re-emits the already-collected evaluation/eval_expanded_{system}.json results
(step 3 of the previous session, run_headline_eval.py -- 55-case battery, n_runs=5)
broken down by threat_level rather than pooled into one aggregate number. Pure
post-processing of existing JSON -- no model calls, no Reporter needed (runs in
under a second).

Per-case fields (intent_accuracy/threat_accuracy/action_accuracy) are already
per-case fractions across the n_runs replicate generations for that case
(see evaluate.py); this script means/stds those ACROSS CASES within each threat
stratum, which is a case-to-case spread, not the run-to-run replicate spread
run_headline_eval.py's mean_across_runs/std_across_runs report -- the two are not
interchangeable, and the "critical" stratum's std is reported anyway for
completeness but should not be trusted (n=2).

Usage:
    python llm_finetuning/report_per_class.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from swarm_intent.llm.prompts import TEST_CASES  # noqa: E402

THREAT_ORDER = ["low", "medium", "high", "critical"]
SYSTEMS = ["rules_lookup", "base", "rules_in_prompt", "v2", "v3a", "v3a-nomask", "v3b"]

NAME_TO_THREAT = {c["name"]: c["expected_threat"] for c in TEST_CASES}


def mean_std(values):
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))


def main():
    out_dir = REPO / "evaluation"
    n_per_threat = defaultdict(int)
    for t in NAME_TO_THREAT.values():
        n_per_threat[t] += 1
    print(f"=== per-class (threat-level) breakdown of the 55-case battery ===")
    print(f"stratum sizes: {dict(n_per_threat)} "
          f"(critical n={n_per_threat['critical']} -- any figure for this stratum is "
          f"NOT statistically meaningful, reported for completeness only)\n")

    all_breakdowns = {}
    for system in SYSTEMS:
        path = out_dir / f"eval_expanded_{system}.json"
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        data = json.loads(path.read_text())
        by_threat = defaultdict(lambda: defaultdict(list))
        for case in data["per_case"]:
            threat = NAME_TO_THREAT.get(case["name"])
            if threat is None:
                continue  # shouldn't happen -- every TEST_CASES name is in NAME_TO_THREAT
            if case["intent_accuracy"] is not None:
                by_threat[threat]["intent"].append(case["intent_accuracy"])
            if case["threat_accuracy"] is not None:
                by_threat[threat]["threat"].append(case["threat_accuracy"])
            if case["action_accuracy"] is not None:
                by_threat[threat]["action"].append(case["action_accuracy"])

        print(f"--- {system} ---")
        print("| threat class | n cases | intent acc (mean±std across cases) | "
              "threat acc | action acc |")
        print("|---|---|---|---|---|")
        system_breakdown = {}
        for threat in THREAT_ORDER:
            n = n_per_threat[threat]
            intent_m, intent_s = mean_std(by_threat[threat]["intent"])
            threat_m, threat_s = mean_std(by_threat[threat]["threat"])
            action_m, action_s = mean_std(by_threat[threat]["action"])
            flag = "  <- n=2, NOT MEANINGFUL" if threat == "critical" else ""
            fmt = lambda m, s: (f"{m:.1%}±{s:.1%}" if m is not None else "N/A")
            print(f"| {threat} | {n} | {fmt(intent_m, intent_s)} | {fmt(threat_m, threat_s)} | "
                  f"{fmt(action_m, action_s)} |{flag}")
            system_breakdown[threat] = {
                "n": n, "intent_mean": intent_m, "intent_std": intent_s,
                "threat_mean": threat_m, "threat_std": threat_s,
                "action_mean": action_m, "action_std": action_s,
            }
        all_breakdowns[system] = system_breakdown
        print()

    (out_dir / "per_class_breakdown.json").write_text(json.dumps(all_breakdowns, indent=2))
    print(f"saved: {out_dir / 'per_class_breakdown.json'}")


if __name__ == "__main__":
    main()
