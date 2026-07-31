"""
Step 1 (second half) of the "sampling vs greedy" session (AUDIT.md sec Z).

Compares the greedy re-run (evaluation/eval_expanded_{system}_greedy.json,
llm_finetuning/run_greedy_eval.py) against the existing sampled results
(evaluation/eval_expanded_{system}.json, do_sample=True/temperature=0.3/
top_p=0.8/top_k=20/repetition_penalty=1.05) per threat class.

Usage:
    python llm_finetuning/report_greedy_vs_sampled.py
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

SYSTEMS = ["v2", "v3a", "v3a-nomask", "v3b"]
NAME_TO_THREAT = {c["name"]: c["expected_threat"] for c in TEST_CASES}


def per_class_threat_accuracy(per_case):
    by_threat = defaultdict(list)
    for c in per_case:
        t = NAME_TO_THREAT.get(c["name"])
        if t:
            by_threat[t].append(c["threat_accuracy"])
    return {t: float(np.mean(v)) for t, v in by_threat.items()}


def main():
    print("=== greedy vs sampled: per-class threat_accuracy ===\n")
    print("| system | class | sampled (temp=0.3) | greedy (temp=0.0) | delta |")
    print("|---|---|---|---|---|")
    any_substantial = False
    for system in SYSTEMS:
        sampled = json.loads((REPO / "evaluation" / f"eval_expanded_{system}.json").read_text())
        greedy = json.loads((REPO / "evaluation" / f"eval_expanded_{system}_greedy.json").read_text())
        s_acc = per_class_threat_accuracy(sampled["per_case"])
        g_acc = per_class_threat_accuracy(greedy["per_case"])
        for cls in ("low", "medium", "high", "critical"):
            s, g = s_acc.get(cls), g_acc.get(cls)
            if s is None or g is None:
                continue
            delta = g - s
            if abs(delta) >= 0.15:
                any_substantial = True
            print(f"| {system} | {cls} | {s:.1%} | {g:.1%} | {delta:+.1%} |")

        s_overall = sampled["aggregate"]["accuracy_when_answerable"]
        g_overall = greedy["aggregate"]["accuracy_when_answerable"]
        print(f"| {system} | **overall (intent)** | {s_overall:.1%} | {g_overall:.1%} | "
              f"{g_overall - s_overall:+.1%} |")
        print("|---|---|---|---|---|")

    print(f"\nAt least one class/system delta >=15 points: {any_substantial}")


if __name__ == "__main__":
    main()
