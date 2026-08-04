"""
AUDIT.md sec AD step 1: exact predicted-value breakdown for high (n=14) and
critical (n=2) cases, rules_in_prompt and composite -- not correct/under/over,
the full distribution of what each expected class actually got called. Answers
whether the error is clean single-step ordinal shrinkage (high->medium,
critical->high) or overshoots further (e.g. critical->medium, skipping high).

Pure post-processing, no GPU calls: reuses the raw per-run predictions already
captured in evaluation/reconcile_c_sampled_standalone.json (rules_in_prompt)
and evaluation/reconcile_d_sampled_composite.json (composite) from the prior
sec AC session (both cover the full 55-case battery at n_runs=5, keyed by
case name -> list of 5 raw assessment dicts).

Usage:
    python llm_finetuning/breakdown_high_crit.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from swarm_intent.llm.prompts import TEST_CASES, is_abstention  # noqa: E402

NAME_TO_CASE = {c["name"]: c for c in TEST_CASES}


def normalize_threat(raw: str) -> str:
    raw = (raw or "").strip().lower()
    for level in ("low", "medium", "high", "critical"):
        if level in raw:
            return level
    return "unparsed"


def breakdown(raw: dict, stratum: str) -> Counter:
    names = [n for n in raw if NAME_TO_CASE[n]["expected_threat"] == stratum]
    counts = Counter()
    total = 0
    for name in names:
        for a in raw[name]:
            total += 1
            if is_abstention(a.get("likely_intent", "")):
                counts["abstained"] += 1
            else:
                counts[normalize_threat(a.get("threat_level", ""))] += 1
    return counts, len(names), total


def main():
    out_dir = REPO / "evaluation"
    systems = {
        "rules_in_prompt": json.loads((out_dir / "reconcile_c_sampled_standalone.json").read_text())["raw"],
        "composite": json.loads((out_dir / "reconcile_d_sampled_composite.json").read_text())["raw"],
    }

    for label, raw in systems.items():
        print(f"=== {label} ===")
        for stratum in ("high", "critical"):
            counts, n_cases, total = breakdown(raw, stratum)
            print(f"  expected={stratum} (n_cases={n_cases}, n_runs*cases={total})")
            for pred, n in counts.most_common():
                print(f"    predicted={pred}: {n}/{total} ({n/total:.1%})")


if __name__ == "__main__":
    main()
