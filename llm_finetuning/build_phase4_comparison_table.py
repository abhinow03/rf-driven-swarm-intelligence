"""
Joins v5-a's real Phase 4 results (evaluation/phase4_v5a_results.json) with
the 4 baseline systems' scored results (evaluation/phase4_baselines_scored.json,
built by score_phase4_baselines.py) into one comparison table across all 5
systems, using the exact field names eval_sft_v5.py's evaluate() produces for
every one of them (same scorer, no divergent logic).

Usage:
    python llm_finetuning/build_phase4_comparison_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
THREAT_ORDER = ("low", "medium", "high", "critical")
SYSTEM_ORDER = ("base", "rules_in_prompt", "v3b-fix", "v2", "v5-a")


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    v5a = load(REPO / "evaluation" / "phase4_v5a_results.json")
    baselines = load(REPO / "evaluation" / "phase4_baselines_scored.json")

    all_results = {**baselines, "v5-a": v5a}

    print("=" * 100)
    print("PHASE 4 COMPARISON: per-class threat accuracy (n, Wilson 95% CI)")
    print("=" * 100)
    print("| system | " + " | ".join(THREAT_ORDER) + " |")
    print("|---|" + "---|" * len(THREAT_ORDER))
    for sysname in SYSTEM_ORDER:
        r = all_results[sysname]["per_class_threat_accuracy"]
        cells = []
        for level in THREAT_ORDER:
            c = r[level]
            if c["accuracy"] is None:
                cells.append("n/a (n=0)")
            else:
                caveat = " LOW-N" if c["low_n_caveat"] else ""
                cells.append(f"{c['accuracy']:.1%} (n={c['n']}){caveat}")
        print(f"| {sysname} | " + " | ".join(cells) + " |")

    print()
    print("=" * 100)
    print("answerability: accuracy_when_answerable / abstention_rate_when_unanswerable / over_abstention_rate")
    print("=" * 100)
    print("| system | accuracy_when_answerable | n | abstention_when_unanswerable | over_abstention_rate |")
    print("|---|---|---|---|---|")
    for sysname in SYSTEM_ORDER:
        a = all_results[sysname]["answerability"]
        aw = f"{a['accuracy_when_answerable']:.1%}" if a["accuracy_when_answerable"] is not None else "n/a"
        ab = f"{a['abstention_rate_when_unanswerable']:.1%}" if a["abstention_rate_when_unanswerable"] is not None else "n/a"
        oa = f"{a['over_abstention_rate']:.1%}" if a["over_abstention_rate"] is not None else "n/a"
        print(f"| {sysname} | {aw} | {a['n_answerable']} | {ab} | {oa} |")

    print()
    print("=" * 100)
    print("escalation direction (separate, never pooled)")
    print("=" * 100)
    print("| system | n | correct | UNDER-escalated | OVER-escalated | abstained |")
    print("|---|---|---|---|---|---|")
    for sysname in SYSTEM_ORDER:
        e = all_results[sysname]["escalation"]
        print(f"| {sysname} | {e['n']} | {e['correct']:.1%} | {e['under_escalated']:.1%} | "
             f"{e['over_escalated']:.1%} | {e['abstained']:.1%} |")

    print()
    print("=" * 100)
    print("schema validity + critical-pair tactical accuracy")
    print("=" * 100)
    print("| system | schema_validity_rate | critical_pair_accuracy |")
    print("|---|---|---|")
    for sysname in SYSTEM_ORDER:
        s = all_results[sysname]["schema_validity_rate"]
        c = all_results[sysname]["critical_pair_tactical_accuracy"]
        cp = f"{c['accuracy']:.1%} (n={c['n']})" if c["accuracy"] is not None else f"n/a (n={c['n']})"
        print(f"| {sysname} | {s['rate']:.1%} ({s['n_valid']}/{s['n_total']}) | {cp} |")


if __name__ == "__main__":
    main()
