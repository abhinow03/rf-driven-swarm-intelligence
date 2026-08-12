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
    print("answerability -- n shown for EVERY cell, not just accuracy_when_answerable's")
    print("=" * 100)
    print("| system | accuracy_when_answerable (n) | abstention_when_unanswerable (n) | over_abstention_rate (n) |")
    print("|---|---|---|---|")
    for sysname in SYSTEM_ORDER:
        a = all_results[sysname]["answerability"]
        n_unanswerable = a["n_unanswerable"]
        # over_abstention_rate's own denominator (the GT/answerable-population total,
        # NOT n_answerable, which already excludes the abstained ones) --
        # n_answerable = gt_total * (1 - over_abstention_rate), so back out gt_total.
        gt_total = round(a["n_answerable"] / (1 - a["over_abstention_rate"])) if a["over_abstention_rate"] < 1 else a["n_answerable"]
        aw = f"{a['accuracy_when_answerable']:.1%} (n={a['n_answerable']})" if a["accuracy_when_answerable"] is not None else f"n/a (n={a['n_answerable']})"
        ab = f"{a['abstention_rate_when_unanswerable']:.1%} (n={n_unanswerable})" if a["abstention_rate_when_unanswerable"] is not None else f"n/a (n={n_unanswerable})"
        oa = f"{a['over_abstention_rate']:.1%} (n={gt_total})" if a["over_abstention_rate"] is not None else f"n/a (n={gt_total})"
        print(f"| {sysname} | {aw} | {ab} | {oa} |")
    print("\n(n_unanswerable=502 and the GT/answerable-population n=498 are CONSTANTS across "
         "every system -- same locked eval set. accuracy_when_answerable's n varies slightly "
         "per system because a system's own abstentions shrink its own answered-case count.)")

    print()
    print("=" * 100)
    print("escalation direction (separate, never pooled) -- n is the GT population, constant across systems")
    print("=" * 100)
    print("| system | n | correct (n) | UNDER-escalated (n) | OVER-escalated (n) | abstained (n) |")
    print("|---|---|---|---|---|---|")
    for sysname in SYSTEM_ORDER:
        e = all_results[sysname]["escalation"]
        n = e["n"]
        def cell(frac):
            return f"{frac:.1%} (n={round(frac*n)})"
        print(f"| {sysname} | {n} | {cell(e['correct'])} | {cell(e['under_escalated'])} | "
             f"{cell(e['over_escalated'])} | {cell(e['abstained'])} |")

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
