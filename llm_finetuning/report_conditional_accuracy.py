"""
AUDIT.md sec AH step 2: sec AG's headline "correct" column scores abstentions
as incorrect in the denominator of ALL ground-truth-determinable cases, which
makes pipeline_v2-robust's 2.2% look like near-total failure without showing
how much of that is "answered wrong" vs "didn't answer." This is pure
post-processing of evaluation/eval_real_stgt_output_robust.json (already on
disk, sec AG step 3's run) -- no new generations, no new experiment.

Three columns per system: accuracy given answered (abstentions excluded from
denominator), unconditional accuracy (sec AG's existing "correct" column,
recomputed here for consistency), and abstention rate. Run-level average
over all n_runs x 249 ground-truth-determinable sequences, weighted by each
stratum's n_runs exactly as sec AG step 3 did.

Usage:
    python llm_finetuning/report_conditional_accuracy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.llm.prompts import is_abstention  # noqa: E402

THREAT_ORDER = ("low", "medium", "high", "critical")
SYSTEMS = ("v2", "rules_in_prompt", "v3b-fix", "pipeline_v2", "pipeline_v2-robust")


def normalize_threat(raw: str) -> str:
    raw = (raw or "").strip().lower()
    for level in THREAT_ORDER:
        if level in raw:
            return level
    return "unparsed"


def main():
    data = json.loads((REPO / "evaluation" / "eval_real_stgt_output_robust.json").read_text())
    seqs = data["seqs"]
    results = data["results"]
    gt_seqs = [s for s in seqs if s["has_ground_truth"]]

    print(f"n ground-truth-determinable sequences: {len(gt_seqs)}")
    print("\n| system | accuracy given answered | unconditional accuracy | abstention rate |")
    print("|---|---|---|---|")

    rows = {}
    for label in SYSTEMS:
        answered_correct_runs = []  # per-run: hits/scored, only when scored>0
        unconditional_runs = []     # per-run: hits/total (abstentions count as wrong)
        abstention_runs = []        # per-run: abstained/total

        n_runs = gt_seqs[0]["n_runs"]
        # sequences are split into two n_runs groups (stratified vs default) --
        # iterate per-sequence run index up to that sequence's own n_runs, then
        # average over the flattened (sequence, run) pool, matching sec AG's
        # "run-level averaged, weighted by each stratum's n_runs" convention.
        hits_by_run = {}
        scored_by_run = {}
        abst_by_run = {}
        total_by_run = {}
        for s in gt_seqs:
            for r in range(s["n_runs"]):
                a = results[label][s["name"]][r]
                abstained = is_abstention(a.get("likely_intent", ""))
                correct = (not abstained) and normalize_threat(a.get("threat_level", "")) == s["expected_threat"]
                key = (s["name"], r)
                hits_by_run[key] = int(correct)
                scored_by_run[key] = int(not abstained)
                abst_by_run[key] = int(abstained)
                total_by_run[key] = 1

        all_hits = list(hits_by_run.values())
        all_scored = list(scored_by_run.values())
        all_abst = list(abst_by_run.values())
        n_total = len(all_hits)
        n_scored = sum(all_scored)

        unconditional_acc = sum(all_hits) / n_total
        abstention_rate = sum(all_abst) / n_total
        conditional_acc = (sum(all_hits) / n_scored) if n_scored else float("nan")

        rows[label] = {"conditional": conditional_acc, "unconditional": unconditional_acc,
                       "abstention": abstention_rate, "n_total": n_total, "n_scored": n_scored}
        cond_str = f"{conditional_acc:.1%}" if n_scored else "n/a (never answers)"
        print(f"| {label} | {cond_str} | {unconditional_acc:.1%} | {abstention_rate:.1%} |")

    out_path = REPO / "evaluation" / "conditional_accuracy.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nsaved {out_path}")

    print("\nnote: conditional accuracy = unconditional / (1 - abstention rate); computed directly")
    print("from per-run hit/scored counts above, not back-derived from rounded percentages.")


if __name__ == "__main__":
    main()
