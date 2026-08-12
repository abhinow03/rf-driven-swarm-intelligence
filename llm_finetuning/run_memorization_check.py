"""
Runs the memorization-vs-generalization scorer (score_memorization.py) on
v5-a's real Phase 4 output. Per PREREGISTRATION.md, this is reported BEFORE
headline accuracy -- it's the check that determines whether the accuracy
numbers mean "learned the task" or "learned the corpus."

Usage:
    python llm_finetuning/run_memorization_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "llm_finetuning"))
sys.path.insert(0, str(REPO / "src"))

from score_memorization import (  # noqa: E402
    build_training_index, overlap_rate, interpret, NULL_HYPOTHESIS_BASELINE, MEMORIZATION_SIGNAL_BAR,
)
from swarm_intent.llm.prompts import is_abstention  # noqa: E402

TRAIN_PATH = REPO / "data" / "sft_train_v5_phase1.jsonl"
RESULTS_PATH = REPO / "evaluation" / "phase4_v5a_results.json"
OUT_PATH = REPO / "evaluation" / "phase4_v5a_memorization.json"


def main():
    print("building training index (49 RULES pairs, data/sft_train_v5_phase1.jsonl)...")
    index = build_training_index(str(TRAIN_PATH))

    with open(RESULTS_PATH) as f:
        results = json.load(f)
    parsed_by_case = results["parsed_by_case"]

    # Build (pair, situation_summary) for every ANSWERED case with an identifiable
    # (form_a, form_b) pair.
    #
    # BUG FIXED (hardening audit step 1): originally this regex-extracted the pair
    # from ctx TEXT via swarm_intent.coverage._extract_pair_from_ctx, which requires
    # the phrase "Transition detected at t=..." -- but REAL production ctx text
    # (stgt_bridge.py's bridge_predictions(), what this eval set actually contains)
    # renders "Transition at t=..." WITHOUT "detected". coverage.py's own docstring
    # already documented this exact discrepancy ("production's build_tactical_context()
    # phrases this line differently... no 'detected'") but nothing had ever applied
    # the fix to a real-output consumer -- this script was silently under-counting.
    # For has_ground_truth=True cases the fix is simpler than a corrected regex: the
    # TRUE (form_a,form_b) pair is already known independently (phase4_eval_set.json's
    # `pair` field, derived from the generator's true_chain, never from ctx text) --
    # use it directly, no extraction needed. For has_ground_truth=False (genuinely
    # multi-hop) cases there IS no single true pair by definition, so those still use
    # ctx-text extraction (kept as a secondary, lower-confidence signal, not "fixed"
    # the same way -- there's nothing to fix it TO).
    items_by_name = {it["name"]: it for it in json.load(open(REPO / "evaluation" / "phase4_eval_set.json"))["items"]}

    import sys as _sys
    _sys.path.insert(0, str(REPO / "src"))
    from swarm_intent.coverage import _extract_pair_from_ctx

    eval_cases = []
    n_no_pair, n_abstained, n_invalid = 0, 0, 0
    n_from_ground_truth, n_from_ctx_regex = 0, 0
    for name, parsed in parsed_by_case.items():
        item = items_by_name[name]
        if parsed is None:
            n_invalid += 1
            continue
        if is_abstention(parsed.get("likely_intent", "")):
            n_abstained += 1
            continue
        if item["has_ground_truth"]:
            pair = tuple(item["pair"])
            n_from_ground_truth += 1
        else:
            pair = _extract_pair_from_ctx(item["ctx"])
            if pair is None:
                n_no_pair += 1
                continue
            n_from_ctx_regex += 1
        eval_cases.append({"name": name, "pair": pair, "situation_summary": parsed.get("situation_summary", "")})

    print(f"answered+pair-identifiable cases: {len(eval_cases)}/1000 "
         f"({n_from_ground_truth} from known ground-truth pair, {n_from_ctx_regex} from ctx-text "
         f"regex on genuinely multi-hop cases) -- excluded: {n_invalid} invalid JSON, "
         f"{n_abstained} abstained, {n_no_pair} no extractable pair (all multi-hop, no ground truth to fall back on)")

    rate, details = overlap_rate(eval_cases, index)

    print("\n" + "=" * 90)
    print("MEMORIZATION-VS-GENERALIZATION VERDICT")
    print("=" * 90)
    print(f"overlap rate: {rate:.1%} ({sum(1 for d in details if d['near_duplicate'])}/{len(details)} "
         f"near-duplicate at >=0.90 cosine similarity to a same-pair training target)")
    print(f"null-hypothesis baseline (measured on the real corpus): {NULL_HYPOTHESIS_BASELINE:.1%}")
    print(f"memorization-signal bar (preregistered): >= {MEMORIZATION_SIGNAL_BAR:.0%}")
    print(interpret(rate))

    OUT_PATH.write_text(json.dumps({"overlap_rate": rate, "n_scored": len(details),
                                   "n_excluded_invalid": n_invalid, "n_excluded_abstained": n_abstained,
                                   "n_excluded_no_pair": n_no_pair, "details": details}, indent=2))
    print(f"\nsaved {OUT_PATH}")


if __name__ == "__main__":
    main()
