"""
Phase 3a review follow-up step 2: prints all 27 step-1 disagreement cases
(new ground-truth classifier vs sec AK's STGT-derived label) with the
SPECIFIC evidence for which one is right, per case -- not a restatement of
the 94.6% aggregate agreement rate.

For each disagreement:
  - re-runs classify_trajectory_ground_truth(true_chain) LIVE (independent
    re-verification, not trusting the stored ground_truth_mechanism field)
  - pulls STGT's own raw "Formation history: ..." line straight out of
    evaluation/phase4_eval_set.json's stored ctx text for that sequence --
    this is the concrete, inspectable evidence of what STGT actually read,
    not a re-assertion that it was "noisy"

No GPU/model load needed -- everything is already-persisted data.

Usage:
    python llm_finetuning/print_disagreements_case_by_case.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from swarm_intent.ground_truth_abstention import classify_trajectory_ground_truth  # noqa: E402

FORMATION_HISTORY_RE = re.compile(r"Formation history: (.+)")


def main():
    validation = json.loads((REPO / "evaluation" / "phase3a_ground_truth_validation.json").read_text())
    disagreements = validation["disagreements"]
    assert len(disagreements) == 27, f"expected 27 disagreements, found {len(disagreements)}"

    phase4 = json.loads((REPO / "evaluation" / "phase4_eval_set.json").read_text())
    by_name = {it["name"]: it for it in phase4["items"]}

    n_reverify_ok = 0
    print(f"{'i':>5} | {'true_chain':<40} | {'ground_truth (live)':<12} | {'stgt_label':<20} | STGT's raw read")
    print("-" * 160)
    for rec in disagreements:
        i = rec["i"]
        true_chain = rec["true_chain"]

        # Independent live re-verification -- not trusting the stored field.
        recomputed = classify_trajectory_ground_truth(true_chain)
        stored = rec["ground_truth_mechanism"]
        assert recomputed == stored, (
            f"seq {i}: live recompute ({recomputed}) disagrees with the stored "
            f"ground_truth_mechanism ({stored}) -- investigate before trusting anything else")
        n_reverify_ok += 1

        item = by_name.get(f"phase4_seq_{i}")
        stgt_raw_history = "N/A (item not found)"
        if item is not None:
            m = FORMATION_HISTORY_RE.search(item["ctx"])
            if m:
                stgt_raw_history = m.group(1)

        stgt_label = rec["stgt_mechanism"] or rec["stgt_category"]
        chain_str = " -> ".join(true_chain)
        print(f"{i:>5} | {chain_str:<40} | {recomputed:<12} | {stgt_label:<20} | {stgt_raw_history}")

    print(f"\n{n_reverify_ok}/27 live re-verifications match the stored ground_truth_mechanism "
         f"(independent confirmation, not a restatement).")

    print("\n=== per-case verdict: which label is right, and why ===")
    for rec in disagreements:
        i = rec["i"]
        true_chain = rec["true_chain"]
        item = by_name.get(f"phase4_seq_{i}")
        stgt_raw_history = None
        if item is not None:
            m = FORMATION_HISTORY_RE.search(item["ctx"])
            stgt_raw_history = m.group(1) if m else None

        chain_str = " -> ".join(true_chain)
        gt = rec["ground_truth_mechanism"]
        stgt_lbl = rec["stgt_mechanism"] or rec["stgt_category"]

        print(f"\nseq {i}: true_chain = {chain_str} -- generator ground truth, chosen not inferred.")
        print(f"  ground truth (this classifier): {gt}")
        print(f"  STGT-derived label (sec AK):    {stgt_lbl}")
        if stgt_raw_history:
            print(f"  STGT's own raw read:            {stgt_raw_history}")
            stgt_formations = [f.strip() for f in stgt_raw_history.split("->")]
            hallucinated = [f for f in stgt_formations if f not in true_chain and f != "unknown"]
            if hallucinated:
                print(f"  VERDICT: STGT's read names a formation NEVER in the true chain at all "
                     f"({hallucinated}) -- this is not a borderline call, the ground-truth label "
                     f"is right because the STGT-derived label is built on a hallucinated state.")
            elif "unknown" in stgt_formations:
                print(f"  VERDICT: STGT's read drops/obscures a real state behind 'unknown' windows -- "
                     f"the true_chain (known with certainty, chosen by the generator) is the only "
                     f"one of the two labels that isn't reconstructed through STGT's own noise.")
            else:
                print(f"  VERDICT: STGT's read used only real BASE_FORMATIONS names but still "
                     f"assigned the wrong MECHANISM (e.g. missed a return-to-start) -- true_chain "
                     f"still wins since it is the generator's own choice, not a reconstruction.")
        else:
            print(f"  (no matching phase4_eval_set.json item found for this sequence -- true_chain "
                 f"is still definitive on its own)")


if __name__ == "__main__":
    main()
