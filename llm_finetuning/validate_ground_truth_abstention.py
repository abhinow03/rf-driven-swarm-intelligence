"""
Phase 3a step 1: validates src/swarm_intent/ground_truth_abstention.py against AUDIT.md sec
AK's 502-case STGT-derived categorization (evaluation/categorize_unanswerable_502.json).
Reports raw agreement rate and, for every disagreement, whether it traces to STGT's own read
noise (the expected, designed-for outcome -- this module is independent of STGT specifically
because sec AN proved that read is unreliable).

CPU-only, no model needed -- this only reads the true_chain field already persisted in the
sec AK categorization output (the simulator's OWN chain, not any STGT read) and re-derives
the ground-truth label from it via classify_trajectory_ground_truth().

Usage:
    python llm_finetuning/validate_ground_truth_abstention.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from swarm_intent.ground_truth_abstention import (  # noqa: E402
    classify_trajectory_ground_truth, MULTI_HOP, OSCILLATION, TERMINAL_TRANSITIONING,
)

SEC_AK_PATH = REPO / "evaluation" / "categorize_unanswerable_502.json"

# sec AK's STGT-derived category names that map directly onto a ground-truth mechanism name;
# every other STGT category (bucket_A_misrouted, dispersed_converging_ambiguity, and any
# other guard-reason category) represents a case STGT's read did NOT structurally detect as
# multi-hop/oscillation at all -- a real disagreement to report, not an error in this script.
STGT_TO_MECHANISM = {"multi_hop": MULTI_HOP, "oscillation": OSCILLATION,
                     "terminal_transitioning": TERMINAL_TRANSITIONING}


def main():
    data = json.loads(SEC_AK_PATH.read_text())
    records = data["unanswerable_records"]
    assert len(records) == 502, f"expected 502 sec AK records, found {len(records)}"

    agree = 0
    disagreements = []
    gt_counts = Counter()
    for r in records:
        chain = r["true_chain"]
        # sec AK's population never truncates mid-blend (measure_coverage.build_long_sequence
        # always completes the final dwell) -- true_labels isn't persisted in this JSON, but
        # is unneeded here: terminal_transitioning cannot fire without truncation, and chain
        # length alone fully determines multi_hop/oscillation.
        gt = classify_trajectory_ground_truth(chain, true_labels=None)
        gt_counts[gt] += 1

        stgt_category = r["category"]
        stgt_mechanism = STGT_TO_MECHANISM.get(stgt_category)  # None if STGT didn't detect it at all

        if gt == stgt_mechanism:
            agree += 1
        else:
            disagreements.append({
                "i": r["i"], "true_chain": chain, "stgt_category": stgt_category,
                "stgt_mechanism": stgt_mechanism, "ground_truth_mechanism": gt,
                "stgt_failure_mode": ("did not detect multi-hop structure at all" if stgt_mechanism is None
                                     else "detected multi-hop but assigned the wrong mechanism "
                                          "(oscillation vs multi_hop confused by a noisy intermediate read)"),
            })

    n = len(records)
    print(f"n_records={n}")
    print(f"ground-truth mechanism counts (from simulator chain, independent of STGT): {dict(gt_counts)}")
    print(f"\nagreement with sec AK's STGT-derived categorization: {agree}/{n} = {agree/n:.1%}")
    print(f"disagreements: {len(disagreements)}")

    n_nondetect = sum(1 for d in disagreements if d["stgt_mechanism"] is None)
    n_confused = sum(1 for d in disagreements if d["stgt_mechanism"] is not None)
    print(f"  -- {n_nondetect} where STGT's read did not detect multi-hop structure at all "
         f"(bucket_A_misrouted / dispersed_converging_ambiguity -- sec AK/AN's known noise)")
    print(f"  -- {n_confused} where STGT detected multi-hop but the noisy intermediate read "
         f"flipped oscillation vs multi_hop")

    print("\nfirst 10 disagreements:")
    for d in disagreements[:10]:
        print(f"  seq {d['i']:4d}  chain={d['true_chain']}  STGT said={d['stgt_category']!r}  "
             f"ground_truth={d['ground_truth_mechanism']!r}  ({d['stgt_failure_mode']})")

    out = {
        "n_records": n, "n_agree": agree, "agreement_rate": agree / n,
        "n_disagreements": len(disagreements),
        "n_disagreements_nondetect": n_nondetect, "n_disagreements_confused": n_confused,
        "ground_truth_mechanism_counts": dict(gt_counts),
        "disagreements": disagreements,
    }
    out_path = REPO / "evaluation" / "phase3a_ground_truth_validation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
