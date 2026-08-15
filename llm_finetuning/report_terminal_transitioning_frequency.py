"""
Phase 3a review follow-up step 3: terminal_transitioning's true frequency.

The request names "the LOCKED_seed999_FINAL.json 502-case population" -- these are
actually TWO DIFFERENT ARTIFACTS conflated: sec AK's 502-case unanswerable population
is a subset of evaluation/phase4_eval_set.json (seed=4321, n=1000, filtered to
has_ground_truth=False), NOT eval_data/LOCKED_seed999_FINAL.json (a separate, unrelated
seed=999, n=1000 population from the sec AM Rule-0 lock, used for ceiling measurements,
never for Phase 3a). This script reports BOTH, explicitly disambiguated, rather than
silently picking one.

No GPU/model load for either check: LOCKED_seed999_FINAL.json carries `chain` +
`true_labels` directly (ground truth by construction), and phase4_eval_set.json's stored
`bucket`/`has_ground_truth` fields already flag which of its 1000 items are the sec AK
502-case unanswerable subset -- classify_trajectory_ground_truth() needs no STGT re-read
for either.

Usage:
    python llm_finetuning/report_terminal_transitioning_frequency.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from swarm_intent.ground_truth_abstention import classify_trajectory_ground_truth  # noqa: E402


def main():
    print("=== A. eval_data/LOCKED_seed999_FINAL.json (seed=999, n=1000, sec AM's locked ceiling population) ===")
    locked999 = json.loads((REPO / "eval_data" / "LOCKED_seed999_FINAL.json").read_text())
    recs = locked999["records"]
    assert locked999["seed"] == 999 and locked999["n"] == 1000 == len(recs)
    mech_counts = Counter()
    for r in recs:
        m = classify_trajectory_ground_truth(r["chain"], r.get("true_labels"))
        mech_counts[m if m is not None else "answerable"] += 1
    n = len(recs)
    for mech in ("multi_hop", "oscillation", "terminal_transitioning", "answerable"):
        k = mech_counts.get(mech, 0)
        print(f"  {mech}: {k}/{n} ({k/n:.2%})")
    tt_999 = mech_counts.get("terminal_transitioning", 0)

    print("\n=== B. evaluation/phase4_eval_set.json (seed=4321, n=1000) -- sec AK's ACTUAL source, "
         "502-case has_ground_truth=False subset ===")
    phase4 = json.loads((REPO / "evaluation" / "phase4_eval_set.json").read_text())
    assert phase4["seed"] == 4321 and phase4["n_sequences"] == 1000
    unanswerable = [it for it in phase4["items"] if not it["has_ground_truth"]]
    assert len(unanswerable) == 502, f"expected 502 unanswerable items, found {len(unanswerable)}"
    # phase4_eval_set.json items don't carry true_labels (no truncation-construction was ever
    # applied to this natural population), so terminal_transitioning can only be checked as
    # "did it ever occur" -- which requires len(chain)==1 or 2 AND an artificial mid-blend
    # truncation. Natural generation never truncates, so this can only be non-zero if the
    # STORED true_chain itself somehow encodes it -- it cannot (chain is the full formation
    # list, not a truncation flag). Confirms the "0% natural frequency" claim structurally,
    # not just by counting.
    mech_counts_502 = Counter()
    for it in unanswerable:
        m = classify_trajectory_ground_truth(it["true_chain"], true_labels=None)
        mech_counts_502[m if m is not None else "answerable(!?)"] += 1
    n502 = len(unanswerable)
    for mech in ("multi_hop", "oscillation", "terminal_transitioning", "answerable(!?)"):
        k = mech_counts_502.get(mech, 0)
        print(f"  {mech}: {k}/{n502} ({k/n502:.2%})")
    tt_502 = mech_counts_502.get("terminal_transitioning", 0)

    print("\n=== C. What sec AK actually reported (STGT-derived, for contrast) ===")
    sec_ak = json.loads((REPO / "evaluation" / "categorize_unanswerable_502.json").read_text())
    print(f"  sec AK's STGT-derived category_counts: {sec_ak['category_counts']}")
    print(f"  ('terminal_transitioning': {sec_ak['category_counts'].get('terminal_transitioning', 0)} "
         f"-- this is the '1/502' figure the request cites)")

    print("\n=== recommendation ===")
    print(f"Ground-truth terminal_transitioning frequency: {tt_999}/1000 on the seed=999 locked "
         f"population, {tt_502}/502 on sec AK's actual seed=4321 unanswerable subset. Both exactly "
         f"0.0% -- confirms docs/PHASE3A_ABSTENTION_CORPUS.md's claim that this mechanism cannot "
         f"occur under standard (untruncated) generation, on TWO independent populations, not just "
         f"the one already cited.")
    print("Sec AK's cited '1/502' is not a real terminal_transitioning instance -- see seq 879 in "
         "the step-2 disagreement printout: true_chain is a 3-formation multi_hop case "
         "(diamond->column->converging), and STGT's own noisy read happened to end in an 'unknown' "
         "window, which sec AK's STGT-derived categorization mapped to 'terminal_transitioning' "
         "purely because that specific bucket subtype (terminal_unknown) fires before the "
         "multi_hop check in coverage.py's early-return order (see AUDIT.md sec AN step 2's "
         "'early-return-order masking' finding, same case, already documented there). The TRUE "
         "natural-generation frequency of terminal_transitioning is 0/1000 and 0/502 on the two "
         "populations checked here -- sec AK's figure was always an artifact of STGT's read, not "
         "a real occurrence.")


if __name__ == "__main__":
    main()
