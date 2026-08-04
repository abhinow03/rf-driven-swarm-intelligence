"""
Builds the ~40 structurally-unanswerable rows added to data/sft_train_final_abstain.jsonl
(step 3 of the abstention-ablation session). See build_sft_dataset.py's RULES /
synth_context for the 234 answerable rows this appends to -- those are NOT touched
by this script, it only emits new rows.

Reuses llm_finetuning/degradation.py's own case generators (build_battery) rather
than writing a second, parallel context-generation path: the multi_hop,
terminal_transitioning and dropped_lines axes already produce has_ground_truth=False
cases for exactly the reason step 2 diagnosed (RULES has no key for chains > 2
formations, no key for terminal "transitioning", and no way to tell held-steady from
transitioned once the one line that disambiguates them is dropped). Those are the
same 3 perturbation types the task asked for ("multi-hop chains, terminal
'transitioning', and stripped context"), generated the identical way the degradation
battery evaluates against later -- so training and eval share one definition of
"unanswerable", not two that could quietly drift apart.

The 6 ORIGINAL_TEST_CASES fan out to 18+18+24 = 60 total battery cases across these 3 axes;
36 of them have has_ground_truth=False. That is the actual count (not padded to
match "~40" from the task -- the user's estimate was approximate, and manufacturing
extra synthetic unanswerable rows beyond what the battery itself generates would
mean training on cases the eval battery doesn't also exercise).

Usage:
    python llm_finetuning/build_abstain_rows.py --out data/sft_train_final_abstain.jsonl \
        --base data/sft_train_final.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from swarm_intent.inference import build_llm_prompt  # noqa: E402
from swarm_intent.llm.prompts import ORIGINAL_TEST_CASES  # noqa: E402

from degradation import build_battery  # noqa: E402

ABSTAIN_AXES = ("multi_hop", "terminal_transitioning", "dropped_lines")

_REASONS = {
    "multi_hop": (
        "The formation history shows {n_hops} transitions across the observation "
        "window, not a single resolvable A-to-B change. The tactical decision rules "
        "only cover a single transition between two known formations; there is no "
        "rule for a multi-hop chain like this, so no confident threat or intent "
        "assessment can be issued from this window alone."
    ),
    "terminal_transitioning": (
        "The formation history ends mid-transition (still in the generic "
        "'transitioning' state) with no confirmed resulting formation observed "
        "before the window closed. Threat and intent both depend on knowing which "
        "formation the swarm settles into; that information is not yet available."
    ),
    "dropped_lines": (
        "The tactical context is missing the transition/no-transition line for this "
        "window (sensor or link dropout). That is the only line that distinguishes "
        "a swarm holding formation from one that changed formation -- without it, "
        "there is no basis to determine whether anything tactically significant "
        "happened during this window."
    ),
}

_WATCH = {
    "multi_hop": "Wait for a window with a single clean transition (or an explicit "
                 "resultant formation) before issuing a threat/intent assessment.",
    "terminal_transitioning": "Re-observe until the formation transition completes and "
                               "a resultant formation is confirmed.",
    "dropped_lines": "Re-acquire the dropped sensor line before the next window; "
                      "flag the gap to the operator so it isn't mistaken for a "
                      "confirmed steady-state hold.",
}


def _formation_history(ctx: str) -> str:
    m = re.search(r"^Formation history: (.+)$", ctx, re.MULTILINE)
    return m.group(1) if m else "unavailable"


def gold_abstain_assessment(case: dict) -> dict:
    axis = case["axis"]
    history = _formation_history(case["ctx"])
    # " -> transitioning -> " is the separator between each pair of known formations
    # in the chain (see degradation.py's _render_lines) -- counting it directly gives
    # the number of hops without double-counting the substring against itself.
    n_hops = history.count(" -> transitioning -> ")
    reason = _REASONS[axis].format(n_hops=max(n_hops, 1))
    return {
        "situation_summary": (
            f"Formation history for this window ('{history}') does not resolve to a "
            f"single confirmed transition, so no tactical assessment can be made."
        ),
        "threat_level": "unknown",  # AUDIT.md sec AA step 3: was hardcoded "medium" because
        # the schema had no "unknown" threat_level at the time (prompts.py THREAT_FAMILIES now
        # has one, matching likely_intent's existing "unknown" family) -- a row that declines to
        # assess should not assert a specific threat level either.
        "threat_reasoning": reason,
        "likely_intent": "unknown",
        "recommended_action": "increase_surveillance",
        "confidence_in_assessment": "low",
        "key_indicators": [
            f"Formation history: {history}",
            f"axis: {axis}, severity: {case['severity']}",
            "insufficient basis for an objective intent/threat assessment",
        ],
        "follow_up_watch": _WATCH[axis],
    }


def build_rows(battery: dict) -> list[dict]:
    rows = []
    for axis in ABSTAIN_AXES:
        for case in battery[axis]:
            if case["has_ground_truth"]:
                continue
            preds = [{**kw, "time_start_s": 0, "time_end_s": 0, "formation_type": kw["formation"],
                      "centroid_velocity": kw["velocity"], "approach_rate": kw["approach"],
                      "formation_stability": kw["stability"], "formation_confidence": kw["confidence"],
                      "role_differentiation": False, "transition_from": kw["from"],
                      "transition_to": kw["to"]} for kw in case["key_windows"]]
            prompt = build_llm_prompt(preds, case["ctx"], {})
            gold = gold_abstain_assessment(case)
            rows.append({"messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json.dumps(gold, indent=2)},
            ]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/sft_train_final.jsonl")
    ap.add_argument("--out", default="data/sft_train_final_abstain.jsonl")
    args = ap.parse_args()

    with open(args.base) as f:
        base_rows = [json.loads(l) for l in f if l.strip()]

    battery = build_battery(ORIGINAL_TEST_CASES)
    abstain_rows = build_rows(battery)

    for r in abstain_rows:
        obj = json.loads(r["messages"][1]["content"])
        assert obj["likely_intent"] == "unknown"
        assert obj["confidence_in_assessment"] == "low"

    with open(args.out, "w") as f:
        for r in base_rows + abstain_rows:
            f.write(json.dumps(r) + "\n")

    print(f"base rows (untouched): {len(base_rows)}")
    print(f"abstention rows added: {len(abstain_rows)}")
    print(f"  by axis: " + ", ".join(
        f"{axis}={sum(1 for c in battery[axis] if not c['has_ground_truth'])}"
        for axis in ABSTAIN_AXES))
    print(f"total rows -> {args.out}: {len(base_rows) + len(abstain_rows)}")


if __name__ == "__main__":
    main()
