"""
Evaluation pipeline for v5-a (and, later, v5-b/c/d) fine-tuned checkpoints.

Metrics, matching this project's established conventions -- checked against
AUDIT.md / src/swarm_intent/llm/evaluate.py / llm_finetuning/eval_real_stgt_output.py
before writing anything new here, per the standing instruction not to invent
metrics ad hoc:

  - per-class threat accuracy (low/medium/high/critical), each with n and a
    Wilson 95% CI. An explicit low-n caveat is printed and recorded whenever
    n<10 -- critical has been n=2 for most of this project's history
    (AUDIT.md: "not statistically meaningful, reported anyway"); never report
    it as a bare percentage without that context.
  - accuracy_when_answerable / abstention_rate_when_unanswerable /
    over_abstention_rate as three SEPARATE fields, never blended into one
    number (src/swarm_intent/llm/evaluate.py's module docstring documents the
    original bug this avoids: an always-abstaining case scored 0.0 accuracy
    AND 1.0 hallucination_rate simultaneously -- double-penalizing the same
    abstention).
  - under-escalation and over-escalation reported SEPARATELY and directionally
    (AUDIT.md sec AC/AD: under-escalation dominates every LLM-in-the-loop
    system measured so far, by a wide margin -- e.g. 15.7% under vs 4.8% over
    on pipeline_v2's real-output eval; collapsing both into one "escalation
    error" number hides which direction actually needs fixing).
  - JSON schema-validity rate: fraction of raw outputs that parse as JSON and
    contain every OUTPUT_SCHEMA key.
  - critical-pair tactical accuracy: same low-n caveat as the per-class table.

This session only exercises --dry-run/--mock mode: a stub in-process predictor
returns synthetic outputs (some correct, some deliberately wrong in each
direction, one schema-invalid) so every metric branch below is demonstrated
end-to-end. There is no real v5-a checkpoint yet (train_sft_v5.py has only
been run with --dry-run this session) -- do not point --adapter at a real path
and expect this to do real inference; that code path is scaffolded but not
exercised or trusted here.

Usage:
    python llm_finetuning/eval_sft_v5.py --mock

Output: logs/eval_v5_results.json -- see OUTPUT_SCHEMA_DOC below for the exact
JSON structure written.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from swarm_intent.llm.prompts import is_abstention, match_threat  # noqa: E402
from swarm_intent.inference import OUTPUT_SCHEMA  # noqa: E402

THREAT_ORDER = ("low", "medium", "high", "critical")
LOW_N_THRESHOLD = 10

# The 2 distinct critical RULES pairs (both directions of the same compound
# escalation event) -- see docs/RULES_EXTENSION_PROPOSAL.md for the full
# context on why this tier is so thin.
CRITICAL_PAIRS = {("converging", "encirclement"), ("encirclement", "converging")}

OUTPUT_SCHEMA_DOC = """
logs/eval_v5_results.json structure:
{
  "meta": {"mode": "mock"|"real", "adapter_path": str|null, "n_cases": int,
           "generated_at_utc": str},
  "per_class_threat_accuracy": {
    "<low|medium|high|critical>": {"n": int, "accuracy": float|null,
                                    "ci_lo": float|null, "ci_hi": float|null,
                                    "low_n_caveat": bool}
  },
  "answerability": {
    "accuracy_when_answerable": float|null, "n_answerable": int,
    "abstention_rate_when_unanswerable": float|null, "n_unanswerable": int,
    "over_abstention_rate": float|null
  },
  "escalation": {"n": int, "correct": float, "under_escalated": float,
                 "over_escalated": float, "abstained": float},
  "schema_validity_rate": {"rate": float, "n_valid": int, "n_total": int},
  "critical_pair_tactical_accuracy": {"n": int, "accuracy": float|null,
                                       "low_n_caveat": bool}
}
"""


def wilson_ci95(k: int, n: int):
    if n == 0:
        return (0.0, 0.0, 0.0)
    import math
    z = 1.959964
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - margin), min(1.0, center + margin))


def mock_cases() -> list[dict]:
    """Small synthetic battery covering all 4 threat tiers plus one
    unanswerable (multi-hop) case, so every metric branch below is exercised.
    NOT drawn from real STGT output or the real corpus -- purely for pipeline
    validation in mock mode."""
    return [
        {"name": "steady_low", "pair": ("column", "column"), "expected_threat": "low",
         "has_ground_truth": True},
        {"name": "transition_medium", "pair": ("shield", "diamond"), "expected_threat": "medium",
         "has_ground_truth": True},
        {"name": "transition_high", "pair": ("column", "converging"), "expected_threat": "high",
         "has_ground_truth": True},
        {"name": "critical_pair", "pair": ("converging", "encirclement"), "expected_threat": "critical",
         "has_ground_truth": True},
        {"name": "under_escalation_case", "pair": ("dispersed", "converging"), "expected_threat": "high",
         "has_ground_truth": True},
        {"name": "over_escalation_case", "pair": ("column", "dispersed"), "expected_threat": "low",
         "has_ground_truth": True},
        {"name": "over_abstention_case", "pair": ("shield", "column"), "expected_threat": "medium",
         "has_ground_truth": True},
        {"name": "multi_hop_unanswerable", "pair": None, "expected_threat": None,
         "has_ground_truth": False},
    ]


def mock_predict(case: dict) -> str:
    """Returns a raw JSON string (as if decoded from the model), deliberately
    varied so every downstream metric branch is exercised: correct answers,
    one under-escalation, one over-escalation, one over-abstention (abstains
    on an answerable case), one correct abstention (unanswerable case), and
    one schema-invalid / malformed output."""
    base = {
        "situation_summary": "mock", "threat_reasoning": "mock",
        "confidence_in_assessment": "high", "key_indicators": ["a", "b", "c"],
        "follow_up_watch": "mock", "recommended_action": "monitor",
    }
    if case["name"] == "under_escalation_case":
        return json.dumps({**base, "threat_level": "low", "likely_intent": "surveillance"})
    if case["name"] == "over_escalation_case":
        return json.dumps({**base, "threat_level": "high", "likely_intent": "approach"})
    if case["name"] == "over_abstention_case":
        return json.dumps({**base, "threat_level": "unknown", "likely_intent": "unknown"})
    if case["name"] == "multi_hop_unanswerable":
        return json.dumps({**base, "threat_level": "unknown", "likely_intent": "unknown"})
    if case["name"] == "critical_pair":
        # deliberately malformed (missing closing brace) to exercise
        # schema_validity_rate < 100%
        return '{"threat_level": "critical", "likely_intent": "encircle"'
    return json.dumps({**base, "threat_level": case["expected_threat"], "likely_intent": "surveillance"})


def parse_and_validate(raw: str) -> tuple[dict | None, bool]:
    """Returns (parsed_dict_or_None, schema_valid). schema_valid requires both
    valid JSON and every OUTPUT_SCHEMA key present."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None, False
    if not isinstance(parsed, dict) or not set(OUTPUT_SCHEMA.keys()) <= parsed.keys():
        return parsed if isinstance(parsed, dict) else None, False
    return parsed, True


def evaluate(cases: list[dict], predict_fn) -> dict:
    raw_by_case = {c["name"]: predict_fn(c) for c in cases}
    parsed_by_case = {}
    valid_by_case = {}
    for name, raw in raw_by_case.items():
        parsed, valid = parse_and_validate(raw)
        parsed_by_case[name] = parsed
        valid_by_case[name] = valid

    n_valid = sum(valid_by_case.values())
    schema_validity_rate = {"rate": n_valid / len(cases) if cases else 0.0,
                            "n_valid": n_valid, "n_total": len(cases)}

    gt_cases = [c for c in cases if c["has_ground_truth"]]
    no_gt_cases = [c for c in cases if not c["has_ground_truth"]]

    # --- per-class threat accuracy ---
    per_class = {}
    for level in THREAT_ORDER:
        level_cases = [c for c in gt_cases if c["expected_threat"] == level]
        hits, scored = 0, 0
        for c in level_cases:
            parsed = parsed_by_case[c["name"]]
            if parsed is None:
                continue  # unparseable -- excluded from accuracy, counted in schema_validity_rate only
            intent = parsed.get("likely_intent", "")
            if is_abstention(intent):
                continue  # abstained -- excluded from accuracy denominator, see answerability section
            scored += 1
            if match_threat(parsed.get("threat_level", ""), level):
                hits += 1
        if scored:
            p, lo, hi = wilson_ci95(hits, scored)
            per_class[level] = {"n": scored, "accuracy": p, "ci_lo": lo, "ci_hi": hi,
                                "low_n_caveat": scored < LOW_N_THRESHOLD}
        else:
            per_class[level] = {"n": 0, "accuracy": None, "ci_lo": None, "ci_hi": None,
                                "low_n_caveat": True}

    # --- answerability: accuracy_when_answerable / abstention_rate_when_unanswerable / over_abstention_rate ---
    answerable_hits, answerable_scored, answerable_abstained = 0, 0, 0
    for c in gt_cases:
        parsed = parsed_by_case[c["name"]]
        if parsed is None:
            continue
        intent = parsed.get("likely_intent", "")
        if is_abstention(intent):
            answerable_abstained += 1
            continue
        answerable_scored += 1
        if match_threat(parsed.get("threat_level", ""), c["expected_threat"]):
            answerable_hits += 1
    accuracy_when_answerable = answerable_hits / answerable_scored if answerable_scored else None
    over_abstention_rate = (answerable_abstained / len(gt_cases)) if gt_cases else None

    unanswerable_abstained, unanswerable_total = 0, 0
    for c in no_gt_cases:
        parsed = parsed_by_case[c["name"]]
        unanswerable_total += 1
        if parsed is not None and is_abstention(parsed.get("likely_intent", "")):
            unanswerable_abstained += 1
    abstention_rate_when_unanswerable = (unanswerable_abstained / unanswerable_total
                                         if unanswerable_total else None)

    answerability = {
        "accuracy_when_answerable": accuracy_when_answerable, "n_answerable": answerable_scored,
        "abstention_rate_when_unanswerable": abstention_rate_when_unanswerable,
        "n_unanswerable": unanswerable_total,
        "over_abstention_rate": over_abstention_rate,
    }

    # --- escalation direction: correct / under / over / abstained, partition of gt_cases ---
    correct = under_esc = over_esc = abstained = 0
    for c in gt_cases:
        parsed = parsed_by_case[c["name"]]
        if parsed is None or is_abstention(parsed.get("likely_intent", "")):
            abstained += 1
            continue
        pred = parsed.get("threat_level", "").strip().lower()
        expected = c["expected_threat"]
        if pred == expected:
            correct += 1
        elif pred in THREAT_ORDER and THREAT_ORDER.index(pred) < THREAT_ORDER.index(expected):
            under_esc += 1
        elif pred in THREAT_ORDER:
            over_esc += 1
        else:
            abstained += 1  # unparseable/garbage threat_level treated as non-answer, not scored either direction
    n_gt = len(gt_cases)
    escalation = {"n": n_gt,
                 "correct": correct / n_gt if n_gt else 0.0,
                 "under_escalated": under_esc / n_gt if n_gt else 0.0,
                 "over_escalated": over_esc / n_gt if n_gt else 0.0,
                 "abstained": abstained / n_gt if n_gt else 0.0}

    # --- critical-pair tactical accuracy ---
    crit_cases = [c for c in gt_cases if c["pair"] in CRITICAL_PAIRS]
    crit_hits, crit_scored = 0, 0
    for c in crit_cases:
        parsed = parsed_by_case[c["name"]]
        if parsed is None or is_abstention(parsed.get("likely_intent", "")):
            continue
        crit_scored += 1
        if match_threat(parsed.get("threat_level", ""), "critical"):
            crit_hits += 1
    critical_pair_tactical_accuracy = {
        "n": crit_scored, "accuracy": (crit_hits / crit_scored if crit_scored else None),
        "low_n_caveat": crit_scored < LOW_N_THRESHOLD,
    }

    return {
        "per_class_threat_accuracy": per_class,
        "answerability": answerability,
        "escalation": escalation,
        "schema_validity_rate": schema_validity_rate,
        "critical_pair_tactical_accuracy": critical_pair_tactical_accuracy,
    }


def print_report(results: dict):
    print("=" * 90)
    print("per-class threat accuracy (Wilson 95% CI)")
    print("=" * 90)
    print("| threat | n | accuracy | 95% CI | caveat |")
    print("|---|---|---|---|---|")
    for level in THREAT_ORDER:
        r = results["per_class_threat_accuracy"][level]
        if r["accuracy"] is None:
            print(f"| {level} | 0 | n/a | n/a | no scored cases |")
        else:
            caveat = f"LOW N (<{LOW_N_THRESHOLD}), not statistically meaningful" if r["low_n_caveat"] else ""
            print(f"| {level} | {r['n']} | {r['accuracy']:.1%} | "
                 f"[{r['ci_lo']:.1%}, {r['ci_hi']:.1%}] | {caveat} |")

    print()
    print("=" * 90)
    print("answerability (three separate columns, never blended)")
    print("=" * 90)
    a = results["answerability"]
    aw = f"{a['accuracy_when_answerable']:.1%}" if a["accuracy_when_answerable"] is not None else "n/a"
    ab = (f"{a['abstention_rate_when_unanswerable']:.1%}"
         if a["abstention_rate_when_unanswerable"] is not None else "n/a")
    oa = f"{a['over_abstention_rate']:.1%}" if a["over_abstention_rate"] is not None else "n/a"
    print(f"accuracy_when_answerable: {aw} (n={a['n_answerable']})")
    print(f"abstention_rate_when_unanswerable: {ab} (n={a['n_unanswerable']})")
    print(f"over_abstention_rate: {oa}")

    print()
    print("=" * 90)
    print("escalation direction (separate, never pooled into one 'escalation error')")
    print("=" * 90)
    e = results["escalation"]
    print(f"n={e['n']}  correct={e['correct']:.1%}  "
         f"UNDER-escalated={e['under_escalated']:.1%}  OVER-escalated={e['over_escalated']:.1%}  "
         f"abstained={e['abstained']:.1%}")

    print()
    s = results["schema_validity_rate"]
    print(f"schema_validity_rate: {s['rate']:.1%} ({s['n_valid']}/{s['n_total']})")

    print()
    c = results["critical_pair_tactical_accuracy"]
    if c["accuracy"] is not None:
        caveat = " (LOW N, not statistically meaningful)" if c["low_n_caveat"] else ""
        print(f"critical_pair_tactical_accuracy: {c['accuracy']:.1%} (n={c['n']}){caveat}")
    else:
        print(f"critical_pair_tactical_accuracy: n/a (n={c['n']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", "--dry-run", dest="mock", action="store_true",
                    help="use a synthetic stub predictor instead of a real checkpoint. "
                         "The ONLY mode this script has been run in so far -- no real "
                         "v5-a adapter exists yet.")
    ap.add_argument("--adapter", default=None,
                    help="path to a real LoRA adapter (NOT exercised this session -- "
                         "scaffolded for when v5-a training actually completes).")
    ap.add_argument("--out", default="logs/eval_v5_results.json")
    args = ap.parse_args()

    if not args.mock and args.adapter is None:
        raise SystemExit("Pass --mock for a pipeline dry-run, or --adapter <path> for real "
                         "inference (not available/exercised this session).")
    if not args.mock:
        raise SystemExit("Real inference is scaffolded but not implemented/exercised this "
                         "session -- only --mock is supported right now. See module docstring.")

    cases = mock_cases()
    results = evaluate(cases, mock_predict)
    results["meta"] = {"mode": "mock", "adapter_path": None, "n_cases": len(cases),
                       "generated_at_utc": datetime.now(timezone.utc).isoformat()}

    print_report(results)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
