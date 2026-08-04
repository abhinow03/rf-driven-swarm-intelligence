"""
Two non-LLM(-ish) baselines for the 4-way eval. Both expose the same
``run_case(case) -> (assessment_dict, tactical_context_str)`` callback signature
``evaluate_llm`` already expects (see ``evaluate_finetuned.py`` for the existing
LocalHFClient precedent).

a. rules_lookup — NO model call. Builds the same synth_context() tactical-context
   text every other system in the 4-way eval sees, then regexes the (from, to)
   formation pair back OUT of that text — the way any real downstream consumer would
   have to, not by reading case["formation_a"]/["formation_b"] directly — looks the
   pair up in RULES, and emits schema-valid JSON with canned (non-model) prose. If
   extraction fails, or the extracted pair isn't a RULES key, it returns
   likely_intent="unknown" and abstains. It does NOT fall back to DEFAULT_RULE —
   abstention on an unparseable/unknown scenario is the intended behaviour, not a
   bug to paper over. This is the control: what a 49-entry dict already achieves
   on clean synthetic inputs, with zero model involved.

b. rules_in_prompt — the base Qwen2.5-7B-Instruct model, no adapter, via the same
   LocalHFClient code path as every other LLM-backed system, but with
   llm_finetuning/RULES.txt's full text passed as ``system_prompt`` (see the small
   additive change to LocalHFClient in src/swarm_intent/llm/client.py — default
   system_prompt=None, so every existing caller is unaffected).
"""
from __future__ import annotations

import os
import re
import sys
from random import Random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from swarm_intent.inference import build_llm_prompt  # noqa: E402

from build_sft_dataset import RULES, synth_context  # type: ignore  # noqa: E402

RULES_TXT_PATH = os.path.join(os.path.dirname(__file__), "RULES.txt")

# Matches synth_context()'s exact phrasing (llm_finetuning/build_sft_dataset.py:140).
# NOTE: production's build_tactical_context() phrases this line differently
# ("Transition at t=...s: A -> B", no "detected") — this regex intentionally targets
# only the synthetic-context format every system in this eval is actually fed, per
# AUDIT.md sec H.
_TRANSITION_RE = re.compile(r"Transition detected at t=[\d.]+s:\s*(\w+)\s*->\s*(\w+)")
_DOMINANT_RE = re.compile(r"Dominant formation:\s*(\w+)")
_NO_TRANSITION_RE = re.compile(r"No formation transitions detected\.")


def _extract_pair(ctx: str):
    """Regex the (from, to) formation pair out of a tactical-context string — does
    NOT use any ground-truth case values. Returns (from, to) or None."""
    m = _TRANSITION_RE.search(ctx)
    if m:
        return m.group(1), m.group(2)
    if _NO_TRANSITION_RE.search(ctx):
        dm = _DOMINANT_RE.search(ctx)
        if dm:
            return dm.group(1), dm.group(1)
    return None


def _canned_assessment(threat: str, intent: str, action: str, from_f: str, to_f: str) -> dict:
    steady = from_f == to_f
    return {
        "situation_summary": (f"Swarm holding {from_f} formation." if steady
                               else f"Swarm transitioning from {from_f} to {to_f} formation."),
        "threat_level": threat,
        "threat_reasoning": f"Rule-table lookup for ({from_f} -> {to_f}): {threat} threat.",
        "likely_intent": intent,
        "recommended_action": action,
        "confidence_in_assessment": "high",
        "key_indicators": ([f"{from_f} steady-state", "rule-table lookup"] if steady
                            else [f"{from_f} -> {to_f} transition", "rule-table lookup"]),
        "follow_up_watch": "Monitor for the next formation change.",
    }


def _abstain(reason: str) -> dict:
    # threat_level="unknown" is schema-legal (prompts.py THREAT_FAMILIES/
    # inference.py OUTPUT_SCHEMA, added in AUDIT.md sec AA step 3 -- this
    # function predates that fix and used to emit "unknown" here despite it not
    # being schema-legal at the time; that asymmetry with likely_intent, which
    # always had an "unknown" family, is documented in AUDIT.md sec D). Also
    # note abstained responses are excluded from accuracy/hallucination scoring
    # entirely regardless (see evaluate.py) — this field being schema-valid now
    # is a correctness fix, not something the scoring logic depended on.
    return {
        "situation_summary": f"Unable to determine formation transition from context ({reason}).",
        "threat_level": "unknown",
        "threat_reasoning": "Formation pair could not be extracted from the tactical context.",
        "likely_intent": "unknown",
        "recommended_action": "monitor",
        "confidence_in_assessment": "low",
        "key_indicators": [],
        "follow_up_watch": "Re-acquire a parseable tactical context.",
    }


def _assess_from_context(ctx: str) -> dict:
    """Core rules_lookup decision logic given an arbitrary tactical-context string.
    Split out from make_rules_lookup_run_case so it's directly unit-testable without
    going through synth_context() (see tests/test_abstention.py)."""
    pair = _extract_pair(ctx)
    if pair is None:
        return _abstain("regex could not locate a formation pair in the context")
    rule = RULES.get(pair)
    if rule is None:
        # Explicitly NOT falling back to DEFAULT_RULE — abstention is the point.
        return _abstain(f"pair {pair} is not a RULES key")
    threat, intent, action = rule
    return _canned_assessment(threat, intent, action, *pair)


def make_rules_lookup_run_case(seed: int = 0):
    """No model call. Returns a run_case(case) -> (assessment, ctx) callback."""
    rng = Random(seed)

    def run_case(case):
        ctx, _key_windows = synth_context(case["formation_a"], case["formation_b"], rng)
        return _assess_from_context(ctx), ctx

    return run_case


def make_rules_in_prompt_run_case(client, seed: int = 0):
    """client: an LLMClient (e.g. LocalHFClient constructed with the RULES.txt text
    as system_prompt). Returns a run_case(case) -> (assessment, ctx) callback."""
    rng = Random(seed)

    def run_case(case):
        ctx, key_windows = synth_context(case["formation_a"], case["formation_b"], rng)
        preds = [{**kw, "time_start_s": 0, "time_end_s": 0, "formation_type": kw["formation"],
                  "centroid_velocity": kw["velocity"], "approach_rate": kw["approach"],
                  "formation_stability": kw["stability"], "formation_confidence": kw["confidence"],
                  "role_differentiation": False, "transition_from": kw["from"],
                  "transition_to": kw["to"]} for kw in key_windows]
        prompt = build_llm_prompt(preds, ctx, {})
        assessment = client.complete(prompt)
        return assessment, ctx

    return run_case


def make_batched_run_case(client, test_cases: list, n_runs: int, batch_size: int, seed: int = 0):
    """Step 2 of the throughput-optimization session (AUDIT.md sec U): a
    run_case(case) -> (assessment, ctx) callback with the exact same signature
    and RNG-driven ctx values as make_rules_in_prompt_run_case, but generated
    via ONE batched pass through client.generate_batch instead of one call per
    case per run.

    evaluate_llm calls run_case(case) in a fixed order -- case-major, run-minor
    (case1 x n_runs, then case2 x n_runs, ...), advancing a single shared
    Random(seed) each call via synth_context(). This function pre-computes ALL
    (case, run) ctx/prompt pairs up front in that SAME order (so the ctx text
    for a given (case, run) is byte-identical to the unbatched path -- same rng
    draws, same seed), generates every completion in one batched pass, then
    replays them via a closure that just advances an index -- a drop-in
    replacement for make_rules_in_prompt_run_case wherever batching is wanted.
    """
    rng = Random(seed)
    ctxs, prompts = [], []
    for case in test_cases:
        for _ in range(n_runs):
            ctx, key_windows = synth_context(case["formation_a"], case["formation_b"], rng)
            preds = [{**kw, "time_start_s": 0, "time_end_s": 0, "formation_type": kw["formation"],
                     "centroid_velocity": kw["velocity"], "approach_rate": kw["approach"],
                     "formation_stability": kw["stability"], "formation_confidence": kw["confidence"],
                     "role_differentiation": False, "transition_from": kw["from"],
                     "transition_to": kw["to"]} for kw in key_windows]
            ctxs.append(ctx)
            prompts.append(build_llm_prompt(preds, ctx, {}))

    assessments = (client.complete_batch(prompts, batch_size=batch_size) if batch_size > 1
                  else [client.complete(p) for p in prompts])

    call_idx = [0]

    def run_case(case):
        i = call_idx[0]
        call_idx[0] += 1
        return assessments[i], ctxs[i]

    return run_case


def load_rules_txt() -> str:
    with open(RULES_TXT_PATH) as f:
        return f.read()
