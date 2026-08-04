"""
Bucket classification (AUDIT.md sec AE): is a swarm observation something a
plain (from, to) -> RULES dict lookup can resolve on its own (A, RESOLVABLE),
something a dict COULD answer but shouldn't be trusted to without a hedge (B,
GUARDABLE), or something no 2-tuple lookup can ever answer regardless of
confidence (C, UNRESOLVABLE)? This is the coverage question behind "why not
just ship the dict" -- pipeline_v2.py routes A to RULES directly, B to a
machine-generated abstention, and C to the fine-tuned LLM layer.

Two input shapes, two entry points, same bucket vocabulary:

  classify_observation(predictions) -- REAL swarm_intent.stgt.inference.
    sliding_window_inference() output (rich per-window dicts, including
    class_probabilities). Built on stgt_bridge.bridge_predictions(), which
    already does the OOV-sentinel / oscillation-preserving / dispersed-
    converging-ambiguity work this module needs -- reused, not reimplemented.
    This is the path used to MEASURE real-world coverage (measure_coverage.py).

  classify_ctx(ctx, key_windows) -- the templated tactical-context TEXT
    representation used by TEST_CASES (build_sft_dataset.synth_context) and
    the degradation/holdout batteries, which have no real per-window
    class_probabilities (synth_context never simulates one) -- the dispersed/
    converging ambiguity guard structurally can never fire on this path,
    disclosed, not a bug. Reuses the same (from, to) extraction regex
    llm_finetuning/baselines.py's composite router already keys its own
    routing decision on, so pipeline_v2's bucket boundary is directly
    comparable to what the existing composite already does. This is the path
    used to EVALUATE pipeline_v2 against the existing 55-case/degradation
    batteries, which carry no real STGT inference output at all.
"""
from __future__ import annotations

import re
from itertools import groupby

from .config import BASE_FORMATIONS
from .stgt_bridge import bridge_predictions, UNKNOWN_FORMATION

BUCKET_A = "A"  # resolvable
BUCKET_B = "B"  # guardable
BUCKET_C = "C"  # unresolvable


def _collapse_consecutive(seq):
    return [k for k, _ in groupby(seq)]


def classify_observation(predictions: list[dict], calibrator=None,
                         max_key_windows: int = 10) -> dict:
    """predictions: swarm_intent.stgt.inference.sliding_window_inference() output
    (or hand-built dicts of the same shape). Returns a dict with keys: bucket
    (A/B/C), subtype (C only, else None), rules_key ((from,to) tuple or None),
    guard_reasons (list[str], B only), context_text, summary, key_windows --
    the last three are bridge_predictions()'s own return values, passed through
    so a caller doesn't have to call it twice."""
    context_text, summary, key_windows = bridge_predictions(predictions, calibrator, max_key_windows)

    def _c(subtype):
        return {"bucket": BUCKET_C, "subtype": subtype, "rules_key": None, "guard_reasons": [],
                "context_text": context_text, "summary": summary, "key_windows": key_windows}

    if summary.get("abstain"):
        return _c("all_unknown")

    history = summary["formation_history"]  # consecutive-collapsed, includes UNKNOWN_FORMATION groups
    if history[-1] == UNKNOWN_FORMATION:
        # the swarm's CURRENT state is unclassifiable (STGT's own "transitioning"
        # class, or any non-BASE_FORMATIONS string) -- covers the degradation
        # battery's terminal_transitioning axis and any real trailing OOV read.
        return _c("terminal_unknown")

    known_history = _collapse_consecutive([f for f in history if f != UNKNOWN_FORMATION])

    if len(known_history) >= 3:
        return _c("oscillation" if known_history[0] == known_history[-1] else "multi_hop")

    key = ((known_history[0], known_history[0]) if len(known_history) == 1
           else (known_history[0], known_history[-1]))
    if key[0] not in BASE_FORMATIONS or key[1] not in BASE_FORMATIONS:
        # defensive/unreachable: _validate_formation (stgt_bridge.py) already maps
        # anything outside BASE_FORMATIONS to UNKNOWN_FORMATION before this point.
        return _c("no_rules_key")

    guard_reasons = []
    if summary["n_unknown_windows"] > 0:
        guard_reasons.append("oov_name")
    if len(known_history) == 2:
        # bridge_predictions' own `dominant` is max(..., key=count) -- ties are
        # broken silently/arbitrarily by set iteration order, which is exactly
        # the case a caller should NOT trust as "the" dominant formation. Count
        # directly rather than trusting summary["dominant_formation"].
        counts = [p["formation_type"] for p in predictions if p["formation_type"] in BASE_FORMATIONS]
        if counts.count(key[0]) == counts.count(key[1]):
            guard_reasons.append("dominant_history_contradiction")
    if summary["n_ambiguous_dispersed_converging_windows"] > 0:
        guard_reasons.append("dispersed_converging_ambiguity")
    if predictions and all(p["formation_confidence"] < 0.6 for p in predictions):
        guard_reasons.append("low_confidence")

    bucket = BUCKET_B if guard_reasons else BUCKET_A
    return {"bucket": bucket, "subtype": None, "rules_key": key, "guard_reasons": guard_reasons,
            "context_text": context_text, "summary": summary, "key_windows": key_windows}


# --- ctx-text representation (TEST_CASES / degradation / holdout batteries) ---

# Matches synth_context()'s and degradation.py's _render_lines()'s exact phrasing
# -- the SAME regex llm_finetuning/baselines.py's _extract_pair uses for the
# composite router, duplicated here (not imported) so this module has no
# dependency on llm_finetuning/ (see pipeline_v2.py for the one place that does).
_TRANSITION_RE = re.compile(r"Transition detected at t=[\d.]+s:\s*(\w+)\s*->\s*(\w+)")
_DOMINANT_RE = re.compile(r"Dominant formation:\s*(\w+)")
_NO_TRANSITION_RE = re.compile(r"No formation transitions detected\.")


def _extract_pair_from_ctx(ctx: str):
    m = _TRANSITION_RE.search(ctx)
    if m:
        return m.group(1), m.group(2)
    if _NO_TRANSITION_RE.search(ctx):
        dm = _DOMINANT_RE.search(ctx)
        if dm:
            return dm.group(1), dm.group(1)
    return None


def classify_ctx(ctx: str, key_windows: list[dict]) -> dict:
    """ctx: a tactical-context string in synth_context()/_render_lines() format.
    key_windows: the case's precomputed key_windows list (each with a
    "confidence" field). Returns the same {bucket, subtype, rules_key,
    guard_reasons} shape as classify_observation (no context_text/summary/
    key_windows passthrough -- caller already has ctx/key_windows)."""
    pair = _extract_pair_from_ctx(ctx)
    if pair is None:
        # "Multiple formation changes detected..." (multi_hop severity>=3,
        # terminal_transitioning) -- no clean 2-tuple to extract at all.
        return {"bucket": BUCKET_C, "subtype": "no_extractable_pair", "rules_key": None,
                "guard_reasons": []}

    guard_reasons = []
    if pair[0] not in BASE_FORMATIONS or pair[1] not in BASE_FORMATIONS:
        guard_reasons.append("oov_name")

    dm = _DOMINANT_RE.search(ctx)
    if dm and dm.group(1) not in pair:
        guard_reasons.append("dominant_history_contradiction")

    if key_windows and all(kw.get("confidence", 1.0) < 0.6 for kw in key_windows):
        guard_reasons.append("low_confidence")

    # dispersed/converging ambiguity has NO signal in this text representation
    # (synth_context() never simulates per-window class_probabilities) -- can
    # never fire here. See module docstring.

    bucket = BUCKET_B if guard_reasons else BUCKET_A
    return {"bucket": bucket, "subtype": None, "rules_key": pair, "guard_reasons": guard_reasons}
