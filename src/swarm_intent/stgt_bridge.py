"""
Integration shim: converts the vendored, read-only teammate's STGT
(``stgt/inference.py::sliding_window_inference``) prediction list into OUR
tactical context (``context_spec`` vocabulary + ``calibration`` thresholds) --
NOT upstream's ``build_tactical_context``/``build_llm_prompt`` (see
``stgt/README.md`` for why those weren't vendored).

Four correctness issues this module exists to fix (AUDIT.md sec V, step 4):

1. ``predict_v2``'s ``transition_from``/``transition_to`` fields are the
   classifier's top-2 probability RANKS for a single window (only populated
   when that window's OWN predicted class is "transitioning"), not a temporal
   "what did the swarm change from/to over time" signal. This module NEVER
   reads those two fields -- transitions are derived purely from consecutive
   ``formation_type`` values across the window sequence, exactly like this
   repo's own ``inference.py::build_tactical_context`` already does for its
   ``transitions_detected`` list (that part was already correct; the bug this
   guards against is a hypothetical future caller keying ``RULES``
   (``llm_finetuning/build_sft_dataset.py``) on the per-window fields instead).

2. Upstream's ``build_tactical_context`` (not vendored) dedupes formation
   history with ``dict.fromkeys``, which collapses ANY repeated value anywhere
   in the sequence, not just consecutive runs -- an oscillation like
   ``dispersed -> encirclement -> dispersed`` collapses to
   ``dispersed -> encirclement``, silently erasing the return trip. This
   module collapses only consecutive repeats (``itertools.groupby``), which
   preserves oscillation while still not spamming one entry per window.

3. ``RULES`` (``build_sft_dataset.py``) is keyed EXCLUSIVELY on
   ``BASE_FORMATIONS x BASE_FORMATIONS`` (49 entries) -- it has no entry for
   the classifier's own "transitioning" class, nor for any other string. A
   window whose ``formation_type`` isn't a member of ``BASE_FORMATIONS`` is
   therefore un-lookupable and must not silently flow into the transition /
   dominant-formation computation as if it were a real geometry. Such windows
   are replaced with the sentinel ``"unknown"`` (matching
   ``OUTPUT_SCHEMA``'s own ``likely_intent: ... / unknown`` abstention word),
   excluded from transition/dominant-formation logic, and if EVERY window in
   the batch is unknown the whole context abstains (``summary["abstain"] =
   True``) rather than fabricating a confident-looking narrative from nothing.

4. ``dispersed`` and ``converging`` share IDENTICAL base geometry in
   ``data_gen.py`` (same ``rng.uniform`` branch) -- when their classifier
   probabilities are close, the predicted label is close to a coin flip. This
   matters because ``converging -> encirclement`` is one of the more severe
   ``RULES`` entries; a window flagged ambiguous here is a place a downstream
   assessment should hedge, not escalate confidently on a near-50/50 read.

Also caps ``key_windows`` at a fixed maximum (long streams over many strides
would otherwise grow the prompt unboundedly) and never emits a character
absent from every training prompt (e.g. upstream's own ``build_tactical_context``
emits a literal "warning" glyph the fine-tuned models have never seen).
"""
from __future__ import annotations

from itertools import groupby

import numpy as np

from . import context_spec as spec
from .calibration import AbsoluteCalibrator, Calibrator
from .config import BASE_FORMATIONS

UNKNOWN_FORMATION = "unknown"
DEFAULT_MAX_KEY_WINDOWS = 10
DISPERSED_CONVERGING_AMBIGUITY_MARGIN = 0.15
DISALLOWED_CHARACTERS = ("⚠",)  # "⚠" -- never in any training prompt


def _validate_formation(name) -> str:
    """Anything not in BASE_FORMATIONS (this INCLUDES the classifier's own
    "transitioning" class -- RULES has no entry for it) becomes the explicit
    abstention sentinel rather than flowing through as if it were real."""
    return name if name in BASE_FORMATIONS else UNKNOWN_FORMATION


def _is_ambiguous_dispersed_converging(class_probabilities: dict) -> bool:
    if not class_probabilities:
        return False
    d, c = class_probabilities.get("dispersed"), class_probabilities.get("converging")
    if d is None or c is None:
        return False
    return abs(d - c) < DISPERSED_CONVERGING_AMBIGUITY_MARGIN


def _select_key_window_indices(predictions, formation_seq, ambiguous_flags, max_key_windows):
    """Priority order: first, last, unknown-formation windows, low-confidence
    windows, ambiguous windows -- deduped, capped at max_key_windows, THEN
    sorted chronologically. Priority-first (not index-first) truncation is
    what guarantees first/last survive the cap even when many earlier windows
    are also salient."""
    n = len(predictions)
    ordered = []

    def add(i):
        if i not in ordered:
            ordered.append(i)

    add(0)
    add(n - 1)
    for i, f in enumerate(formation_seq):
        if f == UNKNOWN_FORMATION:
            add(i)
    for i, p in enumerate(predictions):
        if p["formation_confidence"] < 0.65:
            add(i)
    for i, amb in enumerate(ambiguous_flags):
        if amb:
            add(i)
    return sorted(ordered[:max_key_windows])


def bridge_predictions(predictions: list[dict], calibrator: "Calibrator | None" = None,
                       max_key_windows: int = DEFAULT_MAX_KEY_WINDOWS):
    """predictions: output of ``swarm_intent.stgt.inference.sliding_window_inference``
    (or a hand-assembled list of ``predict_v2`` dicts). Returns
    ``(tactical_context: str, summary: dict, key_windows: list[dict])`` in the same
    shape as ``inference.py``'s ``build_tactical_context``/``build_llm_prompt``, with
    the four fixes in this module's docstring applied.
    """
    calibrator = calibrator or AbsoluteCalibrator()
    n = len(predictions)
    if n == 0:
        return ("No predictions available.",
                {"abstain": True, "abstain_reason": "no predictions", "n_windows": 0}, [])

    formation_seq = [_validate_formation(p["formation_type"]) for p in predictions]
    n_unknown = sum(1 for f in formation_seq if f == UNKNOWN_FORMATION)

    # (1) transitions: consecutive-pair, temporal, NEVER predict_v2's transition_from/to.
    transitions = []
    for i in range(1, n):
        a, b = formation_seq[i - 1], formation_seq[i]
        if UNKNOWN_FORMATION in (a, b):
            continue
        if a != b:
            transitions.append({"at_time_s": predictions[i]["time_start_s"], "from": a, "to": b})

    # (2) formation_history: collapse only CONSECUTIVE repeats -- preserves oscillation.
    formation_history = [k for k, _ in groupby(formation_seq)]

    valid_formations = [f for f in formation_seq if f != UNKNOWN_FORMATION]
    if not valid_formations:
        summary = {
            "abstain": True,
            "abstain_reason": f"all {n} window(s) classified outside BASE_FORMATIONS "
                              f"(transitioning / unrecognized) -- no reliable formation signal",
            "formation_history": formation_history, "n_windows": n, "n_unknown_windows": n_unknown,
        }
        context = ("No reliable formation classification in this observation window "
                  "(every window's classification fell outside the known formation "
                  "vocabulary).")
        return context, summary, []

    dominant = max(set(valid_formations), key=valid_formations.count)

    velocities = [p["centroid_velocity"] for p in predictions]
    mid = n // 2
    delta_v = (np.mean(velocities[mid:]) - np.mean(velocities[:mid])) if mid > 0 else 0.0
    vel_trend = calibrator.velocity_trend(float(delta_v))

    stabilities = [p["formation_stability"] for p in predictions]
    mean_stability = float(np.mean(stabilities))
    early = np.mean(stabilities[:max(1, mid)])
    late = np.mean(stabilities[mid:]) if mid > 0 else stabilities[-1]
    stab_trend = calibrator.stability_trend(float(early), float(late))

    approach = [p["approach_rate"] for p in predictions]
    mean_approach = float(np.mean(approach))
    approach_summary = calibrator.spread_dynamics(mean_approach)

    role_true_count = sum(1 for p in predictions if p.get("role_differentiation"))
    role_str = calibrator.role_differentiation(role_true_count, n)
    role_flag = role_str == spec.ROLE_DIFFERENTIATION_PRESENT

    confidences = [p["formation_confidence"] for p in predictions]
    mean_conf, low_conf = float(np.mean(confidences)), sum(1 for c in confidences if c < 0.6)

    # (4) per-window dispersed/converging ambiguity.
    ambiguous_flags = [_is_ambiguous_dispersed_converging(p.get("class_probabilities", {}))
                       for p in predictions]
    n_ambiguous = sum(ambiguous_flags)

    lines = [
        f"Observation window: {predictions[0]['time_start_s']}s - {predictions[-1]['time_end_s']}s "
        f"({n} windows)",
        f"Dominant formation: {dominant}",
        f"Formation history: {' -> '.join(formation_history)}",
    ]
    if n_unknown:
        lines.append(f"{n_unknown} window(s) had an unrecognized/non-base formation "
                     f"classification and were excluded from transition analysis.")
    lines += ([f"Transition at t={t['at_time_s']}s: {t['from']} -> {t['to']}" for t in transitions]
             or [spec.NO_TRANSITIONS_DETECTED])
    lines += [
        f"Velocity trend: {vel_trend} (delta_v={delta_v:+.2f})",
        f"Formation stability: {stab_trend} (mean={mean_stability:.2f})",
        f"Spread dynamics: {approach_summary} (mean approach_rate={mean_approach:.3f})",
        f"Role differentiation: {role_str}",
        spec.CONFIDENCE_LINE_TEMPLATE.format(mean_conf=mean_conf, low_conf=low_conf),
    ]
    if n_ambiguous:
        lines.append(f"{n_ambiguous} window(s) show near-equal dispersed/converging classifier "
                     f"probability (within {DISPERSED_CONVERGING_AMBIGUITY_MARGIN}) -- these two "
                     f"formations share near-identical base geometry, treat formation identity "
                     f"for those windows with caution.")

    summary = {
        "abstain": False,
        "dominant_formation": dominant, "formation_history": formation_history,
        "transitions_detected": transitions, "velocity_trend": vel_trend,
        "delta_velocity": round(float(delta_v), 3), "stability_trend": stab_trend,
        "mean_stability": round(mean_stability, 3), "spread_dynamics": approach_summary,
        "mean_approach_rate": round(mean_approach, 3), "role_differentiation": role_flag,
        "mean_confidence": round(mean_conf, 3), "low_conf_windows": low_conf, "n_windows": n,
        "n_unknown_windows": n_unknown, "n_ambiguous_dispersed_converging_windows": n_ambiguous,
    }

    # (3) key_windows, capped.
    key_idx = _select_key_window_indices(predictions, formation_seq, ambiguous_flags, max_key_windows)
    key_windows = [{
        "t": f"{predictions[i]['time_start_s']}-{predictions[i]['time_end_s']}s",
        "formation": formation_seq[i],
        "confidence": predictions[i]["formation_confidence"],
        "velocity": predictions[i]["centroid_velocity"],
        "approach": predictions[i]["approach_rate"],
        "stability": predictions[i]["formation_stability"],
        "ambiguous_dispersed_converging": ambiguous_flags[i],
    } for i in key_idx]

    context_text = "\n".join(lines)
    for ch in DISALLOWED_CHARACTERS:
        assert ch not in context_text, f"disallowed character {ch!r} in generated context"
    return context_text, summary, key_windows
