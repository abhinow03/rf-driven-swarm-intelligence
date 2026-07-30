"""
The degradation battery: perturbed variants of the 6 TEST_CASES along five axes,
each with a severity sweep, so results are curves rather than points.

Ground-truth discipline (this is the point of the whole exercise): every generated
case explicitly records has_ground_truth. Where RULES cannot answer a perturbed
input (multi-hop chains, terminal "transitioning"), the CORRECT system behaviour is
abstention, and evaluate_llm (see src/swarm_intent/llm/evaluate.py,
correct_abstention_rate) scores it that way -- abstention on an unanswerable case is
never marked wrong.

rules_lookup is EXPECTED to collapse on axes (a) multi-hop and (b) terminal-
transitioning: it has no RULES key for either, by construction (RULES is keyed on
(BASE_FORMATIONS, BASE_FORMATIONS) 2-tuples only). That collapse -- correctly
abstaining, not silently guessing -- is the finding for those two axes, not a defect
in the baseline. The real question this battery is built to answer is whether the
fine-tuned adapter degrades gracefully on inputs the symbolic lookup structurally
cannot address at all.

Axes:
  a. multi_hop            -- 2/3/4-formation chains (RULES only has 2-tuples)
  b. terminal_transitioning -- history ends in the literal "transitioning" class,
                               which is not itself a member of BASE_FORMATIONS
  c. confidence_decay     -- mean confidence swept 0.95 -> 0.55 (ground truth
                              unchanged -- the formation pair doesn't move, only
                              the confidence metadata degrades)
  d. dropped_lines        -- 0/1/2/3 tactical-context lines removed (sensor
                              dropout); ground truth is lost specifically when the
                              transition/no-transition line (the ONLY line that
                              distinguishes "held steady" from "changed formation"
                              -- Dominant formation always reads form_a regardless)
                              is among the dropped lines
  e. contradictory_cues   -- narrative fields forced into an internally
                              inconsistent combination (ground truth unchanged --
                              RULES answers off the formation pair alone, not the
                              narrative text)
"""
from __future__ import annotations

import hashlib
import os
import sys
from random import Random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from swarm_intent import context_spec as spec
from swarm_intent.inference import build_llm_prompt

from build_sft_dataset import RULES  # type: ignore
from baselines import _assess_from_context  # type: ignore

CHAIN_POOL = ["v_shape", "column", "diamond", "dispersed", "shield"]  # avoids
# encirclement/converging so multi-hop chains don't read as an already-critical
# scenario in their own right, keeping the perturbation about STRUCTURE not threat.


def _stable_seed(*parts) -> int:
    """Deterministic seed derived from `parts`, stable ACROSS process runs -- unlike
    Python's built-in hash() on strings, which is salted per-process (hash
    randomization) and would silently make dropped_lines' line-selection different
    every time this module is imported in a new interpreter. Caught by comparing
    two separate `python -c` runs of build_battery() and finding has_ground_truth
    disagreed on which dropped_lines cases lost their transition line."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _next_chain_formation(used: list[str]) -> str:
    for f in CHAIN_POOL:
        if f not in used:
            return f
    return CHAIN_POOL[len(used) % len(CHAIN_POOL)]


def _render_lines(known_formations: list[str], terminal_transitioning: bool,
                   resolvable: bool, mean_conf: float, low_conf: int,
                   mean_stab: float, stab_word: str, approach: float, spread_word: str,
                   delta_v: float, vel_word: str, dropped_indices: set[int] | None = None):
    """Build the 8-line tactical-context text. resolvable=True + len==2 +
    not terminal_transitioning is the only combination that gets a single clean
    "Transition detected at t=Xs: A -> B" line rules_lookup's regex can extract --
    everything else deliberately does NOT match that regex (chain-summary sentence
    instead), the same way a real multi-hop/unresolved STGT output would give a
    symbolic lookup nothing clean to grab."""
    dominant = known_formations[0]
    history = (" -> transitioning -> ".join(known_formations)
              if len(known_formations) > 1 else known_formations[0])
    if terminal_transitioning:
        history = f"{history} -> transitioning"

    lines = [
        "Observation window: 0.0s - 60.0s (9 overlapping windows)",
        f"Dominant formation: {dominant}",
        f"Formation history: {history}",
    ]
    if len(known_formations) == 1 and not terminal_transitioning:
        lines.append(spec.NO_TRANSITIONS_DETECTED)
    elif resolvable and len(known_formations) == 2 and not terminal_transitioning:
        lines.append(f"Transition detected at t=20.0s: {known_formations[0]} -> {known_formations[1]}")
    else:
        n_hops = len(known_formations) - 1 + (1 if terminal_transitioning else 0)
        lines.append(f"Multiple formation changes detected across the observation "
                     f"window ({n_hops} transitions; see Formation history).")
    lines += [
        f"Velocity trend: {vel_word} (delta_v={delta_v:+.2f})",
        f"Formation stability: {stab_word} (mean={mean_stab:.2f})",
        f"Spread dynamics: {spread_word} (mean approach_rate={approach:.3f})",
        spec.CONFIDENCE_LINE_TEMPLATE.format(mean_conf=mean_conf, low_conf=low_conf),
    ]

    kept_indices = list(range(len(lines)))
    if dropped_indices:
        kept_indices = [i for i in kept_indices if i not in dropped_indices]
        lines = [l for i, l in enumerate(lines) if i not in dropped_indices]
    return "\n".join(lines), kept_indices


def _key_windows(form_a: str, form_b: str, mean_conf: float, mean_stab: float, approach: float):
    return [
        {"t": "0.0-25.0s", "formation": form_a, "confidence": mean_conf,
         "velocity": 0.05, "approach": approach, "stability": mean_stab, "from": None, "to": None},
        {"t": "35.0-60.0s", "formation": form_b, "confidence": mean_conf,
         "velocity": 0.05, "approach": approach, "stability": mean_stab,
         "from": form_a if form_a != form_b else None, "to": form_b if form_a != form_b else None},
    ]


# Neutral narrative defaults for axes that aren't testing velocity/spread/stability
# themselves (a, b, d) -- keeps the axis under test isolated from unrelated noise.
_NEUTRAL = dict(mean_conf=0.85, low_conf=0, mean_stab=0.8, stab_word=spec.STABILITY_HOLDING,
                approach=-0.05, spread_word=spec.SPREAD_STABLE, delta_v=0.02,
                vel_word=spec.VELOCITY_STEADY)


def _base_case_result(case: dict) -> dict:
    return {"expected_intent": case["expected_intent"], "expected_threat": case["expected_threat"],
            "expected_action": case["expected_action"]}


def _make_case(name, axis, severity, base_case, ctx, key_windows, has_ground_truth, **kw):
    d = {"name": name, "axis": axis, "severity": severity, "base_case": base_case["name"],
        "has_ground_truth": has_ground_truth}
    if has_ground_truth:
        d.update(_base_case_result(base_case))
    d.update(kw)
    d["ctx"] = ctx
    d["key_windows"] = key_windows
    return d


def axis_multi_hop(test_cases):
    cases = []
    for base in test_cases:
        form_a, form_b = base["formation_a"], base["formation_b"]
        for severity in (2, 3, 4):
            chain = [form_a, form_b]
            used = [form_a, form_b]
            while len(chain) < severity:
                nxt = _next_chain_formation(used)
                chain.append(nxt)
                used.append(nxt)
            resolvable = (severity == 2)
            ctx, _ = _render_lines(chain, terminal_transitioning=False, resolvable=resolvable, **_NEUTRAL)
            kw = _key_windows(chain[0], chain[-1], _NEUTRAL["mean_conf"], _NEUTRAL["mean_stab"],
                              _NEUTRAL["approach"])
            has_gt = (severity == 2)  # RULES has no entry for chains of length > 2
            cases.append(_make_case(f"{base['name']}__multi_hop__sev{severity}", "multi_hop",
                                    severity, base, ctx, kw, has_gt))
    return cases


def axis_terminal_transitioning(test_cases):
    cases = []
    for base in test_cases:
        form_a, form_b = base["formation_a"], base["formation_b"]
        for severity in (1, 2, 3):
            known = [form_a] if severity == 1 else [form_a, form_b]
            used = list(known)
            while len(known) < severity:
                nxt = _next_chain_formation(used)
                known.append(nxt)
                used.append(nxt)
            ctx, _ = _render_lines(known, terminal_transitioning=True, resolvable=False, **_NEUTRAL)
            kw = _key_windows(known[0], "transitioning", _NEUTRAL["mean_conf"], _NEUTRAL["mean_stab"],
                              _NEUTRAL["approach"])
            # RULES cannot key on "transitioning" (not in BASE_FORMATIONS) at ANY
            # severity here -- correct behaviour is always abstention on this axis.
            cases.append(_make_case(f"{base['name']}__terminal_transitioning__sev{severity}",
                                    "terminal_transitioning", severity, base, ctx, kw, False))
    return cases


def axis_confidence_decay(test_cases):
    cases = []
    for base in test_cases:
        form_a, form_b = base["formation_a"], base["formation_b"]
        for target_mean in (0.95, 0.85, 0.75, 0.65, 0.55):
            offsets = np.linspace(-0.15, 0.15, 9)
            window_confs = [float(max(0.05, min(0.99, target_mean + o))) for o in offsets]
            mean_conf = float(np.mean(window_confs))
            low_conf = sum(1 for c in window_confs if c < 0.6)
            ctx, _ = _render_lines([form_a, form_b], terminal_transitioning=False, resolvable=True,
                                   mean_conf=mean_conf, low_conf=low_conf, mean_stab=0.8,
                                   stab_word=spec.STABILITY_HOLDING, approach=-0.05,
                                   spread_word=spec.SPREAD_STABLE, delta_v=0.02,
                                   vel_word=spec.VELOCITY_STEADY)
            kw = _key_windows(form_a, form_b, mean_conf, 0.8, -0.05)
            cases.append(_make_case(f"{base['name']}__confidence_decay__sev{target_mean}",
                                    "confidence_decay", target_mean, base, ctx, kw, True))
    return cases


def axis_dropped_lines(test_cases):
    cases = []
    for base in test_cases:
        form_a, form_b = base["formation_a"], base["formation_b"]
        for severity in (0, 1, 2, 3):
            rng = Random(_stable_seed(base["name"], severity))
            drop_idx = set(rng.sample(range(8), severity)) if severity else set()
            ctx, kept = _render_lines([form_a, form_b], terminal_transitioning=False, resolvable=True,
                                      dropped_indices=drop_idx, **_NEUTRAL)
            # Index 3 is the transition/no-transition line -- the ONLY line that
            # distinguishes "held steady" from "changed formation" (Dominant
            # formation always reads form_a regardless of whether a transition
            # happened). Ground truth survives iff it survives the drop.
            has_gt = 3 not in drop_idx
            kw = _key_windows(form_a, form_b, _NEUTRAL["mean_conf"], _NEUTRAL["mean_stab"],
                              _NEUTRAL["approach"])
            cases.append(_make_case(f"{base['name']}__dropped_lines__sev{severity}", "dropped_lines",
                                    severity, base, ctx, kw, has_gt, dropped_line_indices=sorted(drop_idx)))
    return cases


def axis_contradictory_cues(test_cases):
    cases = []
    for base in test_cases:
        form_a, form_b = base["formation_a"], base["formation_b"]
        for severity in (1, 2, 3):
            # severity escalates how many narrative fields conflict with each
            # other / with a coherent single story -- the formation pair (and
            # therefore ground truth) never changes.
            stab_word = spec.STABILITY_HOLDING
            mean_conf = 0.85
            if severity >= 2:
                stab_word = spec.STABILITY_DEGRADING  # coherent convergence + falling apart, at once
            if severity >= 3:
                mean_conf = 0.97  # near-certain despite the internally inconsistent picture
            ctx, _ = _render_lines([form_a, form_b], terminal_transitioning=False, resolvable=True,
                                   mean_conf=mean_conf, low_conf=0, mean_stab=0.55, stab_word=stab_word,
                                   approach=-0.4, spread_word=spec.SPREAD_CONVERGING, delta_v=-0.8,
                                   vel_word=spec.VELOCITY_DECELERATING)
            kw = _key_windows(form_a, form_b, mean_conf, 0.55, -0.4)
            cases.append(_make_case(f"{base['name']}__contradictory_cues__sev{severity}",
                                    "contradictory_cues", severity, base, ctx, kw, True))
    return cases


AXES = {
    "multi_hop": axis_multi_hop,
    "terminal_transitioning": axis_terminal_transitioning,
    "confidence_decay": axis_confidence_decay,
    "dropped_lines": axis_dropped_lines,
    "contradictory_cues": axis_contradictory_cues,
}


def build_battery(test_cases) -> dict:
    """Returns {axis_name: [case, ...]}."""
    return {axis: fn(test_cases) for axis, fn in AXES.items()}


# --- run_case factories for the battery (pre-built ctx, unlike baselines.py's
# make_*_run_case which build ctx via synth_context internally each call) ---

def make_rules_lookup_battery_run_case():
    def run_case(case):
        return _assess_from_context(case["ctx"]), case["ctx"]
    return run_case


def make_llm_battery_run_case(client):
    def run_case(case):
        preds = [{**kw, "time_start_s": 0, "time_end_s": 0, "formation_type": kw["formation"],
                  "centroid_velocity": kw["velocity"], "approach_rate": kw["approach"],
                  "formation_stability": kw["stability"], "formation_confidence": kw["confidence"],
                  "role_differentiation": False, "transition_from": kw["from"],
                  "transition_to": kw["to"]} for kw in case["key_windows"]]
        prompt = build_llm_prompt(preds, case["ctx"], {})
        return client.complete(prompt), case["ctx"]
    return run_case
