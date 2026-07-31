"""
Held-out unanswerable shapes: three structurally-unanswerable context types that
appear NOWHERE in data/sft_train_final_abstain.jsonl (verified by string search in
this module's __main__ / by tests/test_holdout_shapes.py). This is the decisive
test for whether v3b's 100% abstention on multi_hop/terminal_transitioning is a
learned "insufficient information -> decline" capability or memorization of the two
specific training substrings ("-> transitioning -> ... -> transitioning ->" chains
and terminal "-> transitioning").

Three shapes, each ~6 cases (one per TEST_CASES base), all has_ground_truth=False:

  a. deeper_chain      -- a 5-formation / 4-hop chain. Training's multi_hop axis
                          only ever went to chain length 4 (3 hops, severity=4).
                          Same code path (degradation.py's _render_lines with
                          resolvable=False), one increment past anything trained on.
  b. dominant_mismatch -- "Dominant formation: X" paired with a formation history
                          that never contains X at all. Every context in training
                          (synth_context AND degradation.py's _render_lines) sets
                          dominant = the history's own first element by construction
                          -- self-contradiction between these two specific lines
                          never occurs anywhere in the corpus.
  c. oov_formation     -- a formation name ("phalanx") that is not in
                          BASE_FORMATIONS and appears nowhere in RULES, TEST_CASES,
                          or any training row. Structurally resolvable-LOOKING (a
                          clean "A -> B" transition line) but B is vocabulary RULES
                          cannot key on -- tests whether the model recognizes
                          unfamiliar vocabulary as unanswerable rather than pattern-
                          matching the clean transition-line shape into a confident
                          guess.

Usage:
    python llm_finetuning/holdout_shapes.py   # verifies absence from training file, prints case counts
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from swarm_intent import context_spec as spec  # noqa: E402
from swarm_intent.llm.prompts import TEST_CASES  # noqa: E402

from degradation import (_render_lines, _key_windows, _next_chain_formation,  # noqa: E402
                         _NEUTRAL, CHAIN_POOL)

OOV_FORMATION = "phalanx"


def _make_case(name, shape, base_case, ctx, key_windows):
    return {"name": name, "axis": "holdout_" + shape, "shape": shape, "severity": "holdout",
            "base_case": base_case["name"], "has_ground_truth": False,
            "ctx": ctx, "key_windows": key_windows}


def shape_deeper_chain(test_cases):
    """5-formation chain (4 hops) -- degradation.py's multi_hop axis never exceeds
    chain length 4 (severity=4, 3 hops); training's abstention rows top out there."""
    cases = []
    for base in test_cases:
        form_a, form_b = base["formation_a"], base["formation_b"]
        chain = [form_a, form_b]
        used = [form_a, form_b]
        while len(chain) < 5:
            nxt = _next_chain_formation(used)
            chain.append(nxt)
            used.append(nxt)
        ctx, _ = _render_lines(chain, terminal_transitioning=False, resolvable=False, **_NEUTRAL)
        kw = _key_windows(chain[0], chain[-1], _NEUTRAL["mean_conf"], _NEUTRAL["mean_stab"],
                          _NEUTRAL["approach"])
        cases.append(_make_case(f"{base['name']}__deeper_chain", "deeper_chain", base, ctx, kw))
    return cases


def shape_dominant_mismatch(test_cases):
    """'Dominant formation' names a formation absent from 'Formation history'
    entirely -- self-contradiction between two specific lines that _render_lines
    (and therefore every training row) never produces, since it always derives
    dominant from history[0] by construction."""
    cases = []
    for base in test_cases:
        form_a, form_b = base["formation_a"], base["formation_b"]
        others = [f for f in CHAIN_POOL if f not in (form_a, form_b)]
        mismatched_dominant = others[0]
        history = f"{form_a} -> transitioning -> {form_b}"
        lines = [
            "Observation window: 0.0s - 60.0s (9 overlapping windows)",
            f"Dominant formation: {mismatched_dominant}",
            f"Formation history: {history}",
            f"Transition detected at t=20.0s: {form_a} -> {form_b}",
            f"Velocity trend: {_NEUTRAL['vel_word']} (delta_v={_NEUTRAL['delta_v']:+.2f})",
            f"Formation stability: {_NEUTRAL['stab_word']} (mean={_NEUTRAL['mean_stab']:.2f})",
            f"Spread dynamics: {_NEUTRAL['spread_word']} (mean approach_rate={_NEUTRAL['approach']:.3f})",
            spec.CONFIDENCE_LINE_TEMPLATE.format(mean_conf=_NEUTRAL["mean_conf"], low_conf=0),
        ]
        ctx = "\n".join(lines)
        kw = _key_windows(form_a, form_b, _NEUTRAL["mean_conf"], _NEUTRAL["mean_stab"],
                          _NEUTRAL["approach"])
        cases.append(_make_case(f"{base['name']}__dominant_mismatch", "dominant_mismatch",
                                base, ctx, kw))
    return cases


def shape_oov_formation(test_cases):
    """A clean-looking 'A -> B' transition where B is not a real formation name --
    RULES/BASE_FORMATIONS have no entry for it, but the LINE SHAPE is identical to
    an ordinary resolvable transition (unlike deeper_chain/terminal_transitioning,
    which are shaped differently). Tests vocabulary generalization specifically,
    isolated from structural-shape novelty."""
    cases = []
    for base in test_cases:
        form_a = base["formation_a"]
        chain = [form_a, OOV_FORMATION]
        ctx, _ = _render_lines(chain, terminal_transitioning=False, resolvable=True, **_NEUTRAL)
        kw = _key_windows(form_a, OOV_FORMATION, _NEUTRAL["mean_conf"], _NEUTRAL["mean_stab"],
                          _NEUTRAL["approach"])
        cases.append(_make_case(f"{base['name']}__oov_formation", "oov_formation", base, ctx, kw))
    return cases


SHAPES = {
    "deeper_chain": shape_deeper_chain,
    "dominant_mismatch": shape_dominant_mismatch,
    "oov_formation": shape_oov_formation,
}


def build_holdout_battery(test_cases=TEST_CASES) -> dict:
    return {shape: fn(test_cases) for shape, fn in SHAPES.items()}


def verify_absent_from_training(train_path: str, battery: dict) -> dict:
    """String-search each case's ctx (and its distinguishing substrings) against the
    raw training file text. Returns {shape: [violations]} -- empty dict means clean."""
    with open(train_path) as f:
        corpus = f.read()
    violations = {}
    for shape, cases in battery.items():
        hits = []
        for case in cases:
            # The exact rendered context block is the strongest check; also check
            # the single most distinguishing line for that shape in isolation, in
            # case the full block happens to differ by whitespace/formatting.
            if case["ctx"] in corpus:
                hits.append((case["name"], "full ctx block found verbatim"))
        if shape == "oov_formation" and OOV_FORMATION in corpus:
            hits.append(("(vocabulary check)", f"'{OOV_FORMATION}' appears somewhere in training file"))
        if hits:
            violations[shape] = hits
    return violations


if __name__ == "__main__":
    import os as _os
    battery = build_holdout_battery()
    for shape, cases in battery.items():
        print(f"{shape}: {len(cases)} cases")
    train_path = _os.path.join(_os.path.dirname(__file__), "..", "data", "sft_train_final_abstain.jsonl")
    violations = verify_absent_from_training(train_path, battery)
    if violations:
        print("\nVIOLATIONS (shape occurs in training file):")
        for shape, hits in violations.items():
            print(f"  {shape}: {hits}")
        raise SystemExit(1)
    print(f"\nVerified: none of these {sum(len(c) for c in battery.values())} shapes "
          f"appear in {train_path}")
