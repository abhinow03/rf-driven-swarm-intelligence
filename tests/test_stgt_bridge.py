"""
Step 4 of the "vendor teammate's retrained STGT" session (AUDIT.md sec V):
tests for src/swarm_intent/stgt_bridge.py, one per requirement (a-f) it exists
to satisfy. Uses hand-built prediction dicts shaped like
swarm_intent.stgt.inference.sliding_window_inference's output -- no checkpoint
needed, these are pure unit tests of the bridge logic.

Usage:
    python -m unittest tests.test_stgt_bridge -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from swarm_intent.stgt_bridge import (  # noqa: E402
    bridge_predictions, UNKNOWN_FORMATION, DISPERSED_CONVERGING_AMBIGUITY_MARGIN,
)


def make_window(formation_type, t, velocity=5.0, approach=0.0, stability=0.8,
                confidence=0.9, transition_from="WRONG_FROM", transition_to="WRONG_TO",
                class_probabilities=None):
    """transition_from/transition_to default to obviously-wrong sentinel strings --
    if the bridge ever reads them, tests relying on the correct temporal answer
    would fail loudly instead of accidentally passing."""
    return {
        "formation_type": formation_type,
        "formation_confidence": confidence,
        "centroid_velocity": velocity,
        "approach_rate": approach,
        "formation_stability": stability,
        "role_differentiation": False,
        "transition_from": transition_from,
        "transition_to": transition_to,
        "time_start_s": float(t),
        "time_end_s": float(t) + 24.5,
        "class_probabilities": class_probabilities or {},
    }


class TestTemporalTransitionDerivation(unittest.TestCase):
    """(a) transitions must come from consecutive formation_type values, never
    from predict_v2's transition_from/transition_to."""

    def test_transitions_match_temporal_sequence_not_injected_garbage(self):
        preds = [
            make_window("encirclement", 0),
            make_window("encirclement", 10),
            make_window("column", 20),
            make_window("column", 30),
            make_window("diamond", 40),
        ]
        _, summary, _ = bridge_predictions(preds)
        pairs = [(t["from"], t["to"]) for t in summary["transitions_detected"]]
        self.assertEqual(pairs, [("encirclement", "column"), ("column", "diamond")])

    def test_ignores_transition_from_to_fields_even_when_present_and_plausible(self):
        # transition_from/to on every window claim "shield -> v_shape" -- if the
        # bridge ever reads them, this would leak into transitions_detected.
        preds = [
            make_window("dispersed", 0, transition_from="shield", transition_to="v_shape"),
            make_window("dispersed", 10, transition_from="shield", transition_to="v_shape"),
            make_window("converging", 20, transition_from="shield", transition_to="v_shape"),
        ]
        _, summary, _ = bridge_predictions(preds)
        pairs = [(t["from"], t["to"]) for t in summary["transitions_detected"]]
        self.assertEqual(pairs, [("dispersed", "converging")])
        self.assertNotIn("shield", str(summary["transitions_detected"]))
        self.assertNotIn("v_shape", str(summary["transitions_detected"]))


class TestOscillationPreserved(unittest.TestCase):
    """(b) formation_history must not use dict.fromkeys -- oscillation (a repeat
    that isn't consecutive) must survive."""

    def test_non_consecutive_repeat_is_preserved(self):
        preds = [
            make_window("dispersed", 0), make_window("dispersed", 10),
            make_window("encirclement", 20),
            make_window("dispersed", 30), make_window("dispersed", 40),
        ]
        _, summary, _ = bridge_predictions(preds)
        # dict.fromkeys(["dispersed","dispersed","encirclement","dispersed","dispersed"])
        # would give ["dispersed", "encirclement"] -- losing the return trip.
        self.assertEqual(summary["formation_history"], ["dispersed", "encirclement", "dispersed"])

    def test_consecutive_repeats_still_collapse(self):
        preds = [make_window("column", t) for t in (0, 10, 20)]
        _, summary, _ = bridge_predictions(preds)
        self.assertEqual(summary["formation_history"], ["column"])


class TestAbstentionOnUnknownFormation(unittest.TestCase):
    """(c) any formation_type outside BASE_FORMATIONS (including the model's own
    "transitioning" class) must abstain, not silently pass through."""

    def test_transitioning_class_excluded_from_transitions_and_dominant(self):
        preds = [
            make_window("encirclement", 0), make_window("encirclement", 10),
            make_window("transitioning", 20),
            make_window("column", 30), make_window("column", 40), make_window("column", 50),
        ]
        _, summary, key_windows = bridge_predictions(preds)
        self.assertEqual(summary["dominant_formation"], "column")
        self.assertEqual(summary["n_unknown_windows"], 1)
        # 2026-08-09 fix: "transitioning" is the classifier's own VALID class, not a
        # genuinely out-of-vocabulary name -- n_genuinely_oov_windows must be 0 here
        # even though n_unknown_windows (the broader, narrative-only count) is 1.
        self.assertEqual(summary["n_genuinely_oov_windows"], 0)
        self.assertIn(UNKNOWN_FORMATION, summary["formation_history"])
        # no transition pair should mention "transitioning" on either side
        for t in summary["transitions_detected"]:
            self.assertNotEqual(t["from"], "transitioning")
            self.assertNotEqual(t["to"], "transitioning")
        # the unknown window is surfaced explicitly in key_windows, not dropped silently
        unknown_kw = [kw for kw in key_windows if kw["formation"] == UNKNOWN_FORMATION]
        self.assertEqual(len(unknown_kw), 1)

    def test_garbage_formation_name_also_treated_as_unknown(self):
        preds = [make_window("v_shape", 0), make_window("not_a_real_formation", 10)]
        _, summary, _ = bridge_predictions(preds)
        self.assertEqual(summary["n_unknown_windows"], 1)
        # a genuinely unrecognized name (unlike "transitioning") IS a real
        # data-integrity concern -- n_genuinely_oov_windows must count it.
        self.assertEqual(summary["n_genuinely_oov_windows"], 1)

    def test_all_unknown_triggers_full_abstention(self):
        preds = [make_window("transitioning", 0), make_window("transitioning", 10)]
        context, summary, key_windows = bridge_predictions(preds)
        self.assertTrue(summary["abstain"])
        self.assertIn("abstain_reason", summary)
        self.assertEqual(key_windows, [])

    def test_no_predictions_abstains(self):
        context, summary, key_windows = bridge_predictions([])
        self.assertTrue(summary["abstain"])


class TestKeyWindowsCapped(unittest.TestCase):
    """(d) key_windows must be capped at a fixed maximum, preferentially keeping
    first/last even when many other windows are also salient."""

    def test_capped_at_max_key_windows(self):
        # all low-confidence -> every window is individually salient
        preds = [make_window("column", t * 10, confidence=0.3) for t in range(30)]
        _, _, key_windows = bridge_predictions(preds, max_key_windows=5)
        self.assertLessEqual(len(key_windows), 5)

    def test_first_and_last_survive_the_cap(self):
        preds = [make_window("column", t * 10, confidence=0.3) for t in range(30)]
        _, _, key_windows = bridge_predictions(preds, max_key_windows=5)
        times = [kw["t"] for kw in key_windows]
        self.assertEqual(times[0], f"{preds[0]['time_start_s']}-{preds[0]['time_end_s']}s")
        self.assertEqual(times[-1], f"{preds[-1]['time_start_s']}-{preds[-1]['time_end_s']}s")

    def test_key_windows_returned_in_chronological_order(self):
        preds = [make_window("column", t * 10, confidence=0.3) for t in range(30)]
        _, _, key_windows = bridge_predictions(preds, max_key_windows=8)
        starts = [float(kw["t"].split("-")[0]) for kw in key_windows]
        self.assertEqual(starts, sorted(starts))


class TestDispersedConvergingAmbiguity(unittest.TestCase):
    """(e) near-equal dispersed/converging class probability must be flagged."""

    def test_close_probabilities_flagged_ambiguous(self):
        probs = {"dispersed": 0.42, "converging": 0.40, "column": 0.18}
        preds = [make_window("dispersed", 0, class_probabilities=probs)]
        self.assertLess(abs(probs["dispersed"] - probs["converging"]),
                        DISPERSED_CONVERGING_AMBIGUITY_MARGIN)
        _, summary, key_windows = bridge_predictions(preds)
        self.assertEqual(summary["n_ambiguous_dispersed_converging_windows"], 1)
        self.assertTrue(key_windows[0]["ambiguous_dispersed_converging"])

    def test_distinct_probabilities_not_flagged(self):
        probs = {"dispersed": 0.85, "converging": 0.05, "column": 0.10}
        preds = [make_window("dispersed", 0, class_probabilities=probs)]
        _, summary, key_windows = bridge_predictions(preds)
        self.assertEqual(summary["n_ambiguous_dispersed_converging_windows"], 0)
        self.assertFalse(key_windows[0]["ambiguous_dispersed_converging"])

    def test_missing_class_probabilities_not_flagged(self):
        preds = [make_window("dispersed", 0, class_probabilities={})]
        _, summary, _ = bridge_predictions(preds)
        self.assertEqual(summary["n_ambiguous_dispersed_converging_windows"], 0)

    def test_close_but_irrelevant_residual_mass_not_flagged(self):
        """2026-08-09 fix (docs/CEILING.md 2026-08-09 step 2): a window confidently
        predicted as some OTHER class, where dispersed/converging are both tiny
        residual probabilities that happen to be close to each other in absolute
        terms, must NOT be flagged -- this was the exact spurious-firing mechanism
        (V5 Phase 0 step 1: 66.1% of firings had neither class in the window's
        top-2). Real example from scripts/phase0_guard_audit.py: shield predicted
        at 98.97%, dispersed=0.0012, converging=0.0005."""
        probs = {"shield": 0.9897, "dispersed": 0.0012, "converging": 0.0005,
                 "v_shape": 0.003, "column": 0.002, "diamond": 0.001,
                 "encirclement": 0.0005, "transitioning": 0.0011}
        self.assertLess(abs(probs["dispersed"] - probs["converging"]),
                        DISPERSED_CONVERGING_AMBIGUITY_MARGIN)  # close in absolute terms
        preds = [make_window("shield", 0, class_probabilities=probs)]
        _, summary, key_windows = bridge_predictions(preds)
        self.assertEqual(summary["n_ambiguous_dispersed_converging_windows"], 0)
        self.assertFalse(key_windows[0]["ambiguous_dispersed_converging"])

    def test_top2_but_not_close_not_flagged(self):
        """The two conditions are independent: top-2 alone isn't enough without
        also being close in probability."""
        probs = {"dispersed": 0.55, "converging": 0.20, "column": 0.15, "shield": 0.10}
        preds = [make_window("dispersed", 0, class_probabilities=probs)]
        _, summary, _ = bridge_predictions(preds)
        self.assertEqual(summary["n_ambiguous_dispersed_converging_windows"], 0)


class TestNoDisallowedCharacters(unittest.TestCase):
    """(f) no character absent from training prompts (no "warning" glyph)."""

    def test_no_warning_glyph_even_with_low_confidence_and_ambiguous_windows(self):
        probs = {"dispersed": 0.40, "converging": 0.39}
        preds = [make_window("dispersed", t * 10, confidence=0.1, class_probabilities=probs)
                for t in range(5)]
        context, summary, key_windows = bridge_predictions(preds)
        self.assertNotIn("⚠", context)  # U+26A0 WARNING SIGN
        for kw in key_windows:
            self.assertNotIn("⚠", str(kw))

    def test_no_warning_glyph_on_abstention_path(self):
        preds = [make_window("transitioning", 0)]
        context, summary, _ = bridge_predictions(preds)
        self.assertNotIn("⚠", context)


if __name__ == "__main__":
    unittest.main()
