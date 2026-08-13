"""
AUDIT.md sec AE: unit tests for src/swarm_intent/coverage.py's bucket
classification (A resolvable / B guardable / C unresolvable), the routing
logic pipeline_v2.py's Layer 1/2/3 split depends on entirely. Hand-built
prediction dicts / ctx strings, no checkpoint needed.

Usage:
    python -m unittest tests.test_coverage -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from swarm_intent.coverage import (  # noqa: E402
    classify_observation, classify_ctx, BUCKET_A, BUCKET_B, BUCKET_C,
)


def make_window(formation_type, t, confidence=0.9, class_probabilities=None):
    return {
        "formation_type": formation_type, "formation_confidence": confidence,
        "centroid_velocity": 5.0, "approach_rate": 0.0, "formation_stability": 0.8,
        "role_differentiation": False, "transition_from": None, "transition_to": None,
        "time_start_s": float(t), "time_end_s": float(t) + 24.5,
        "class_probabilities": class_probabilities or {},
    }


class TestBucketA(unittest.TestCase):
    def test_steady_state_is_resolvable(self):
        preds = [make_window("column", 0), make_window("column", 10), make_window("column", 20)]
        r = classify_observation(preds)
        self.assertEqual(r["bucket"], BUCKET_A)
        self.assertEqual(r["rules_key"], ("column", "column"))
        self.assertEqual(r["guard_reasons"], [])

    def test_single_clean_transition_is_resolvable(self):
        preds = [make_window("v_shape", 0), make_window("v_shape", 10), make_window("v_shape", 15),
                make_window("encirclement", 20), make_window("encirclement", 30)]
        r = classify_observation(preds)
        self.assertEqual(r["bucket"], BUCKET_A)
        self.assertEqual(r["rules_key"], ("v_shape", "encirclement"))


class TestBucketBGuards(unittest.TestCase):
    def test_transitioning_blip_amid_clean_transition_is_now_resolvable(self):
        """2026-08-09 fix: a "transitioning" read is the classifier's own valid
        class, not a genuinely out-of-vocabulary name -- this must no longer
        trigger oov_name (it did before the fix; that was the bug)."""
        preds = [make_window("column", 0), make_window("transitioning", 10),
                make_window("diamond", 20), make_window("diamond", 30)]
        r = classify_observation(preds)
        self.assertEqual(r["bucket"], BUCKET_A)
        self.assertNotIn("oov_name", r["guard_reasons"])
        self.assertEqual(r["rules_key"], ("column", "diamond"))

    def test_genuinely_oov_name_amid_clean_transition_is_still_guardable(self):
        """The guard's actual claimed purpose -- a real out-of-vocabulary
        formation name (not the classifier's own "transitioning" class) --
        must still fire."""
        preds = [make_window("column", 0), make_window("phalanx", 10),
                make_window("diamond", 20), make_window("diamond", 30)]
        r = classify_observation(preds)
        self.assertEqual(r["bucket"], BUCKET_B)
        self.assertIn("oov_name", r["guard_reasons"])
        self.assertEqual(r["rules_key"], ("column", "diamond"))

    def test_dispersed_converging_near_tie_is_guardable(self):
        probs = {"dispersed": 0.48, "converging": 0.44}
        preds = [make_window("dispersed", 0, class_probabilities=probs),
                make_window("dispersed", 10, class_probabilities=probs)]
        r = classify_observation(preds)
        self.assertEqual(r["bucket"], BUCKET_B)
        self.assertIn("dispersed_converging_ambiguity", r["guard_reasons"])

    def test_all_windows_low_confidence_is_guardable(self):
        preds = [make_window("shield", 0, confidence=0.5), make_window("shield", 10, confidence=0.55)]
        r = classify_observation(preds)
        self.assertEqual(r["bucket"], BUCKET_B)
        self.assertIn("low_confidence", r["guard_reasons"])

    def test_tied_window_count_is_no_longer_guardable(self):
        """2026-08-09 fix: `key` (the derived (from,to) pair) is determined by
        TEMPORAL ORDER, never by window count -- a raw 2/2 count tie says
        nothing about whether `key` is trustworthy (a clean, obviously-correct
        split ties exactly as easily as an uncertain one). This exact shape
        (100% spurious when it was the sole blocker) must no longer guard.
        Note: a GENUINE dominant/key contradiction is provably unreachable via
        classify_observation's own code path post-fix (dominant_formation is
        always the mode over the same valid-formations set known_history is
        built from, so it can never differ from both members of key) -- there
        is no hand-buildable input that fires this guard anymore, which is
        the correct, audited outcome, not a gap in this test."""
        preds = [make_window("column", 0), make_window("column", 10),
                make_window("diamond", 20), make_window("diamond", 30)]
        r = classify_observation(preds)
        self.assertEqual(r["bucket"], BUCKET_A)
        self.assertNotIn("dominant_history_contradiction", r["guard_reasons"])

    def test_one_window_confident_is_not_guardable_on_confidence(self):
        preds = [make_window("shield", 0, confidence=0.5), make_window("shield", 10, confidence=0.95)]
        r = classify_observation(preds)
        self.assertNotIn("low_confidence", r["guard_reasons"])


class TestMaxWindowsGuard(unittest.TestCase):
    """AUDIT.md sec AK/AL: catches the 9 real bucket_A_misrouted cases -- a
    genuine 3+-formation sequence whose classifier read collapsed to a
    confident 2-formation pair, with no per-window low-confidence/ambiguity
    signal at all (an observation simply too long for the state count it
    reduced to)."""

    def test_steady_state_within_bound_is_still_resolvable(self):
        preds = [make_window("column", t) for t in range(0, 90, 10)]  # 9 windows, at the bound
        self.assertEqual(len(preds), 9)
        r = classify_observation(preds)
        self.assertEqual(r["bucket"], BUCKET_A)
        self.assertNotIn("observation_too_long_for_reduced_state_count", r["guard_reasons"])

    def test_clean_transition_within_bound_is_still_resolvable(self):
        preds = ([make_window("v_shape", t) for t in range(0, 40, 10)]
                + [make_window("encirclement", t) for t in range(40, 90, 10)])
        self.assertEqual(len(preds), 9)
        r = classify_observation(preds)
        self.assertEqual(r["bucket"], BUCKET_A)
        self.assertNotIn("observation_too_long_for_reduced_state_count", r["guard_reasons"])

    def test_two_formations_but_too_many_windows_is_guardable(self):
        """The exact sec AK shape: only 2 distinct formations ever appear in the
        classifier's own read, every window confident, but the observation ran
        longer than any genuine <=2-formation sequence could -- a real
        intermediate state (e.g. seq 105's dropped "encirclement") was very
        likely present and silently lost."""
        preds = ([make_window("shield", t) for t in range(0, 30, 10)]
                + [make_window("v_shape", t) for t in range(30, 170, 10)])
        self.assertGreater(len(preds), 9)
        r = classify_observation(preds)
        self.assertEqual(r["bucket"], BUCKET_B)
        self.assertIn("observation_too_long_for_reduced_state_count", r["guard_reasons"])
        # the (from, to) pair is still reported (for the abstention message / any
        # downstream diagnostic) -- only the BUCKET changes, not the derived key.
        self.assertEqual(r["rules_key"], ("shield", "v_shape"))

    def test_fires_regardless_of_robust_recovery(self):
        """Unconditional, unlike oov_name/dominant_history_contradiction --
        robust recovery answers a different question (is noisy per-window
        disagreement trustworthy), not this one (did the observation run long
        enough that a whole state was plausibly missed)."""
        preds = ([make_window("shield", t) for t in range(0, 30, 10)]
                + [make_window("v_shape", t) for t in range(30, 170, 10)])
        r = classify_observation(preds, robust=True)
        self.assertIn("observation_too_long_for_reduced_state_count", r["guard_reasons"])


class TestMaxWindowsConstantNotDrifted(unittest.TestCase):
    """coverage.py deliberately does NOT import from data.py (keeps this
    module's import surface free of data.py's sklearn dependency) -- so
    MAX_WINDOWS_PER_SINGLE_TRANSITION is a hand-derived constant, not a live
    computation. This test is the tripwire: if data.py's ranges ever change,
    this fails loudly instead of silently invalidating the guard."""

    def test_constant_matches_current_data_py_ranges(self):
        from swarm_intent.data import LEAD_IN_RANGE, BLEND_DURATION_RANGE, MIN_DWELL_RANGE
        from swarm_intent.coverage import MAX_WINDOWS_PER_SINGLE_TRANSITION

        max_transition_timesteps = (LEAD_IN_RANGE[1] - 1) + (BLEND_DURATION_RANGE[1] - 1) + (MIN_DWELL_RANGE[1] - 1)
        max_windows = (max_transition_timesteps - 50) // 10 + 1
        self.assertEqual(MAX_WINDOWS_PER_SINGLE_TRANSITION, max_windows,
                         "data.py's LEAD_IN_RANGE/BLEND_DURATION_RANGE/MIN_DWELL_RANGE changed -- "
                         "MAX_WINDOWS_PER_SINGLE_TRANSITION in coverage.py must be updated to match")


class TestBucketCUnresolvable(unittest.TestCase):
    def test_all_unknown_is_unresolvable(self):
        preds = [make_window("transitioning", 0), make_window("transitioning", 10)]
        r = classify_observation(preds)
        self.assertEqual(r["bucket"], BUCKET_C)
        self.assertEqual(r["subtype"], "all_unknown")

    def test_terminal_unknown_is_unresolvable(self):
        preds = [make_window("column", 0), make_window("diamond", 10), make_window("transitioning", 20)]
        r = classify_observation(preds)
        self.assertEqual(r["bucket"], BUCKET_C)
        self.assertEqual(r["subtype"], "terminal_unknown")

    def test_monotonic_three_hop_chain_is_multi_hop(self):
        preds = [make_window("column", 0), make_window("diamond", 10), make_window("shield", 20)]
        r = classify_observation(preds)
        self.assertEqual(r["bucket"], BUCKET_C)
        self.assertEqual(r["subtype"], "multi_hop")

    def test_return_to_start_is_oscillation(self):
        preds = [make_window("dispersed", 0), make_window("encirclement", 10), make_window("dispersed", 20)]
        r = classify_observation(preds)
        self.assertEqual(r["bucket"], BUCKET_C)
        self.assertEqual(r["subtype"], "oscillation")


class TestClassifyCtx(unittest.TestCase):
    def test_steady_state_ctx_is_resolvable(self):
        ctx = ("Dominant formation: column\n"
              "Formation history: column\n"
              "No formation transitions detected.")
        r = classify_ctx(ctx, [{"confidence": 0.9}])
        self.assertEqual(r["bucket"], BUCKET_A)
        self.assertEqual(r["rules_key"], ("column", "column"))

    def test_clean_transition_ctx_is_resolvable(self):
        ctx = ("Dominant formation: v_shape\n"
              "Formation history: v_shape -> transitioning -> encirclement\n"
              "Transition detected at t=20.0s: v_shape -> encirclement")
        r = classify_ctx(ctx, [{"confidence": 0.9}, {"confidence": 0.9}])
        self.assertEqual(r["bucket"], BUCKET_A)
        self.assertEqual(r["rules_key"], ("v_shape", "encirclement"))

    def test_multi_hop_phrasing_has_no_extractable_pair(self):
        ctx = ("Dominant formation: column\n"
              "Formation history: column -> transitioning -> diamond -> transitioning -> shield\n"
              "Multiple formation changes detected across the observation window (2 transitions; see Formation history).")
        r = classify_ctx(ctx, [{"confidence": 0.9}])
        self.assertEqual(r["bucket"], BUCKET_C)
        self.assertEqual(r["subtype"], "no_extractable_pair")

    def test_oov_formation_name_is_guardable(self):
        ctx = ("Dominant formation: column\n"
              "Formation history: column -> transitioning -> phalanx\n"
              "Transition detected at t=20.0s: column -> phalanx")
        r = classify_ctx(ctx, [{"confidence": 0.9}])
        self.assertEqual(r["bucket"], BUCKET_B)
        self.assertIn("oov_name", r["guard_reasons"])

    def test_dominant_mismatch_is_guardable(self):
        ctx = ("Dominant formation: shield\n"
              "Formation history: v_shape -> transitioning -> encirclement\n"
              "Transition detected at t=20.0s: v_shape -> encirclement")
        r = classify_ctx(ctx, [{"confidence": 0.9}])
        self.assertEqual(r["bucket"], BUCKET_B)
        self.assertIn("dominant_history_contradiction", r["guard_reasons"])

    def test_all_key_windows_low_confidence_is_guardable(self):
        ctx = ("Dominant formation: column\n"
              "Formation history: column\n"
              "No formation transitions detected.")
        r = classify_ctx(ctx, [{"confidence": 0.4}, {"confidence": 0.5}])
        self.assertEqual(r["bucket"], BUCKET_B)
        self.assertIn("low_confidence", r["guard_reasons"])


if __name__ == "__main__":
    unittest.main()
