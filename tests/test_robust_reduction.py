"""
AUDIT.md sec AG step 2: unit tests for stgt_bridge.py's robust reduction
(``robust=True``) and its interaction with coverage.classify_observation's
guard suppression. No checkpoint needed -- hand-built prediction dicts,
same convention as tests/test_stgt_bridge.py.

Usage:
    python -m unittest tests.test_robust_reduction -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from swarm_intent.stgt_bridge import bridge_predictions, UNKNOWN_FORMATION  # noqa: E402
from swarm_intent.coverage import classify_observation, BUCKET_A, BUCKET_B, BUCKET_C  # noqa: E402


def make_window(formation_type, t, confidence=0.9, class_probabilities=None):
    return {
        "formation_type": formation_type, "formation_confidence": confidence,
        "centroid_velocity": 5.0, "approach_rate": 0.0, "formation_stability": 0.8,
        "role_differentiation": False, "transition_from": None, "transition_to": None,
        "time_start_s": float(t), "time_end_s": float(t) + 24.5,
        "class_probabilities": class_probabilities or {},
    }


class TestRobustFalseIsUnchanged(unittest.TestCase):
    """robust=False (default) must reproduce robust=True-eligible-but-unrequested
    behaviour byte-for-byte -- i.e. omitting the flag changes nothing."""

    def test_trailing_transitioning_run_still_terminal_unknown_when_robust_false(self):
        preds = [make_window("column", t) for t in range(0, 80, 10)] + \
                [make_window("transitioning", t) for t in range(80, 100, 10)]
        r = classify_observation(preds)  # robust not passed -> False
        self.assertEqual(r["bucket"], BUCKET_C)
        self.assertEqual(r["subtype"], "terminal_unknown")
        self.assertIsNone(r["robust_recovery"])


class TestRobustReductionRecoversTrailingRun(unittest.TestCase):
    def test_trailing_transitioning_run_recovered_as_steady_state(self):
        preds = [make_window("column", t) for t in range(0, 80, 10)] + \
                [make_window("transitioning", t) for t in range(80, 100, 10)]
        r = classify_observation(preds, robust=True)
        self.assertEqual(r["bucket"], BUCKET_A)
        self.assertEqual(r["rules_key"], ("column", "column"))
        self.assertTrue(r["robust_recovery"]["recovered"])
        self.assertEqual(r["robust_recovery"]["stripped_trailing"], 2)

    def test_trailing_run_after_a_real_transition_is_recovered_as_that_transition(self):
        preds = ([make_window("column", t) for t in range(0, 40, 10)]
                + [make_window("diamond", t) for t in range(40, 80, 10)]
                + [make_window("transitioning", t) for t in range(80, 100, 10)])
        r = classify_observation(preds, robust=True)
        self.assertEqual(r["bucket"], BUCKET_A)
        self.assertEqual(r["rules_key"], ("column", "diamond"))


class TestRobustReductionAllUnknownFallback(unittest.TestCase):
    def test_well_separated_probabilities_resolve_to_steady_state(self):
        probs = {"column": 0.5, "diamond": 0.1, "v_shape": 0.06, "encirclement": 0.06,
                "dispersed": 0.3, "converging": 0.03, "shield": 0.05, "transitioning": -0.1}
        # dispersed/converging deliberately NOT near-tied (0.3 vs 0.03) so this test
        # isolates the all-unknown probability-fallback path from the (unconditional,
        # step 2d) ambiguity guard, which has its own dedicated test above.
        probs["transitioning"] = round(1 - sum(v for k, v in probs.items() if k != "transitioning"), 4)
        preds = [make_window("transitioning", t, class_probabilities=probs) for t in range(0, 60, 10)]
        r = classify_observation(preds, robust=True)
        self.assertEqual(r["bucket"], BUCKET_A)
        self.assertEqual(r["rules_key"], ("column", "column"))
        self.assertTrue(r["robust_recovery"]["low_confidence"])

    def test_poorly_separated_probabilities_stay_unresolved(self):
        probs = {"column": 0.20, "diamond": 0.19, "v_shape": 0.15, "encirclement": 0.12,
                "dispersed": 0.12, "converging": 0.12, "shield": 0.05, "transitioning": 0.05}
        preds = [make_window("transitioning", t, class_probabilities=probs) for t in range(0, 60, 10)]
        r = classify_observation(preds, robust=True)
        self.assertEqual(r["bucket"], BUCKET_C)
        self.assertEqual(r["subtype"], "all_unknown")


class TestRobustReductionRespectsThreshold(unittest.TestCase):
    def test_noisy_below_threshold_falls_back_to_original_logic(self):
        # first half: 2/4 column, 2/4 diamond -- no majority at threshold=0.7
        preds = [make_window("column", 0), make_window("diamond", 10),
                make_window("column", 20), make_window("diamond", 30),
                make_window("shield", 40), make_window("shield", 50)]
        r = classify_observation(preds, robust=True, robust_threshold=0.7)
        # falls back to unanimity logic -- 4 distinct groups (column,diamond,column,diamond,shield)
        # collapses via groupby to column,diamond,column,diamond,shield = 5 groups -> multi_hop
        self.assertEqual(r["bucket"], BUCKET_C)

    def test_clean_majority_above_threshold_recovers(self):
        preds = ([make_window("column", t) for t in range(0, 50, 10)]
                + [make_window("diamond", t) for t in range(50, 100, 10)])
        r = classify_observation(preds, robust=True, robust_threshold=0.7)
        self.assertEqual(r["bucket"], BUCKET_A)
        self.assertEqual(r["rules_key"], ("column", "diamond"))


class TestDispersedConvergingGuardStillUnconditional(unittest.TestCase):
    """Step 2d: the ambiguity guard must still fire on a robustly-recovered pair --
    robust reduction must never bypass it."""

    def test_ambiguous_window_still_guards_a_robustly_recovered_case(self):
        amb = {"dispersed": 0.48, "converging": 0.44}
        preds = ([make_window("column", t, class_probabilities=amb) for t in range(0, 50, 10)]
                + [make_window("transitioning", t) for t in range(50, 70, 10)])
        r = classify_observation(preds, robust=True)
        self.assertEqual(r["bucket"], BUCKET_B)
        self.assertIn("dispersed_converging_ambiguity", r["guard_reasons"])

    def test_oov_guard_suppressed_when_robust_recovers_but_ambiguity_guard_is_not(self):
        preds = [make_window("column", t) for t in range(0, 80, 10)] + \
                [make_window("transitioning", t) for t in range(80, 100, 10)]
        r = classify_observation(preds, robust=True)
        # trailing run recovered cleanly, no ambiguous windows -> bucket A, no oov_name guard
        self.assertEqual(r["bucket"], BUCKET_A)
        self.assertNotIn("oov_name", r["guard_reasons"])


class TestBridgePredictionsRobustFallback(unittest.TestCase):
    """robust=True but reduction fails -> falls back to the SAME unanimity output
    robust=False would give, not a broken/different result."""

    def test_robust_true_failure_matches_robust_false_output(self):
        preds = [make_window("column", 0), make_window("diamond", 10),
                make_window("shield", 20), make_window("v_shape", 30)]
        ctx_false, summary_false, kw_false = bridge_predictions(preds, robust=False)
        ctx_true, summary_true, kw_true = bridge_predictions(preds, robust=True, robust_threshold=0.99)
        self.assertEqual(summary_false["formation_history"], summary_true["formation_history"])
        self.assertEqual(summary_false["dominant_formation"], summary_true["dominant_formation"])
        self.assertIsNone(summary_true["robust_reduction"])


if __name__ == "__main__":
    unittest.main()
