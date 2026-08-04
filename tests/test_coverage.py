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
    def test_oov_blip_amid_clean_transition_is_guardable(self):
        preds = [make_window("column", 0), make_window("transitioning", 10),
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

    def test_tied_dominant_count_is_guardable(self):
        preds = [make_window("column", 0), make_window("column", 10),
                make_window("diamond", 20), make_window("diamond", 30)]
        r = classify_observation(preds)
        self.assertEqual(r["bucket"], BUCKET_B)
        self.assertIn("dominant_history_contradiction", r["guard_reasons"])

    def test_one_window_confident_is_not_guardable_on_confidence(self):
        preds = [make_window("shield", 0, confidence=0.5), make_window("shield", 10, confidence=0.95)]
        r = classify_observation(preds)
        self.assertNotIn("low_confidence", r["guard_reasons"])


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
