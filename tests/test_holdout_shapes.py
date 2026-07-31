"""
Tests for llm_finetuning/holdout_shapes.py -- the decisive generalization check for
whether v3b's abstention is a learned capability or memorization of the two training
substrings. No GPU/model required.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llm_finetuning"))

from swarm_intent.llm.prompts import ORIGINAL_TEST_CASES

from holdout_shapes import build_holdout_battery, verify_absent_from_training, OOV_FORMATION

REPO = os.path.join(os.path.dirname(__file__), "..")
TRAIN_PATH = os.path.join(REPO, "data", "sft_train_final_abstain.jsonl")


class TestHoldoutShapeStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.battery = build_holdout_battery(ORIGINAL_TEST_CASES)

    def test_three_shapes_six_cases_each(self):
        self.assertEqual(set(self.battery.keys()),
                         {"deeper_chain", "dominant_mismatch", "oov_formation"})
        for shape, cases in self.battery.items():
            self.assertEqual(len(cases), len(ORIGINAL_TEST_CASES), shape)

    def test_all_cases_are_unanswerable_by_design(self):
        for cases in self.battery.values():
            for c in cases:
                self.assertFalse(c["has_ground_truth"])

    def test_deeper_chain_has_four_hops(self):
        for c in self.battery["deeper_chain"]:
            self.assertEqual(c["ctx"].count(" -> transitioning -> "), 4)

    def test_dominant_mismatch_dominant_absent_from_history(self):
        for c in self.battery["dominant_mismatch"]:
            lines = c["ctx"].splitlines()
            dominant = lines[1].split(": ", 1)[1]
            history = lines[2].split(": ", 1)[1]
            self.assertNotIn(dominant, history.split(" -> transitioning -> "))

    def test_oov_formation_not_in_base_formations(self):
        from swarm_intent.config import BASE_FORMATIONS
        self.assertNotIn(OOV_FORMATION, BASE_FORMATIONS)
        for c in self.battery["oov_formation"]:
            self.assertIn(OOV_FORMATION, c["ctx"])


class TestAbsentFromTrainingData(unittest.TestCase):
    """The whole point of this battery: it must not overlap with what v3b trained
    on, or a positive result would be memorization, not generalization."""

    def test_no_shape_appears_verbatim_in_training_file(self):
        self.assertTrue(os.path.exists(TRAIN_PATH), f"missing {TRAIN_PATH}")
        battery = build_holdout_battery(ORIGINAL_TEST_CASES)
        violations = verify_absent_from_training(TRAIN_PATH, battery)
        self.assertEqual(violations, {}, f"held-out shapes leaked into training data: {violations}")


if __name__ == "__main__":
    unittest.main()
