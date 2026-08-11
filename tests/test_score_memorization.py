"""
Tests for llm_finetuning/score_memorization.py, using a small synthetic
training index (not the real 12,001-row corpus, so tests are fast and
self-contained) with hand-crafted eval outputs where the "right answer"
(memorized vs novel) is known by construction.

Usage:
    python -m unittest tests.test_score_memorization -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llm_finetuning"))

from score_memorization import overlap_rate, interpret, MEMORIZATION_SIGNAL_BAR  # noqa: E402


TRAIN_INDEX = {
    ("column", "diamond"): [
        "A group of UAVs transitioned from a column formation to a diamond shape at "
        "t=20s while maintaining steady velocity and stable spread.",
        "Multiple drones shifted from column to diamond formation, showing coordinated "
        "movement without acceleration.",
    ],
    ("shield", "shield"): [
        "The UAV group maintains a tight shield formation while slowing down, "
        "indicating a defensive posture.",
    ],
}


class TestOverlapRateKnownCases(unittest.TestCase):
    def test_exact_copy_of_training_target_scores_as_near_duplicate(self):
        """Feed it an output that IS (verbatim) a training target for that
        pair -- this must score max_similarity == 1.0 and count as a
        near-duplicate. This is the clearest possible memorization signal."""
        cases = [{
            "name": "memorized_case",
            "pair": ("column", "diamond"),
            "situation_summary": TRAIN_INDEX[("column", "diamond")][0],  # verbatim copy
        }]
        rate, details = overlap_rate(cases, TRAIN_INDEX)
        self.assertEqual(rate, 1.0)
        self.assertAlmostEqual(details[0]["max_similarity"], 1.0, places=3)
        self.assertTrue(details[0]["near_duplicate"])

    def test_clearly_novel_output_scores_low(self):
        """Feed it output that shares the topic/pair but is genuinely
        different prose (different structure, different details) -- this
        must score well below the near-duplicate threshold."""
        cases = [{
            "name": "novel_case",
            "pair": ("column", "diamond"),
            "situation_summary": "Sensor data indicates the swarm is regrouping into a "
                                 "more defensive posture after an extended patrol phase, "
                                 "with no signs of acceleration toward the perimeter.",
        }]
        rate, details = overlap_rate(cases, TRAIN_INDEX)
        self.assertEqual(rate, 0.0)
        self.assertLess(details[0]["max_similarity"], 0.90)
        self.assertFalse(details[0]["near_duplicate"])

    def test_mixed_batch_computes_correct_fraction(self):
        """2 memorized + 2 novel -> rate must be exactly 0.5, not some other
        aggregate (e.g. mean similarity, which would give a different, wrong
        number)."""
        cases = [
            {"name": "mem1", "pair": ("column", "diamond"),
             "situation_summary": TRAIN_INDEX[("column", "diamond")][0]},
            {"name": "mem2", "pair": ("shield", "shield"),
             "situation_summary": TRAIN_INDEX[("shield", "shield")][0]},
            {"name": "novel1", "pair": ("column", "diamond"),
             "situation_summary": "A completely different tactical narrative about "
                                  "surveillance patterns near the eastern perimeter."},
            {"name": "novel2", "pair": ("shield", "shield"),
             "situation_summary": "Weather conditions are degrading visibility for the "
                                  "operator monitoring this specific sector today."},
        ]
        rate, details = overlap_rate(cases, TRAIN_INDEX)
        self.assertEqual(rate, 0.5)
        near_dup_names = {d["name"] for d in details if d["near_duplicate"]}
        self.assertEqual(near_dup_names, {"mem1", "mem2"})

    def test_paraphrase_level_near_duplicate_still_caught(self):
        """A one-word-swapped paraphrase of a training target (the exact
        pattern found for real in report_distinctness_similarity.py) must
        still be caught -- this is what makes the check useful against
        actual model behavior, not just literal copy-paste."""
        cases = [{
            "name": "paraphrase_case",
            "pair": ("shield", "shield"),
            "situation_summary": "The UAV swarm maintains a tight shield formation while "
                                 "slowing down, indicating a defensive posture.",  # only "group"->"swarm" changed
        }]
        rate, details = overlap_rate(cases, TRAIN_INDEX)
        self.assertGreaterEqual(details[0]["max_similarity"], 0.70)

    def test_empty_batch_returns_zero_not_error(self):
        rate, details = overlap_rate([], TRAIN_INDEX)
        self.assertEqual(rate, 0.0)
        self.assertEqual(details, [])

    def test_pair_with_no_training_rows_scores_zero_not_crash(self):
        cases = [{"name": "orphan", "pair": ("v_shape", "v_shape"),
                  "situation_summary": "anything at all"}]
        rate, details = overlap_rate(cases, TRAIN_INDEX)
        self.assertEqual(rate, 0.0)
        self.assertEqual(details[0]["max_similarity"], 0.0)


class TestInterpret(unittest.TestCase):
    def test_low_rate_reads_as_generalization(self):
        self.assertIn("GENERALIZATION", interpret(0.01))

    def test_high_rate_reads_as_memorization_signal(self):
        self.assertIn("MEMORIZATION SIGNAL", interpret(MEMORIZATION_SIGNAL_BAR))
        self.assertIn("MEMORIZATION SIGNAL", interpret(0.5))

    def test_middle_rate_reads_as_ambiguous(self):
        self.assertIn("AMBIGUOUS", interpret(0.08))


if __name__ == "__main__":
    unittest.main()
