"""
Phase 3a step 1: unit tests for src/swarm_intent/ground_truth_abstention.py -- the
STGT-independent ground-truth abstention-mechanism classifier the whole Phase 3a corpus
depends on. No model/checkpoint needed -- pure function over hand-built chain/true_labels.

Usage:
    python -m unittest tests.test_ground_truth_abstention -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from swarm_intent.ground_truth_abstention import (  # noqa: E402
    classify_trajectory_ground_truth, MULTI_HOP, OSCILLATION, TERMINAL_TRANSITIONING,
)
from swarm_intent.config import TRANSITION_CLASS  # noqa: E402


class TestAnswerableChains(unittest.TestCase):
    def test_single_formation_is_answerable(self):
        self.assertIsNone(classify_trajectory_ground_truth(["column"]))

    def test_two_formation_transition_is_answerable(self):
        self.assertIsNone(classify_trajectory_ground_truth(["column", "diamond"]))

    def test_two_formation_fully_dwelled_is_answerable(self):
        labels = ["column"] * 30 + [TRANSITION_CLASS] * 15 + ["diamond"] * 30
        self.assertIsNone(classify_trajectory_ground_truth(["column", "diamond"], labels))


class TestMultiHop(unittest.TestCase):
    def test_three_distinct_formations_is_multihop(self):
        self.assertEqual(classify_trajectory_ground_truth(["column", "diamond", "shield"]), MULTI_HOP)

    def test_four_distinct_formations_no_repeat_is_multihop(self):
        self.assertEqual(
            classify_trajectory_ground_truth(["column", "diamond", "shield", "v_shape"]), MULTI_HOP)


class TestOscillation(unittest.TestCase):
    def test_returns_to_first_formation_is_oscillation(self):
        self.assertEqual(classify_trajectory_ground_truth(["column", "diamond", "column"]), OSCILLATION)

    def test_four_hop_return_to_start_is_oscillation(self):
        self.assertEqual(
            classify_trajectory_ground_truth(["column", "diamond", "shield", "column"]), OSCILLATION)

    def test_middle_repeat_without_returning_to_first_is_still_multihop(self):
        # first != last -- oscillation is specifically "returns to where it started",
        # not "any repeat anywhere in the chain".
        self.assertEqual(
            classify_trajectory_ground_truth(["column", "diamond", "column", "shield"]), MULTI_HOP)


class TestTerminalTransitioning(unittest.TestCase):
    def test_truncated_mid_blend_is_terminal_transitioning(self):
        # observation cut off WHILE still in the blend window -- last label is the
        # TRANSITION_CLASS sentinel, not a settled destination formation.
        labels = ["column"] * 30 + [TRANSITION_CLASS] * 10
        self.assertEqual(
            classify_trajectory_ground_truth(["column", "diamond"], labels), TERMINAL_TRANSITIONING)

    def test_single_formation_truncated_mid_blend_also_terminal_transitioning(self):
        labels = ["column"] * 20 + [TRANSITION_CLASS] * 5
        self.assertEqual(
            classify_trajectory_ground_truth(["column"], labels), TERMINAL_TRANSITIONING)

    def test_chain_length_3_takes_priority_over_terminal_transitioning_check(self):
        # len(chain)>=3 is decided from chain alone -- true_labels is never consulted once
        # that branch is taken, per the function's own documented precedence.
        labels = ["column"] * 30 + [TRANSITION_CLASS] * 10  # would look "terminal" in isolation
        self.assertEqual(
            classify_trajectory_ground_truth(["column", "diamond", "shield"], labels), MULTI_HOP)


class TestNoTrueLabelsSupplied(unittest.TestCase):
    def test_terminal_transitioning_undetectable_without_true_labels_defaults_answerable(self):
        # true_labels is optional -- multi_hop/oscillation are still fully determined by
        # chain alone, but terminal_transitioning can never be detected without it.
        self.assertIsNone(classify_trajectory_ground_truth(["column", "diamond"]))


if __name__ == "__main__":
    unittest.main()
