"""
Direct unit tests for llm_finetuning/literal_pair_extraction.py, covering the two text
hazards found during hand-validation (llm_finetuning/validate_literal_pair_extraction.py):
non-chronological prose and the converging/dispersed spread-dynamics vocabulary collision.

Usage:
    python -m unittest tests.test_literal_pair_extraction -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llm_finetuning"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from literal_pair_extraction import extract_literal_pair, true_pair_from_chain  # noqa: E402


class TestSteadyState(unittest.TestCase):
    def test_single_formation_mentioned_once_reads_as_steady_state(self):
        parsed = {"situation_summary": "The group maintained a shield formation throughout.",
                 "key_indicators": [], "threat_reasoning": ""}
        self.assertEqual(extract_literal_pair(parsed), ("shield", "shield"))


class TestSimpleTransition(unittest.TestCase):
    def test_two_distinct_mentions_in_order(self):
        parsed = {"situation_summary": "Started in a converging formation but quickly "
                                       "dispersed into a wide spread.",
                 "key_indicators": [], "threat_reasoning": ""}
        self.assertEqual(extract_literal_pair(parsed), ("converging", "dispersed"))


class TestNonChronologicalProseHazard(unittest.TestCase):
    """The dominant/final state is often narrated BEFORE the transition history."""

    def test_from_a_to_b_phrase_overrides_raw_mention_order(self):
        parsed = {
            "situation_summary": "The group maintained a diamond formation throughout, "
                                 "showing high stability.",
            "key_indicators": ["Transition from column to diamond at t=10s indicating "
                               "coordinated patrol"],
            "threat_reasoning": "",
        }
        self.assertEqual(extract_literal_pair(parsed), ("column", "diamond"))

    def test_multi_hop_chain_takes_first_and_last_not_first_two(self):
        parsed = {
            "situation_summary": "The UAV group maintained a diamond formation for the "
                                 "first 15 seconds before transitioning to a shield "
                                 "formation, then to a V-shape at 25 seconds.",
            "key_indicators": [], "threat_reasoning": "",
        }
        # (diamond, shield) would undersell it -- the chain's own endpoints are diamond->v_shape.
        self.assertEqual(extract_literal_pair(parsed), ("diamond", "v_shape"))


class TestSpreadDynamicsVocabularyCollision(unittest.TestCase):
    """converging/dispersed name both a formation type AND a spread-dynamics trend
    descriptor elsewhere in this project's tactical-context template."""

    def test_dispersing_spread_descriptor_not_counted_as_a_formation_mention(self):
        parsed = {
            "situation_summary": "Multiple UAVs are observed in a converging formation, "
                                 "moving steadily toward the protected area while gradually "
                                 "spreading out.",
            "key_indicators": ["Dispersing spread (mean approach_rate 0.129) suggesting "
                               "preparation for encirclement"],
            "threat_reasoning": "",
        }
        # Only "converging" is formation-qualified; "Dispersing spread" must be excluded.
        self.assertEqual(extract_literal_pair(parsed), ("converging", "converging"))


class TestExtractionFailure(unittest.TestCase):
    def test_no_formation_name_anywhere_returns_none(self):
        parsed = {"situation_summary": "No unusual activity detected in this sector.",
                 "key_indicators": [], "threat_reasoning": ""}
        self.assertIsNone(extract_literal_pair(parsed))


class TestTruePairFromChain(unittest.TestCase):
    def test_length_one_chain_is_steady_state(self):
        self.assertEqual(true_pair_from_chain(["diamond"]), ("diamond", "diamond"))

    def test_length_two_chain_is_the_pair_in_order(self):
        self.assertEqual(true_pair_from_chain(["diamond", "shield"]), ("diamond", "shield"))

    def test_length_three_chain_raises(self):
        with self.assertRaises(AssertionError):
            true_pair_from_chain(["diamond", "shield", "column"])


if __name__ == "__main__":
    unittest.main()
