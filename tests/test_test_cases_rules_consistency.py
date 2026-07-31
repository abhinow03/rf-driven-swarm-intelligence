"""
Standing guard: every TEST_CASES entry's expected_intent/expected_threat/
expected_action must match RULES (build_sft_dataset.py) exactly for its
(formation_a, formation_b) pair. Catches a future manual edit introducing a
mismatch instead of silently trusting whichever value was typed in prompts.py.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llm_finetuning"))

from swarm_intent.config import BASE_FORMATIONS
from swarm_intent.llm.prompts import TEST_CASES, ORIGINAL_TEST_CASES, RULES_COVERAGE_CASES

from build_sft_dataset import RULES


class TestTestCasesMatchRules(unittest.TestCase):
    def test_every_case_matches_rules_exactly(self):
        mismatches = []
        for c in TEST_CASES:
            key = (c["formation_a"], c["formation_b"])
            self.assertIn(key, RULES, f"{c['name']}: {key} is not a RULES key at all")
            threat, intent, action = RULES[key]
            if (c["expected_threat"], c["expected_intent"], c["expected_action"]) != (threat, intent, action):
                mismatches.append((c["name"], key,
                                   (c["expected_threat"], c["expected_intent"], c["expected_action"]),
                                   (threat, intent, action)))
        self.assertEqual(mismatches, [], f"TEST_CASES contradict RULES: {mismatches}")

    def test_original_six_unchanged_count(self):
        self.assertEqual(len(ORIGINAL_TEST_CASES), 6)

    def test_at_least_fifty_cases_total(self):
        self.assertGreaterEqual(len(TEST_CASES), 50)

    def test_full_rules_pair_coverage(self):
        rules_pairs = set(RULES.keys())
        covered_pairs = {(c["formation_a"], c["formation_b"]) for c in TEST_CASES}
        self.assertEqual(covered_pairs, rules_pairs,
                         f"missing: {rules_pairs - covered_pairs}, extra/unknown: {covered_pairs - rules_pairs}")

    def test_rules_coverage_cases_are_one_per_rules_key_no_duplicates(self):
        keys = [(c["formation_a"], c["formation_b"]) for c in RULES_COVERAGE_CASES]
        self.assertEqual(len(keys), len(RULES))
        self.assertEqual(len(set(keys)), len(keys), "RULES_COVERAGE_CASES has a duplicate pair")

    def test_includes_all_steady_state_pairs(self):
        steady_state_pairs = {(f, f) for f in BASE_FORMATIONS}
        covered = {(c["formation_a"], c["formation_b"]) for c in TEST_CASES}
        self.assertTrue(steady_state_pairs.issubset(covered))

    def test_includes_all_low_threat_pairs(self):
        low_pairs = {k for k, v in RULES.items() if v[0] == "low"}
        covered = {(c["formation_a"], c["formation_b"]) for c in TEST_CASES
                  if c["expected_threat"] == "low"}
        self.assertTrue(low_pairs.issubset(covered))


if __name__ == "__main__":
    unittest.main()
