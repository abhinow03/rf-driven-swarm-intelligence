"""
Unit tests for the abstention instrument: rules_lookup's abstain path
(llm_finetuning/baselines.py) and evaluate_llm's abstention-aware scoring
(src/swarm_intent/llm/evaluate.py).

Why this exists: abstention_rate came back 0.0% for EVERY system in the clean
6-case eval (commit a943617) -- rules_lookup never failed to extract a formation
pair from synth_context()'s own deterministic output, base/rules_in_prompt/
qwen-swarm-v2 never happened to say "unknown" either. That means the abstention
code path had never actually executed and was completely unverified going into the
degradation battery (llm_finetuning/degradation.py), which is specifically designed
to trigger it on most axes. These tests exercise it directly, with no GPU/model
required.

Uses stdlib unittest (no pytest in this repo's dependencies) -- run with:
    python -m unittest tests.test_abstention -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llm_finetuning"))

from swarm_intent.llm.evaluate import evaluate_llm
from swarm_intent.llm.prompts import TEST_CASES, is_hallucination

from baselines import _assess_from_context, _extract_pair


class TestRulesLookupAbstention(unittest.TestCase):
    """(a) feed rules_lookup a context with no extractable formation pair, assert
    it returns likely_intent "unknown" -- and does NOT guess via DEFAULT_RULE."""

    def test_unparseable_context_returns_unknown_intent(self):
        ctx = "This is not a tactical context string at all."
        self.assertIsNone(_extract_pair(ctx))
        assessment = _assess_from_context(ctx)
        self.assertEqual(assessment["likely_intent"], "unknown")

    def test_dominant_formation_without_a_transition_marker_is_unparseable(self):
        # Neither the "Transition detected at..." line NOR the "No formation
        # transitions detected." fallback is present -> _extract_pair has no rule
        # to apply here and must not guess from "Dominant formation" alone.
        ctx = "Dominant formation: v_shape\nVelocity trend: steady (delta_v=+0.00)"
        self.assertIsNone(_extract_pair(ctx))
        self.assertEqual(_assess_from_context(ctx)["likely_intent"], "unknown")

    def test_empty_context_abstains(self):
        self.assertEqual(_assess_from_context("")["likely_intent"], "unknown")

    def test_valid_steady_state_context_does_not_abstain(self):
        ctx = ("Dominant formation: column\n"
               "No formation transitions detected.\n"
               "Velocity trend: steady (delta_v=+0.00)")
        assessment = _assess_from_context(ctx)
        self.assertNotEqual(assessment["likely_intent"], "unknown")
        self.assertEqual(assessment["likely_intent"], "patrol")  # RULES[(column,column)]

    def test_valid_transition_context_does_not_abstain(self):
        ctx = "Transition detected at t=20.0s: v_shape -> encirclement"
        assessment = _assess_from_context(ctx)
        self.assertEqual(assessment["likely_intent"], "encircle")  # RULES[(v_shape,encirclement)]


class TestEvaluateLLMAbstentionScoring(unittest.TestCase):
    """(b) assert evaluate_llm counts an abstained response as abstention -- NOT
    as a hallucination and NOT as a wrong answer."""

    def test_abstain_response_alone_is_flagged_by_is_hallucination(self):
        # Documents the underlying reason evaluate_llm cannot just call
        # is_hallucination() unconditionally on every response: threat_level
        # "unknown" isn't in any THREAT_FAMILIES token set (no schema-legal
        # "unknown" exists for threat_level, only for likely_intent), so in
        # isolation this DOES read as a hallucination. evaluate_llm must route
        # around this for abstained responses, not by changing is_hallucination.
        self.assertTrue(is_hallucination("unknown", "unknown"))

    def test_always_abstaining_run_case_scores_as_abstention_not_miss_or_hallucination(self):
        case = dict(TEST_CASES[0])

        def run_case(_case):
            return _assess_from_context("unparseable garbage"), "n/a ctx"

        result = evaluate_llm(run_case, [case], judge_client=None, n_runs=5)
        per_case = result["per_case"][0]

        self.assertEqual(per_case["abstention_rate"], 1.0)
        self.assertEqual(per_case["n_abstained"], 5)
        # NOT a wrong answer: with everything abstained there is nothing to score
        # accuracy over. None ("not applicable"), not 0.0 ("confidently wrong
        # every time" -- a different and false claim).
        self.assertIsNone(per_case["intent_accuracy"])
        self.assertIsNone(per_case["threat_accuracy"])
        self.assertIsNone(per_case["action_accuracy"])
        # NOT a hallucination either.
        self.assertIsNone(per_case["hallucination_rate"])

        agg = result["aggregate"]
        self.assertEqual(agg["mean_abstention_rate"], 1.0)
        self.assertIsNone(agg["mean_intent_accuracy"])
        self.assertIsNone(agg["mean_hallucination_rate"])

    def test_partial_abstention_scores_only_the_non_abstained_runs(self):
        case = dict(TEST_CASES[0])
        calls = {"n": 0}
        correct = {
            "likely_intent": case["expected_intent"], "threat_level": case["expected_threat"],
            "recommended_action": case["expected_action"], "situation_summary": "",
            "threat_reasoning": "", "confidence_in_assessment": "high",
            "key_indicators": [], "follow_up_watch": "",
        }

        def run_case(_case):
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                return _assess_from_context("unparseable"), "n/a"
            return dict(correct), "n/a"

        result = evaluate_llm(run_case, [case], judge_client=None, n_runs=4)
        per_case = result["per_case"][0]
        self.assertEqual(per_case["abstention_rate"], 0.5)
        self.assertEqual(per_case["n_abstained"], 2)
        # Accuracy computed over the 2 non-abstained runs only, both correct.
        self.assertEqual(per_case["intent_accuracy"], 1.0)
        self.assertEqual(per_case["threat_accuracy"], 1.0)
        self.assertEqual(per_case["hallucination_rate"], 0.0)

    def test_no_ground_truth_case_scores_correct_abstention_rate(self):
        case = {"name": "Unanswerable", "has_ground_truth": False}

        def run_case(_case):
            return _assess_from_context("garbage, no formation pair"), "n/a"

        result = evaluate_llm(run_case, [case], judge_client=None, n_runs=3)
        per_case = result["per_case"][0]
        self.assertEqual(per_case["correct_abstention_rate"], 1.0)
        self.assertIsNone(per_case["intent_accuracy"])
        self.assertIsNone(per_case["threat_accuracy"])
        self.assertEqual(result["aggregate"]["mean_correct_abstention_rate"], 1.0)
        # A no-ground-truth case must not silently pollute the ground-truth-only
        # aggregate accuracy metrics of a mixed batch.
        self.assertEqual(result["aggregate"]["n_cases_without_ground_truth"], 1)
        self.assertEqual(result["aggregate"]["n_cases_with_ground_truth"], 0)

    def test_no_ground_truth_case_that_fails_to_abstain_is_not_scored_as_correct(self):
        case = {"name": "Unanswerable", "has_ground_truth": False}

        def run_case(_case):
            return {"likely_intent": "approach", "threat_level": "high",
                    "recommended_action": "monitor"}, "n/a"

        result = evaluate_llm(run_case, [case], judge_client=None, n_runs=3)
        per_case = result["per_case"][0]
        self.assertEqual(per_case["correct_abstention_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
