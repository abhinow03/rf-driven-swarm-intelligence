"""
AUDIT.md sec AE step 3: unit tests for src/swarm_intent/pipeline_v2.py's
Layer 1 (deterministic RULES + narrator LLM, validated/overwritten) and
Layer 2 (guard abstention) logic, using fake clients -- no GPU/model needed.
Layer 3 (v3b-fix + scoped correction) needs a real model/tokenizer for its
teacher-forced rescoring pass and is exercised by the real eval run instead
(scoped_correct() itself is fully unit-tested in tests/test_prior_correction.py).

Usage:
    python -m unittest tests.test_pipeline_v2 -v
"""
from __future__ import annotations

import os
import sys
import unittest
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from swarm_intent.pipeline_v2 import (  # noqa: E402
    _finalize_layer1, _layer2_abstain, assess_ctx, make_pipeline_v2_run_case,
    make_pipeline_v2_battery_run_case, LAYER_1_DETERMINISTIC, LAYER_2_GUARD,
)


class FakeNarratorClient:
    """A stand-in for LocalHFClient: .complete(prompt, system_prompt=None) -> dict.
    `response` may be a fixed dict, or a callable(prompt, system_prompt) -> dict."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete(self, prompt, system_prompt=None):
        self.calls.append((prompt, system_prompt))
        return self.response(prompt, system_prompt) if callable(self.response) else dict(self.response)


class TestFinalizeLayer1(unittest.TestCase):
    def test_compliant_llm_output_has_no_deviation(self):
        llm_out = {"threat_level": "low", "likely_intent": "patrol", "recommended_action": "monitor",
                  "situation_summary": "steady patrol"}
        assessment, deviation = _finalize_layer1(llm_out, "low", "patrol", "monitor")
        self.assertEqual(deviation, {})
        self.assertEqual(assessment["threat_level"], "low")
        self.assertEqual(assessment["situation_summary"], "steady patrol")

    def test_deviating_llm_output_is_overwritten_and_logged(self):
        llm_out = {"threat_level": "critical", "likely_intent": "encircle",
                  "recommended_action": "deploy_countermeasure"}
        assessment, deviation = _finalize_layer1(llm_out, "low", "patrol", "monitor")
        # overwritten regardless of what the LLM said
        self.assertEqual(assessment["threat_level"], "low")
        self.assertEqual(assessment["likely_intent"], "patrol")
        self.assertEqual(assessment["recommended_action"], "monitor")
        # but the deviation is logged, not silently dropped
        self.assertEqual(deviation, {"threat_level": "critical", "likely_intent": "encircle",
                                     "recommended_action": "deploy_countermeasure"})

    def test_non_dict_llm_output_still_produces_a_valid_assessment(self):
        assessment, deviation = _finalize_layer1({"error": "JSON parse failed"}, "high", "approach", "alert_operator")
        self.assertEqual(assessment["threat_level"], "high")
        self.assertIn("situation_summary", assessment)
        # a failed/errored LLM call has no decision-field keys at all -- every field
        # counts as a (None) deviation, logged just like a wrong answer would be.
        self.assertEqual(deviation, {"threat_level": None, "likely_intent": None, "recommended_action": None})


class TestLayer2Abstain(unittest.TestCase):
    def test_abstention_is_schema_legal_unknown(self):
        a = _layer2_abstain(["low_confidence"])
        self.assertEqual(a["threat_level"], "unknown")
        self.assertEqual(a["likely_intent"], "unknown")
        self.assertIn("0.6 confidence", a["threat_reasoning"])

    def test_multiple_guard_reasons_all_appear_in_reasoning(self):
        a = _layer2_abstain(["oov_name", "dominant_history_contradiction"])
        self.assertIn("out-of-vocabulary", a["threat_reasoning"])
        self.assertIn("dominant formation", a["threat_reasoning"])


class TestAssessCtxRouting(unittest.TestCase):
    def test_bucket_a_steady_state_uses_rules_dict_not_llm_decision(self):
        ctx = ("Dominant formation: column\nFormation history: column\n"
              "No formation transitions detected.")
        key_windows = [{"confidence": 0.9, "formation": "column", "velocity": 5.0,
                        "approach": 0.0, "stability": 0.8, "from": None, "to": None}]
        # narrator returns a WRONG decision -- must be overwritten
        narrator = FakeNarratorClient({"threat_level": "critical", "likely_intent": "encircle",
                                       "recommended_action": "deploy_countermeasure",
                                       "situation_summary": "hallucinated"})
        assessment, layer, detail = assess_ctx(narrator, None, ctx, key_windows, {})
        self.assertEqual(layer, LAYER_1_DETERMINISTIC)
        # column->column is RULES: (low, patrol, monitor)
        self.assertEqual(assessment["threat_level"], "low")
        self.assertEqual(assessment["likely_intent"], "patrol")
        self.assertEqual(assessment["recommended_action"], "monitor")
        self.assertEqual(detail["llm_deviation"]["threat_level"], "critical")
        self.assertEqual(narrator.calls[0][1], "You are a tactical AI analyst assisting a counter-UAV "
                         "operator. The threat_level, likely_intent, and recommended_action for this "
                         "observation have ALREADY been determined by the canonical rule engine and are "
                         "given to you below -- you must NOT question, second-guess, or change them under "
                         "any circumstance. Your only job is to write the supporting narrative fields "
                         "(situation_summary, threat_reasoning, key_indicators, follow_up_watch) that "
                         "explain and support the given decision. Echo threat_level / likely_intent / "
                         "recommended_action back EXACTLY as given.")

    def test_bucket_b_oov_never_calls_any_client(self):
        ctx = ("Dominant formation: column\nFormation history: column -> transitioning -> phalanx\n"
              "Transition detected at t=20.0s: column -> phalanx")
        key_windows = [{"confidence": 0.9}]
        assessment, layer, detail = assess_ctx(None, None, ctx, key_windows, {})
        self.assertEqual(layer, LAYER_2_GUARD)
        self.assertEqual(assessment["threat_level"], "unknown")
        self.assertIn("oov_name", detail["guard_reasons"])


class TestRunCaseFactories(unittest.TestCase):
    def test_pipeline_v2_run_case_logs_layer_for_a_known_test_case(self):
        narrator = FakeNarratorClient({"threat_level": "low", "likely_intent": "patrol",
                                       "recommended_action": "monitor"})
        layer_log = defaultdict(list)
        run_case = make_pipeline_v2_run_case(narrator, None, {}, layer_log, seed=0)
        case = {"name": "test_steady_column", "formation_a": "column", "formation_b": "column"}
        assessment, ctx = run_case(case)
        self.assertEqual(assessment["threat_level"], "low")
        self.assertEqual(layer_log["test_steady_column"][0]["layer"], LAYER_1_DETERMINISTIC)

    def test_pipeline_v2_battery_run_case_uses_precomputed_ctx(self):
        narrator = FakeNarratorClient({"threat_level": "low", "likely_intent": "patrol",
                                       "recommended_action": "monitor"})
        layer_log = defaultdict(list)
        run_case = make_pipeline_v2_battery_run_case(narrator, None, {}, layer_log)
        case = {"name": "battery_case_1",
               "ctx": "Dominant formation: diamond\nFormation history: diamond\n"
                      "No formation transitions detected.",
               "key_windows": [{"confidence": 0.9}]}
        assessment, ctx = run_case(case)
        self.assertEqual(ctx, case["ctx"])
        self.assertEqual(layer_log["battery_case_1"][0]["layer"], LAYER_1_DETERMINISTIC)
        # diamond->diamond is RULES: (low, patrol, monitor)
        self.assertEqual(assessment["threat_level"], "low")


if __name__ == "__main__":
    unittest.main()
