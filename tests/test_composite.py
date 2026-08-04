"""
Unit tests for llm_finetuning/composite.py's routing logic -- no GPU/model
calls, mock clients record which branch fired and what prompt they received.

Run with: python -m unittest tests.test_composite -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llm_finetuning"))

from swarm_intent.llm.prompts import ORIGINAL_TEST_CASES, TEST_CASES

from composite import make_composite_run_case, make_composite_battery_run_case, _route
from degradation import build_battery
from holdout_shapes import build_holdout_battery


class _StubClient:
    """Records every prompt it's asked to complete(); returns a fixed
    schema-valid assessment so evaluate_llm-style scoring wouldn't choke."""

    def __init__(self, name):
        self.name = name
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return {
            "situation_summary": "stub", "threat_level": "low", "threat_reasoning": "stub",
            "likely_intent": "unknown", "recommended_action": "monitor",
            "confidence_in_assessment": "low", "key_indicators": [], "follow_up_watch": "stub",
        }


class TestRouteFunction(unittest.TestCase):
    """_route on real ctx strings pulled from the actual battery generators,
    not hand-written strings -- so a change to _render_lines' format is
    caught here too."""

    def test_answerable_degradation_cases_route_to_rules(self):
        battery = build_battery(ORIGINAL_TEST_CASES)
        for case in battery["confidence_decay"]:  # always resolvable, has_ground_truth=True
            self.assertEqual(_route(case["ctx"]), "rules_in_prompt", msg=case["name"])

    def test_multi_hop_unresolvable_routes_to_finetuned(self):
        battery = build_battery(ORIGINAL_TEST_CASES)
        for case in battery["multi_hop"]:
            if case["severity"] in (3, 4):  # unresolvable chains, has_ground_truth=False
                self.assertEqual(_route(case["ctx"]), "finetuned", msg=case["name"])
            elif case["severity"] == 2:  # resolvable 2-hop
                self.assertEqual(_route(case["ctx"]), "rules_in_prompt", msg=case["name"])

    def test_terminal_transitioning_routes_to_finetuned(self):
        battery = build_battery(ORIGINAL_TEST_CASES)
        for case in battery["terminal_transitioning"]:
            self.assertEqual(_route(case["ctx"]), "finetuned", msg=case["name"])

    def test_dropped_transition_line_routes_to_finetuned(self):
        # dropped_lines sev>=1 sometimes drops index 3 (the transition line) --
        # has_ground_truth False in exactly those cases, and _extract_pair has
        # nothing to find without that line either.
        battery = build_battery(ORIGINAL_TEST_CASES)
        for case in battery["dropped_lines"]:
            expected = "finetuned" if not case["has_ground_truth"] else "rules_in_prompt"
            self.assertEqual(_route(case["ctx"]), expected, msg=case["name"])

    def test_deeper_chain_routes_to_finetuned(self):
        battery = build_holdout_battery()
        for case in battery["deeper_chain"]:
            self.assertEqual(_route(case["ctx"]), "finetuned", msg=case["name"])

    def test_dominant_mismatch_routes_to_rules_despite_the_contradiction(self):
        # Another deliberate edge case: shape_dominant_mismatch always includes
        # an explicit "Transition detected at t=20.0s: A -> B" line (the
        # self-contradiction is only in the separate "Dominant formation"
        # line) -- _extract_pair's regex matches the transition line and
        # never looks at "Dominant formation" at all, so this routes to
        # rules_in_prompt exactly like an ordinary resolvable transition
        # would, even though the case is designed to be unanswerable.
        battery = build_holdout_battery()
        for case in battery["dominant_mismatch"]:
            self.assertEqual(_route(case["ctx"]), "rules_in_prompt", msg=case["name"])

    def test_oov_formation_routes_to_rules_despite_unfamiliar_vocabulary(self):
        # Deliberate, documented edge case (composite.py module docstring):
        # "A -> phalanx" is SHAPED like an ordinary resolvable transition, so
        # extractability (a syntactic check) says yes even though "phalanx"
        # isn't a real formation RULES can key on.
        battery = build_holdout_battery()
        for case in battery["oov_formation"]:
            self.assertEqual(_route(case["ctx"]), "rules_in_prompt", msg=case["name"])


class TestCompositeRunCaseFactories(unittest.TestCase):
    def test_clean_battery_run_case_routes_every_case_to_rules(self):
        # TEST_CASES are all synth_context()-generated clean transitions --
        # always resolvable, so the composite should never touch the
        # fine-tuned branch on this battery.
        rules_client, ft_client = _StubClient("rules"), _StubClient("ft")
        branch_log = {}
        run_case = make_composite_run_case(rules_client, ft_client, branch_log, seed=0)
        for case in TEST_CASES:
            run_case(case)
        self.assertEqual(set(branch_log.values()), {"rules_in_prompt"})
        self.assertEqual(len(rules_client.prompts), len(TEST_CASES))
        self.assertEqual(len(ft_client.prompts), 0)

    def test_battery_run_case_splits_across_both_branches(self):
        rules_client, ft_client = _StubClient("rules"), _StubClient("ft")
        branch_log = {}
        run_case = make_composite_battery_run_case(rules_client, ft_client, branch_log)
        battery = build_battery(ORIGINAL_TEST_CASES)
        for case in battery["multi_hop"] + battery["terminal_transitioning"]:
            run_case(case)
        self.assertIn("rules_in_prompt", branch_log.values())
        self.assertIn("finetuned", branch_log.values())
        self.assertGreater(len(rules_client.prompts), 0)
        self.assertGreater(len(ft_client.prompts), 0)
        self.assertEqual(len(rules_client.prompts) + len(ft_client.prompts), len(branch_log))

    def test_branch_log_records_one_entry_per_distinct_case_name(self):
        rules_client, ft_client = _StubClient("rules"), _StubClient("ft")
        branch_log = {}
        run_case = make_composite_battery_run_case(rules_client, ft_client, branch_log)
        battery = build_battery(ORIGINAL_TEST_CASES)
        cases = battery["dropped_lines"]
        for case in cases:
            run_case(case)
        self.assertEqual(len(branch_log), len(cases))


if __name__ == "__main__":
    unittest.main()
