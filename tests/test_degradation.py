"""
Tests for the degradation battery (llm_finetuning/degradation.py). No GPU/model
required -- these test battery construction and rules_lookup's behaviour on it,
same style as tests/test_abstention.py.
"""
from __future__ import annotations

import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llm_finetuning"))

from swarm_intent.llm.evaluate import evaluate_llm
from swarm_intent.llm.prompts import ORIGINAL_TEST_CASES

from degradation import build_battery, make_rules_lookup_battery_run_case, _stable_seed

REPO = os.path.join(os.path.dirname(__file__), "..")


class TestBatteryStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.battery = build_battery(ORIGINAL_TEST_CASES)

    def test_expected_axis_and_case_counts(self):
        expected = {"multi_hop": 18, "terminal_transitioning": 18, "confidence_decay": 30,
                    "dropped_lines": 24, "contradictory_cues": 18}
        counts = {axis: len(cases) for axis, cases in self.battery.items()}
        self.assertEqual(counts, expected)

    def test_every_case_has_required_fields(self):
        for axis, cases in self.battery.items():
            for c in cases:
                self.assertIn("ctx", c)
                self.assertIn("key_windows", c)
                self.assertIn("has_ground_truth", c)
                if c["has_ground_truth"]:
                    self.assertIn("expected_intent", c)
                    self.assertIn("expected_threat", c)
                    self.assertIn("expected_action", c)


class TestStableSeedReproducibility(unittest.TestCase):
    """The original bug: dropped_lines used Python's built-in hash() on a string
    tuple as an RNG seed. hash() on strings is salted per-process (hash
    randomization, on by default) -- so which lines got dropped, and therefore
    which cases lost ground truth, silently changed every time the interpreter
    restarted. Caught by literally running build_battery() in two separate
    `python -c` processes and diffing the output (both showed identical results
    before this fix would have -- since it uses the same run's hash seed for the
    whole script -- so the fix is verified with a real subprocess comparison here,
    not just an in-process call which can't exercise the bug at all)."""

    def test_stable_seed_is_a_pure_function(self):
        self.assertEqual(_stable_seed("Converging Attack", 1), _stable_seed("Converging Attack", 1))
        self.assertNotEqual(_stable_seed("Converging Attack", 1), _stable_seed("Converging Attack", 2))

    def test_dropped_lines_has_ground_truth_identical_across_separate_processes(self):
        script = (
            "import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'llm_finetuning')\n"
            "from degradation import build_battery\n"
            "from swarm_intent.llm.prompts import ORIGINAL_TEST_CASES\n"
            "battery = build_battery(ORIGINAL_TEST_CASES)\n"
            "print([(c['name'], c['has_ground_truth']) for c in battery['dropped_lines']])\n"
        )
        outputs = set()
        for _ in range(2):
            result = subprocess.run([sys.executable, "-c", script], cwd=REPO,
                                    capture_output=True, text=True, check=True)
            outputs.add(result.stdout)
        self.assertEqual(len(outputs), 1, "dropped_lines has_ground_truth differed across process runs")


class TestRulesLookupOnBattery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.battery = build_battery(ORIGINAL_TEST_CASES)
        # staticmethod() prevents the descriptor protocol from binding `self` as an
        # implicit first argument when a plain function is stored as a class attr.
        cls.run_case = staticmethod(make_rules_lookup_battery_run_case())

    def test_abstains_exactly_when_no_ground_truth(self):
        # rules_lookup has no model/randomness in the loop -- its abstention should
        # match has_ground_truth exactly, case for case, on every axis.
        for axis, cases in self.battery.items():
            for c in cases:
                assessment, _ = self.run_case(c)
                abstained = assessment["likely_intent"] == "unknown"
                self.assertEqual(abstained, not c["has_ground_truth"],
                                 f"{c['name']}: has_ground_truth={c['has_ground_truth']} "
                                 f"but abstained={abstained}")

    def test_multi_hop_collapses_at_severity_3_and_4(self):
        for c in self.battery["multi_hop"]:
            if c["severity"] in (3, 4):
                self.assertFalse(c["has_ground_truth"])

    def test_terminal_transitioning_always_unanswerable_by_rules(self):
        for c in self.battery["terminal_transitioning"]:
            self.assertFalse(c["has_ground_truth"])

    def test_evaluate_llm_correct_abstention_rate_on_multi_hop(self):
        by_sev = {}
        for c in self.battery["multi_hop"]:
            by_sev.setdefault(c["severity"], []).append(c)
        res2 = evaluate_llm(self.run_case, by_sev[2], judge_client=None, n_runs=1)
        res3 = evaluate_llm(self.run_case, by_sev[3], judge_client=None, n_runs=1)
        self.assertEqual(res2["aggregate"]["mean_intent_accuracy"], 1.0)
        self.assertIsNone(res2["aggregate"]["mean_correct_abstention_rate"])
        self.assertIsNone(res3["aggregate"]["mean_intent_accuracy"])
        self.assertEqual(res3["aggregate"]["mean_correct_abstention_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
