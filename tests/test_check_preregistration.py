"""
Tests for scripts/check_preregistration.py against synthetic results
dictionaries crafted to pass some bars and fail others, so the scorer's
per-bar logic is verified before any real v5-a result exists.

Usage:
    python -m unittest tests.test_check_preregistration -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from check_preregistration import run_check  # noqa: E402


def make_results(threat_acc=0.60, low_acc=0.70, over_abst=0.10, schema_rate=0.97,
                 under_esc=0.10, over_esc=0.05):
    return {
        "answerability": {"accuracy_when_answerable": threat_acc, "over_abstention_rate": over_abst},
        "per_class_threat_accuracy": {"low": {"accuracy": low_acc}},
        "schema_validity_rate": {"rate": schema_rate},
        "escalation": {"under_escalated": under_esc, "over_escalated": over_esc},
    }


class TestAllBarsPass(unittest.TestCase):
    def test_all_pass_gives_overall_pass(self):
        results = make_results(threat_acc=0.60, low_acc=0.70, over_abst=0.10,
                               schema_rate=0.97, under_esc=0.10, over_esc=0.05)
        rows, overall = run_check(results)
        self.assertEqual(overall, "PASS")
        by_bar = {r["bar"]: r["status"] for r in rows}
        self.assertEqual(by_bar["threat_accuracy_pooled_when_answerable"], "PASS")
        self.assertEqual(by_bar["threat_accuracy_low_per_class"], "PASS")
        self.assertEqual(by_bar["over_abstention_rate"], "PASS")
        self.assertEqual(by_bar["schema_validity_rate"], "PASS")
        self.assertTrue(by_bar["escalation_direction_under_ge_over"].startswith("PASS"))


class TestSomeBarsFail(unittest.TestCase):
    def test_threat_accuracy_below_bar_fails(self):
        results = make_results(threat_acc=0.40)  # bar is >=55%
        rows, overall = run_check(results)
        by_bar = {r["bar"]: r["status"] for r in rows}
        self.assertEqual(by_bar["threat_accuracy_pooled_when_answerable"], "FAIL")
        self.assertEqual(overall, "FAIL")

    def test_low_threat_accuracy_below_bar_fails(self):
        results = make_results(low_acc=0.50)  # bar is >=65%
        rows, overall = run_check(results)
        by_bar = {r["bar"]: r["status"] for r in rows}
        self.assertEqual(by_bar["threat_accuracy_low_per_class"], "FAIL")
        self.assertEqual(overall, "FAIL")

    def test_over_abstention_above_bar_fails(self):
        results = make_results(over_abst=0.30)  # bar is <=15%
        rows, overall = run_check(results)
        by_bar = {r["bar"]: r["status"] for r in rows}
        self.assertEqual(by_bar["over_abstention_rate"], "FAIL")
        self.assertEqual(overall, "FAIL")

    def test_schema_validity_below_bar_fails(self):
        results = make_results(schema_rate=0.80)  # bar is >=95%
        rows, overall = run_check(results)
        by_bar = {r["bar"]: r["status"] for r in rows}
        self.assertEqual(by_bar["schema_validity_rate"], "FAIL")
        self.assertEqual(overall, "FAIL")

    def test_over_escalation_dominant_flagged_as_fail(self):
        results = make_results(under_esc=0.05, over_esc=0.20)
        rows, overall = run_check(results)
        by_bar = {r["bar"]: r["status"] for r in rows}
        self.assertTrue(by_bar["escalation_direction_under_ge_over"].startswith("FAIL"))
        self.assertEqual(overall, "FAIL")

    def test_mixed_pass_and_fail_gives_overall_fail_not_averaged(self):
        """4/5 bars passing must NOT average to an overall pass -- any single
        FAIL means overall FAIL, no partial credit."""
        results = make_results(threat_acc=0.60, low_acc=0.70, over_abst=0.10,
                               schema_rate=0.50,  # this one fails
                               under_esc=0.10, over_esc=0.05)
        rows, overall = run_check(results)
        by_bar = {r["bar"]: r["status"] for r in rows}
        self.assertEqual(by_bar["schema_validity_rate"], "FAIL")
        self.assertEqual(by_bar["threat_accuracy_pooled_when_answerable"], "PASS")
        self.assertEqual(overall, "FAIL")


class TestMissingFields(unittest.TestCase):
    def test_missing_metric_reported_as_missing_not_pass_or_fail(self):
        results = {"answerability": {}, "per_class_threat_accuracy": {},
                  "schema_validity_rate": {}, "escalation": {}}
        rows, overall = run_check(results)
        by_bar = {r["bar"]: r["status"] for r in rows}
        self.assertEqual(by_bar["threat_accuracy_pooled_when_answerable"], "MISSING")
        self.assertEqual(by_bar["schema_validity_rate"], "MISSING")
        # MISSING must block overall PASS, exactly like FAIL does -- an incomplete
        # results file must never be reported as having cleared the bars.
        self.assertEqual(overall, "FAIL")

    def test_missing_is_visibly_distinct_from_pass_in_the_row(self):
        results = {"answerability": {}, "per_class_threat_accuracy": {},
                  "schema_validity_rate": {}, "escalation": {}}
        rows, _ = run_check(results)
        statuses = {r["status"] for r in rows}
        self.assertIn("MISSING", statuses)
        self.assertNotIn("PASS", {r["status"] for r in rows if r["bar"] != "escalation_direction_under_ge_over"
                                  and r["bar"] != "pair_accuracy_pooled_when_answerable"})


class TestPairAccuracyFlaggedNotComputable(unittest.TestCase):
    def test_pair_accuracy_bar_always_reports_not_computable(self):
        """This bar exists in PREREGISTRATION.md but has no corresponding
        field in eval_sft_v5.py's schema -- must never be silently scored as
        a pass, regardless of what the rest of the results look like."""
        results = make_results()  # otherwise all-passing
        rows, _ = run_check(results)
        pair_row = next(r for r in rows if r["bar"] == "pair_accuracy_pooled_when_answerable")
        self.assertIn("NOT COMPUTABLE", pair_row["status"])


if __name__ == "__main__":
    unittest.main()
