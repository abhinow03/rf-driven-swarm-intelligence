"""
Tests for scripts/check_preregistration_v5a2.py. Every bar gets at least one test that FAILS
when the condition making the bar meaningful is absent -- e.g. a synthetic 0%-abstention
results blob must fail bar e (proving it isn't vacuous, the way v5-a's real 0.0%
abstention_rate_when_unanswerable does), not just a test that a good number passes.

Usage:
    python -m unittest tests.test_check_preregistration_v5a2 -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from check_preregistration_v5a2 import run_check, SAME_POPULATION_CEILING  # noqa: E402


def make_answerable_item(name, true_chain):
    return {"name": name, "true_chain": true_chain, "has_ground_truth": True}


def make_unanswerable_item(name, true_chain):
    return {"name": name, "true_chain": true_chain, "has_ground_truth": False}


def make_results(threat_acc=0.60, over_abst=0.10, under_esc=0.10, over_esc=0.05,
                 schema_rate=0.97, parsed_by_case=None):
    return {
        "answerability": {"accuracy_when_answerable": threat_acc, "over_abstention_rate": over_abst},
        "escalation": {"under_escalated": under_esc, "over_escalated": over_esc},
        "schema_validity_rate": {"rate": schema_rate},
        "parsed_by_case": parsed_by_case or {},
    }


class TestPooledBarsFromAggregates(unittest.TestCase):
    def test_good_aggregates_pass(self):
        results = make_results(threat_acc=0.60, over_abst=0.10, under_esc=0.10, over_esc=0.05)
        rows, _ = run_check(results, memorization=None, items=[])
        by_bar = {r["bar"]: r["status"] for r in rows}
        self.assertEqual(by_bar["threat_accuracy_pooled_when_answerable"], "PASS")
        self.assertEqual(by_bar["over_abstention_rate"], "PASS")
        self.assertEqual(by_bar["escalation_under_escalation_rate"], "PASS")

    def test_bad_threat_accuracy_fails(self):
        results = make_results(threat_acc=0.30)
        rows, overall = run_check(results, memorization=None, items=[])
        by_bar = {r["bar"]: r["status"] for r in rows}
        self.assertEqual(by_bar["threat_accuracy_pooled_when_answerable"], "FAIL")
        self.assertEqual(overall, "FAIL")

    def test_ceiling_normalized_accuracy_uses_locked_denominator(self):
        results = make_results(threat_acc=SAME_POPULATION_CEILING)  # numerator == denominator
        rows, _ = run_check(results, memorization=None, items=[])
        by_bar = {r["bar"]: r["actual"] for r in rows}
        self.assertEqual(by_bar["ceiling_normalized_accuracy"], "100.0%")

    def test_escalation_ceiling_is_a_real_numeric_bound_not_qualitative(self):
        """The OLD bar ('under >= over') passes by construction whenever under-escalation is
        merely the larger of the two -- e.g. under=90%, over=1% would pass that old bar despite
        being catastrophic. The new numeric ceiling must fail this."""
        results = make_results(under_esc=0.90, over_esc=0.01)
        rows, overall = run_check(results, memorization=None, items=[])
        by_bar = {r["bar"]: r["status"] for r in rows}
        self.assertEqual(by_bar["escalation_under_escalation_rate"], "FAIL")
        self.assertEqual(overall, "FAIL")


class TestRealPairAccuracy(unittest.TestCase):
    def test_literal_pair_match_passes(self):
        items = [make_answerable_item("c1", ["diamond", "shield"])]
        parsed = {"c1": {"situation_summary": "The group held a diamond formation before "
                                              "transitioning to a shield formation.",
                         "likely_intent": "defensive"}}
        results = make_results(parsed_by_case=parsed)
        rows, _ = run_check(results, memorization=None, items=items)
        by_bar = {r["bar"]: r for r in rows}
        self.assertEqual(by_bar["pair_accuracy_pooled_when_answerable_real"]["actual"], "100.0%")

    def test_wrong_pair_fails_the_bar_not_silently_passed(self):
        """Proves the metric discriminates: a model narrating the WRONG pair must not be
        scored as correct just because it named some real formation."""
        items = [make_answerable_item(f"c{i}", ["diamond", "shield"]) for i in range(5)]
        parsed = {f"c{i}": {"situation_summary": "The group held a steady column formation "
                                                 "throughout with no transitions.",
                            "likely_intent": "patrol"} for i in range(5)}
        results = make_results(parsed_by_case=parsed)
        rows, overall = run_check(results, memorization=None, items=items)
        by_bar = {r["bar"]: r for r in rows}
        self.assertEqual(by_bar["pair_accuracy_pooled_when_answerable_real"]["actual"], "0.0%")
        self.assertEqual(by_bar["pair_accuracy_pooled_when_answerable_real"]["status"].split(" ")[0], "FAIL")


class TestRegressionVsV5a(unittest.TestCase):
    """Bar i: catches catastrophic forgetting on the same population v5-a was scored on.
    Must fail a threat_accuracy well below v5-a's real 75.65% baseline, and must fail a
    pair_accuracy well below v5-a's REAL (non-proxy) 57.83% baseline -- not the old 63.6%
    proxy figure, which would be a stale, non-comparable denominator."""

    def test_threat_accuracy_far_below_v5a_baseline_fails_regression_bar(self):
        results = make_results(threat_acc=0.30)  # v5-a real: 0.7565, tolerance 5pp -> floor 0.7065
        rows, overall = run_check(results, memorization=None, items=[])
        by_bar = {r["bar"]: r["status"] for r in rows}
        self.assertEqual(by_bar["regression_vs_v5a_threat_accuracy"], "FAIL")
        self.assertEqual(overall, "FAIL")

    def test_threat_accuracy_matching_v5a_baseline_passes_regression_bar(self):
        results = make_results(threat_acc=0.7565)
        rows, _ = run_check(results, memorization=None, items=[])
        by_bar = {r["bar"]: r["status"] for r in rows}
        self.assertEqual(by_bar["regression_vs_v5a_threat_accuracy"], "PASS")

    def test_pair_accuracy_far_below_v5a_real_baseline_fails_regression_bar(self):
        """Uses the REAL (literal) pair_accuracy path, not the old proxy -- a model that
        narrates the wrong pair on every case must fail this, not just the standalone bar b."""
        items = [make_answerable_item(f"c{i}", ["diamond", "shield"]) for i in range(5)]
        parsed = {f"c{i}": {"situation_summary": "steady column formation, no transitions",
                            "likely_intent": "patrol"} for i in range(5)}
        results = make_results(parsed_by_case=parsed)
        rows, overall = run_check(results, memorization=None, items=items)
        by_bar = {r["bar"]: r["status"] for r in rows}
        self.assertEqual(by_bar["regression_vs_v5a_pair_accuracy"], "FAIL")
        self.assertEqual(overall, "FAIL")

    def test_pair_accuracy_matching_v5a_real_baseline_passes_regression_bar(self):
        items = [make_answerable_item("c1", ["diamond", "shield"])]
        parsed = {"c1": {"situation_summary": "transitioned from a diamond formation to a "
                                              "shield formation", "likely_intent": "defensive"}}
        results = make_results(parsed_by_case=parsed)
        rows, _ = run_check(results, memorization=None, items=items)
        by_bar = {r["bar"]: r["status"] for r in rows}
        self.assertEqual(by_bar["regression_vs_v5a_pair_accuracy"], "PASS")


class TestAbstentionBarsAreNotVacuous(unittest.TestCase):
    """The whole point of bars e/f: v5-a scored 0.0% on this exact check (never abstains).
    A results blob that never abstains MUST fail here -- if it didn't, the bar would be
    exactly as vacuous as the old schema_validity_rate bar this document explicitly drops."""

    def _unanswerable_population(self):
        return [
            make_unanswerable_item("mh1", ["diamond", "shield", "column"]),   # multi_hop
            make_unanswerable_item("mh2", ["v_shape", "column", "diamond"]),  # multi_hop
            make_unanswerable_item("mh3", ["shield", "diamond", "v_shape"]),  # multi_hop
            make_unanswerable_item("mh4", ["column", "shield", "diamond"]),   # multi_hop
            make_unanswerable_item("osc1", ["diamond", "shield", "diamond"]),  # oscillation
            make_unanswerable_item("osc2", ["column", "v_shape", "column"]),   # oscillation
        ]

    def test_never_abstaining_fails_both_mechanism_bars(self):
        """Erratum: the pooled 'under_abstention_rate' bar was dropped -- proven
        mathematically identical to the n-weighted average of the two mechanism bars, so it
        added no protection beyond them. This test now asserts on f alone, which already
        fully catches the never-abstain case on its own (both floors fail independently)."""
        items = self._unanswerable_population()
        parsed = {it["name"]: {"situation_summary": "steady formation", "likely_intent": "patrol"}
                 for it in items}
        results = make_results(parsed_by_case=parsed)
        rows, overall = run_check(results, memorization=None, items=items)
        by_bar = {r["bar"]: r["status"] for r in rows}
        self.assertEqual(by_bar["correct_abstention_rate_multi_hop"], "FAIL")
        self.assertEqual(by_bar["correct_abstention_rate_oscillation"], "FAIL")
        self.assertEqual(overall, "FAIL")

    def test_always_correctly_abstaining_passes_both_mechanism_bars(self):
        items = self._unanswerable_population()
        parsed = {it["name"]: {"situation_summary": "ambiguous, multiple formation changes",
                               "likely_intent": "unknown"} for it in items}
        results = make_results(parsed_by_case=parsed)
        rows, _ = run_check(results, memorization=None, items=items)
        by_bar = {r["bar"]: r["status"] for r in rows}
        self.assertEqual(by_bar["correct_abstention_rate_multi_hop"], "PASS")
        self.assertEqual(by_bar["correct_abstention_rate_oscillation"], "PASS")

    def test_pooled_abstention_is_reported_diagnostic_only_never_scored(self):
        """Confirms the rename/drop actually took effect: the pooled figure still appears in
        the report (useful for eyeballing) but as an N/A-status diagnostic row, and its value
        is exactly the n-weighted average of the two mechanism rates -- proving it carries no
        information the separate bars didn't already have."""
        items = self._unanswerable_population()
        parsed = {it["name"]: {"situation_summary": "ambiguous", "likely_intent": "unknown"}
                 if it["name"].startswith("mh") else
                 {"situation_summary": "steady", "likely_intent": "patrol"}
                 for it in items}
        results = make_results(parsed_by_case=parsed)
        rows, _ = run_check(results, memorization=None, items=items)
        pooled_rows = [r for r in rows if r["bar"].startswith("correct_abstention_rate_pooled")]
        self.assertEqual(len(pooled_rows), 1)
        self.assertEqual(pooled_rows[0]["status"], "N/A")
        # 4 multi_hop correct + 0 oscillation correct = 4/6 = 66.7%, the n-weighted average.
        self.assertEqual(pooled_rows[0]["actual"], "66.7%")

    def test_mechanisms_scored_separately_a_strong_one_cannot_mask_a_weak_one(self):
        """sec AK's masking pattern: pooling let a strong mechanism hide a weak one. Here
        multi_hop is always correct and oscillation is never correct -- pooled alone would
        show ~67% (4/6) and could pass a lenient pooled-only bar; the SEPARATE oscillation
        bar must still fail."""
        items = self._unanswerable_population()
        parsed = {}
        for it in items:
            mechanism_is_mh = it["name"].startswith("mh")
            parsed[it["name"]] = {
                "situation_summary": "ambiguous" if mechanism_is_mh else "steady formation",
                "likely_intent": "unknown" if mechanism_is_mh else "patrol",
            }
        results = make_results(parsed_by_case=parsed)
        rows, overall = run_check(results, memorization=None, items=items)
        by_bar = {r["bar"]: r["status"] for r in rows}
        self.assertEqual(by_bar["correct_abstention_rate_multi_hop"], "PASS")
        self.assertEqual(by_bar["correct_abstention_rate_oscillation"], "FAIL")
        self.assertEqual(overall, "FAIL", "a weak oscillation rate must sink overall even "
                                          "though multi_hop alone is perfect")


class TestMemorizationBarIsNotVacuous(unittest.TestCase):
    def test_missing_memorization_file_reports_missing_not_silent_pass(self):
        results = make_results()
        rows, _ = run_check(results, memorization=None, items=[])
        by_bar = {r["bar"]: r["status"] for r in rows}
        self.assertEqual(by_bar["memorization_vs_generalization"], "MISSING")

    def test_high_overlap_rate_fails_as_a_memorization_signal(self):
        results = make_results()
        rows, overall = run_check(results, memorization={"overlap_rate": 0.20}, items=[])
        by_bar = {r["bar"]: r["status"] for r in rows}
        self.assertTrue(by_bar["memorization_vs_generalization"].startswith("FAIL"))
        self.assertEqual(overall, "FAIL")

    def test_low_overlap_rate_passes(self):
        results = make_results()
        rows, _ = run_check(results, memorization={"overlap_rate": 0.01}, items=[])
        by_bar = {r["bar"]: r["status"] for r in rows}
        self.assertTrue(by_bar["memorization_vs_generalization"].startswith("PASS"))


class TestSchemaValidityIsDiagnosticOnly(unittest.TestCase):
    def test_schema_validity_never_scored_into_overall(self):
        """A terrible schema_validity_rate must NOT sink overall -- it's explicitly dropped
        as a bar (shown to pass by construction even for the untrained base model)."""
        results = make_results(threat_acc=0.60, over_abst=0.10, under_esc=0.10, over_esc=0.05,
                               schema_rate=0.0)
        rows, overall = run_check(results, memorization=None, items=[])
        by_bar = {r["bar"]: r["status"] for r in rows}
        self.assertEqual(by_bar["schema_validity_rate (DIAGNOSTIC ONLY, not scored)"], "N/A")


if __name__ == "__main__":
    unittest.main()
