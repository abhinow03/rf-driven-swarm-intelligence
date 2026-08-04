"""
AUDIT.md sec AE step 1: unit tests for src/swarm_intent/llm/prior_correction.py's
scoped_correct(). The guard this file exists to enforce: the sec AD global
correction took high-threat accuracy from 35.7% to 14.3% by letting the
log-p(c) boost for a rare class (critical, ~4% frequency) drag correctly-
classified `high` predictions into `critical`. scoped_correct restricts
correction to the medium/low near-tie shape only -- these tests assert that
restriction actually holds, with no GPU/model needed (pure dict math).

Usage:
    python -m unittest tests.test_prior_correction -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from swarm_intent.llm.prior_correction import scoped_correct, CANDIDATES  # noqa: E402

# RULES' own canonical class frequency (llm_finetuning/report_class_balance.py),
# the same reference distribution AUDIT.md sec AD/AE use.
RULES_CLASS_FREQ = {"low": 13 / 49, "medium": 22 / 49, "high": 12 / 49, "critical": 2 / 49}


def _p(**kw):
    """Build a full CANDIDATES softmax dict, filling any omitted class with a
    tiny residual so every dict sums to ~1 and every key is present."""
    total = sum(kw.values())
    residual_classes = [c for c in CANDIDATES if c not in kw]
    residual = max(1e-6, 1.0 - total)
    per = residual / len(residual_classes) if residual_classes else 0.0
    out = {c: kw.get(c, per) for c in CANDIDATES}
    return out


class TestHighCriticalNeverModified(unittest.TestCase):
    """The core guard: whatever the runner-up is, a high or critical ARGMAX must
    never be touched by scoped_correct -- this is the exact failure mode (a
    correctly-classified `high` prediction flipping to `critical`) sec AD found."""

    def test_high_argmax_with_critical_runnerup_untouched(self):
        raw_p = _p(high=0.72, critical=0.14, medium=0.10, low=0.04)
        out = scoped_correct(raw_p, RULES_CLASS_FREQ)
        self.assertFalse(out["applied"])
        self.assertEqual(out["corrected_argmax"], "high")

    def test_high_argmax_with_medium_runnerup_untouched(self):
        raw_p = _p(high=0.55, medium=0.30, low=0.10, critical=0.05)
        out = scoped_correct(raw_p, RULES_CLASS_FREQ)
        self.assertFalse(out["applied"])
        self.assertEqual(out["corrected_argmax"], "high")

    def test_critical_argmax_untouched(self):
        raw_p = _p(critical=0.60, high=0.35, medium=0.04, low=0.01)
        out = scoped_correct(raw_p, RULES_CLASS_FREQ)
        self.assertFalse(out["applied"])
        self.assertEqual(out["corrected_argmax"], "critical")

    def test_every_high_critical_argmax_case_in_a_sweep_is_never_modified(self):
        """Sweep many (argmax=high/critical, varying runner-up) shapes -- none
        should ever flip, regardless of how the remaining probability mass is
        distributed among the other three classes."""
        import itertools
        for argmax_cls in ("high", "critical"):
            for runner_up in CANDIDATES:
                if runner_up == argmax_cls:
                    continue
                raw_p = _p(**{argmax_cls: 0.5, runner_up: 0.3})
                out = scoped_correct(raw_p, RULES_CLASS_FREQ)
                self.assertFalse(out["applied"], f"{argmax_cls}/{runner_up} was modified")
                self.assertEqual(out["corrected_argmax"], argmax_cls)


class TestMediumHighCriticalRunnerupOutOfScope(unittest.TestCase):
    """The exact sec AD bug shape: argmax=medium, runner-up=high or critical --
    this must be out of scope (never corrected), unlike a medium/low near-tie."""

    def test_medium_argmax_high_runnerup_out_of_scope(self):
        raw_p = _p(medium=0.50, high=0.35, low=0.10, critical=0.05)
        out = scoped_correct(raw_p, RULES_CLASS_FREQ)
        self.assertFalse(out["applied"])
        self.assertEqual(out["corrected_argmax"], "medium")
        self.assertIn("out of scope", out["reason"])

    def test_medium_argmax_critical_runnerup_out_of_scope(self):
        raw_p = _p(medium=0.60, critical=0.25, low=0.10, high=0.05)
        out = scoped_correct(raw_p, RULES_CLASS_FREQ)
        self.assertFalse(out["applied"])
        self.assertEqual(out["corrected_argmax"], "medium")


class TestMediumLowNearTieInScope(unittest.TestCase):
    """The one shape scoped_correct is allowed to touch: argmax=medium,
    runner-up=low. Mirrors the actual low-threat collapse (secs Y/CC/AA)."""

    def test_near_tie_can_flip_to_low(self):
        # medium's raw edge is thin (0.02) but medium's RULES frequency (44.9%)
        # is ~1.7x low's (26.5%) -- the mild ratio the original correction was
        # designed for -- so the correction should be strong enough to flip this.
        raw_p = _p(medium=0.43, low=0.41, high=0.12, critical=0.04)
        out = scoped_correct(raw_p, RULES_CLASS_FREQ)
        self.assertTrue(out["applied"])
        self.assertEqual(out["corrected_argmax"], "low")

    def test_near_tie_can_stay_medium_if_gap_too_wide(self):
        raw_p = _p(medium=0.70, low=0.20, high=0.07, critical=0.03)
        out = scoped_correct(raw_p, RULES_CLASS_FREQ)
        self.assertFalse(out["applied"])
        self.assertEqual(out["corrected_argmax"], "medium")

    def test_in_scope_correction_never_produces_high_or_critical(self):
        """Even in scope, the corrected label must come from {low, medium}
        only -- never high/critical, regardless of how much raw mass they hold,
        since they were never the near-tie candidate in the first place."""
        raw_p = _p(medium=0.34, low=0.33, high=0.30, critical=0.03)
        out = scoped_correct(raw_p, RULES_CLASS_FREQ)
        self.assertIn(out["corrected_argmax"], ("low", "medium"))


if __name__ == "__main__":
    unittest.main()
