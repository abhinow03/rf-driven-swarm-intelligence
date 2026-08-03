"""
Step 2 of the "settle delta_v, recalibrate synth_context, diagnose the prior
skew" session (AUDIT.md sec W/continued): asserts synth_context()
(llm_finetuning/build_sft_dataset.py) samples its numeric fields from the real
STGT population (REAL_REG_PERCENTILES, AUDIT.md sec V, n=5879) rather than the
old hand-picked uniform ranges, and that the resulting converging/dispersing/
stable proportions track the real rates within a few points.

Usage:
    python -m unittest tests.test_synth_context_recalibration -v
"""
from __future__ import annotations

import os
import random
import sys
import unittest
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llm_finetuning"))

from swarm_intent.config import BASE_FORMATIONS  # noqa: E402

from build_sft_dataset import synth_context, REAL_REG_PERCENTILES, _sample_real  # noqa: E402

N_SAMPLES = 2000
# Real rates, AUDIT.md sec V (n=5879 real STGT regression labels).
REAL_CONVERGING_PCT = 29.73
REAL_DISPERSING_PCT = 9.66
REAL_STABLE_PCT = 60.60
TOLERANCE_POINTS = 6.0


class TestFieldsWithinRealRange(unittest.TestCase):
    """velocity/approach/delta_v are direct bootstrap draws from
    REAL_REG_PERCENTILES, so they're within the real range by construction --
    this test still exercises it end-to-end through synth_context() rather
    than trusting the construction argument alone."""

    @classmethod
    def setUpClass(cls):
        rng = random.Random(1)
        cls.samples = []
        for _ in range(N_SAMPLES):
            form_a, form_b = rng.choice(BASE_FORMATIONS), rng.choice(BASE_FORMATIONS)
            ctx, key_windows = synth_context(form_a, form_b, rng)
            cls.samples.append((ctx, key_windows))

    # Fields are round()-ed to 2-3dp for display after being sampled, so a value
    # drawn exactly at the empirical min/max can round to a hair outside it
    # (e.g. 0.4558 -> 0.456) -- EPS absorbs display rounding, not sampling error.
    EPS = 0.006

    def test_velocity_within_real_range(self):
        lo, hi = min(REAL_REG_PERCENTILES["velocity_physical"]), max(REAL_REG_PERCENTILES["velocity_physical"])
        for _, key_windows in self.samples:
            for kw in key_windows:
                self.assertGreaterEqual(kw["velocity"], lo - self.EPS)
                self.assertLessEqual(kw["velocity"], hi + self.EPS)

    def test_approach_within_real_range(self):
        lo, hi = min(REAL_REG_PERCENTILES["approach_rate"]), max(REAL_REG_PERCENTILES["approach_rate"])
        for _, key_windows in self.samples:
            for kw in key_windows:
                self.assertGreaterEqual(kw["approach"], lo - self.EPS)
                self.assertLessEqual(kw["approach"], hi + self.EPS)

    def test_delta_v_within_real_range(self):
        lo, hi = min(REAL_REG_PERCENTILES["delta_v_physical"]), max(REAL_REG_PERCENTILES["delta_v_physical"])
        for ctx, _ in self.samples:
            line = next(l for l in ctx.splitlines() if l.startswith("Velocity trend"))
            delta_v = float(line.split("delta_v=")[1].rstrip(")"))
            self.assertGreaterEqual(delta_v, lo - self.EPS)
            self.assertLessEqual(delta_v, hi + self.EPS)

    def test_stability_within_valid_clipped_range(self):
        # stab_early/stab_late are DERIVED (mean_draw +- delta_draw/2, clipped to
        # [0,1]) from two independent real marginal draws, not a joint real
        # sample -- so they're guaranteed within [0,1] (production's own clip
        # range) but not strictly within the raw real [min,max], see
        # synth_context()'s docstring.
        for _, key_windows in self.samples:
            for kw in key_windows:
                self.assertGreaterEqual(kw["stability"], 0.0)
                self.assertLessEqual(kw["stability"], 1.0)

    def test_sample_real_draws_are_from_the_percentile_array(self):
        rng = random.Random(2)
        for field, breakpoints in REAL_REG_PERCENTILES.items():
            for _ in range(200):
                self.assertIn(_sample_real(rng, field), breakpoints)


class TestSpreadDynamicsProportionsMatchReal(unittest.TestCase):
    """The one proportion match the session explicitly requires: converging/
    dispersing/stable must track the real rates within a few points."""

    @classmethod
    def setUpClass(cls):
        rng = random.Random(3)
        counts = Counter()
        for _ in range(N_SAMPLES):
            form_a, form_b = rng.choice(BASE_FORMATIONS), rng.choice(BASE_FORMATIONS)
            ctx, _ = synth_context(form_a, form_b, rng)
            line = next(l for l in ctx.splitlines() if l.startswith("Spread dynamics"))
            label = line.split(": ", 1)[1].split(" (mean")[0]
            counts[label] += 1
        cls.pct = {k: 100 * v / N_SAMPLES for k, v in counts.items()}

    def test_converging_within_tolerance(self):
        pct = self.pct.get("converging (drones closing in)", 0.0)
        self.assertAlmostEqual(pct, REAL_CONVERGING_PCT, delta=TOLERANCE_POINTS)

    def test_dispersing_within_tolerance(self):
        pct = self.pct.get("dispersing (drones spreading out)", 0.0)
        self.assertAlmostEqual(pct, REAL_DISPERSING_PCT, delta=TOLERANCE_POINTS)

    def test_stable_within_tolerance(self):
        pct = self.pct.get("stable spread", 0.0)
        self.assertAlmostEqual(pct, REAL_STABLE_PCT, delta=TOLERANCE_POINTS)


if __name__ == "__main__":
    unittest.main()
