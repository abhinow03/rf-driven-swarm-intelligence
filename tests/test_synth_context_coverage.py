"""
Step 3 of the "resolve the calibration/gap contradiction" session (AUDIT.md sec BB):
asserts every context_spec.py value is reachable from synth_context() (llm_finetuning/
build_sft_dataset.py), after fixing two real train/serve mismatches:
  a. "Role differentiation: ..." was never rendered into the training prompt text at
     all, despite build_tactical_context() always emitting it in production.
  b. spread_dynamics and stability_trend were binary in the generator (missing the
     "dispersing" and "improving" branches) despite context_spec.py defining 3-way
     vocabularies matching production's calibration.py thresholds.

Does NOT touch RULES or regenerate any dataset -- this is a reachability test on the
context-generation function only.

Usage:
    python -m unittest tests.test_synth_context_coverage -v
"""
from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llm_finetuning"))

from swarm_intent import context_spec as spec  # noqa: E402
from swarm_intent.config import BASE_FORMATIONS  # noqa: E402

from build_sft_dataset import synth_context  # noqa: E402

N_SAMPLES = 500


class TestSynthContextCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = random.Random(12345)
        cls.ctxs = []
        cls.key_windows_list = []
        for _ in range(N_SAMPLES):
            form_a = rng.choice(BASE_FORMATIONS)
            form_b = rng.choice(BASE_FORMATIONS)
            ctx, key_windows = synth_context(form_a, form_b, rng)
            cls.ctxs.append(ctx)
            cls.key_windows_list.append(key_windows)

    def _realized_values(self, marker: str, candidates: tuple) -> set:
        """Matches by which known candidate value the line's payload STARTS WITH --
        not by generic "(" splitting, since spread_dynamics' own values (e.g.
        "converging (drones closing in)") legitimately contain a parenthetical, and
        naively splitting at the first "(" would truncate them."""
        realized = set()
        for ctx in self.ctxs:
            for line in ctx.splitlines():
                if line.startswith(marker):
                    payload = line[len(marker):]
                    for cand in candidates:
                        if payload.startswith(cand):
                            realized.add(cand)
                            break
        return realized

    def test_velocity_trend_fully_reachable(self):
        realized = self._realized_values("Velocity trend: ", spec.VELOCITY_TREND_VALUES)
        self.assertEqual(realized, set(spec.VELOCITY_TREND_VALUES))

    def test_spread_dynamics_fully_reachable(self):
        realized = self._realized_values("Spread dynamics: ", spec.SPREAD_DYNAMICS_VALUES)
        self.assertEqual(realized, set(spec.SPREAD_DYNAMICS_VALUES),
                         "spread_dynamics must reach 'dispersing' too, not just "
                         "'converging'/'stable' -- this was the binary-threshold bug")

    def test_stability_trend_fully_reachable(self):
        realized = self._realized_values("Formation stability: ", spec.STABILITY_TREND_VALUES)
        self.assertEqual(realized, set(spec.STABILITY_TREND_VALUES),
                         "stability_trend must reach 'improving' too -- this was "
                         "unreachable when the generator compared one scalar against "
                         "one threshold instead of an early/late delta")

    def test_role_differentiation_rendered_and_fully_reachable(self):
        # First: the line must exist at all (sec X found it was never rendered).
        for ctx in self.ctxs[:5]:
            self.assertIn("Role differentiation: ", ctx)
        realized = self._realized_values("Role differentiation: ", spec.ROLE_DIFFERENTIATION_VALUES)
        self.assertEqual(realized, set(spec.ROLE_DIFFERENTIATION_VALUES))

    def test_role_differentiation_consistent_with_key_windows(self):
        for ctx, key_windows in zip(self.ctxs, self.key_windows_list):
            role_line = next(l for l in ctx.splitlines() if l.startswith("Role differentiation: "))
            rendered_present = role_line == f"Role differentiation: {spec.ROLE_DIFFERENTIATION_PRESENT}"
            for kw in key_windows:
                self.assertEqual(kw["role_differentiation"], rendered_present)

    def test_rules_module_untouched_by_this_change(self):
        # Sanity check on scope: this fix must not have touched RULES's keys/values.
        from build_sft_dataset import RULES
        from swarm_intent.config import BASE_FORMATIONS as BF
        self.assertEqual(len(RULES), 49)
        self.assertEqual(set(RULES.keys()), {(a, b) for a in BF for b in BF})


if __name__ == "__main__":
    unittest.main()
