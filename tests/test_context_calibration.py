"""
Regression test for the context_spec/calibration refactor (see calibration.py,
context_spec.py). Proves the refactor changed nothing: build_tactical_context()
with AbsoluteCalibrator (the default) must reproduce EXACTLY what the pre-refactor,
hardcoded-literal version of the function produced.

evaluation/llm_run_output.json is the only real captured STGT pipeline run on disk
(see AUDIT.md sec A — evaluation/finetuned_eval.json does not exist). It does NOT
contain the raw tactical_context STRING (only its "predictions" input list and its
"ml_summary" dict, the function's second return value) — so this test rebuilds the
expected string from a frozen, inlined copy of the exact pre-refactor logic applied
to that same real predictions list, rather than comparing against a saved string
that was never written to disk. It additionally cross-checks the refactored
function's summary dict against the ACTUAL saved ml_summary field, which does exist
in the file — that part compares against real historical output, not just a
same-session oracle.

PROVENANCE CHECK (done): `git show 2653c96:src/swarm_intent/inference.py` — the
initial commit, and inference.py's only ancestor before the calibration refactor
(e083950); the file was untouched in between — was diffed line-for-line against the
`_pre_refactor_build_tactical_context` function below. The only differences were
blank lines and the docstring; every line of executable logic matched exactly. The
oracle below is not just "written to look right", it has been checked against the
actual git history it claims to reproduce.

Uses stdlib unittest (no pytest in this repo's dependencies) — run with:
    python -m unittest tests.test_context_calibration -v
"""
from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from swarm_intent.calibration import AbsoluteCalibrator
from swarm_intent.inference import build_tactical_context

REPO = os.path.join(os.path.dirname(__file__), "..")
DEMO_JSON = os.path.join(REPO, "evaluation", "llm_run_output.json")

TRANSITION_CLASS = "transitioning"


def _pre_refactor_build_tactical_context(predictions):
    """Frozen, inlined copy of build_tactical_context() exactly as it was before
    the context_spec/calibration refactor (hardcoded +-0.5 / +-0.1 / +-0.1
    literals, no calibrator). Reference oracle for this test ONLY — do not import
    from inference.py here, that would make the test tautological."""
    n = len(predictions)
    if n == 0:
        return "No predictions available.", {}
    formation_seq = [p["formation_type"] for p in predictions]
    transitions = [
        {"at_time_s": predictions[i]["time_start_s"], "from": formation_seq[i - 1], "to": formation_seq[i]}
        for i in range(1, n) if formation_seq[i] != formation_seq[i - 1]
    ]
    non_trans = [f for f in formation_seq if f != TRANSITION_CLASS]
    dominant = max(set(non_trans), key=non_trans.count) if non_trans else TRANSITION_CLASS
    velocities = [p["centroid_velocity"] for p in predictions]
    mid = n // 2
    delta_v = (np.mean(velocities[mid:]) - np.mean(velocities[:mid])) if mid > 0 else 0.0
    vel_trend = "accelerating" if delta_v > 0.5 else "decelerating" if delta_v < -0.5 else "steady"
    stabilities = [p["formation_stability"] for p in predictions]
    mean_stability = float(np.mean(stabilities))
    early, late = np.mean(stabilities[:max(1, mid)]), (np.mean(stabilities[mid:]) if mid > 0 else stabilities[-1])
    stab_trend = "degrading" if late < early - 0.1 else "improving" if late > early + 0.1 else "holding"
    approach = [p["approach_rate"] for p in predictions]
    mean_approach = float(np.mean(approach))
    approach_summary = ("converging (drones closing in)" if mean_approach < -0.1
                        else "dispersing (drones spreading out)" if mean_approach > 0.1
                        else "stable spread")
    role_flag = sum(1 for p in predictions if p["role_differentiation"]) > (n // 2)
    confidences = [p["formation_confidence"] for p in predictions]
    mean_conf, low_conf = float(np.mean(confidences)), sum(1 for c in confidences if c < 0.6)
    lines = [
        f"Observation window: {predictions[0]['time_start_s']}s - {predictions[-1]['time_end_s']}s "
        f"({n} overlapping {len(predictions) and 50}-step windows)",
        f"Dominant formation: {dominant}",
        f"Formation history: {' -> '.join(dict.fromkeys(formation_seq))}",
    ]
    lines += ([f"Transition at t={t['at_time_s']}s: {t['from']} -> {t['to']}" for t in transitions]
              or ["No formation transitions detected."])
    lines += [
        f"Velocity trend: {vel_trend} (delta_v={delta_v:+.2f})",
        f"Formation stability: {stab_trend} (mean={mean_stability:.2f})",
        f"Spread dynamics: {approach_summary} (mean approach_rate={mean_approach:.3f})",
        f"Role differentiation: {'present' if role_flag else 'not prominent'}",
        f"Classifier confidence: mean={mean_conf:.2f} ({low_conf} low-confidence windows)",
    ]
    summary = {
        "dominant_formation": dominant, "formation_history": list(dict.fromkeys(formation_seq)),
        "transitions_detected": transitions, "velocity_trend": vel_trend,
        "delta_velocity": round(float(delta_v), 3), "stability_trend": stab_trend,
        "mean_stability": round(mean_stability, 3), "spread_dynamics": approach_summary,
        "mean_approach_rate": round(mean_approach, 3), "role_differentiation": role_flag,
        "mean_confidence": round(mean_conf, 3), "low_conf_windows": low_conf, "n_windows": n,
    }
    return "\n".join(lines), summary


class TestAbsoluteCalibratorRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(DEMO_JSON) as f:
            cls.demo = json.load(f)
        cls.predictions = cls.demo["predictions"]

    def test_predictions_present_and_nonempty(self):
        # Guards the test itself against a silently-empty/malformed fixture.
        self.assertEqual(len(self.predictions), 9)

    def test_explicit_absolute_calibrator_matches_pre_refactor_string_exactly(self):
        expected_ctx, _ = _pre_refactor_build_tactical_context(self.predictions)
        actual_ctx, _ = build_tactical_context(self.predictions, calibrator=AbsoluteCalibrator())
        self.assertEqual(actual_ctx, expected_ctx)

    def test_default_calibrator_matches_pre_refactor_string_exactly(self):
        # No calibrator passed -> must default to AbsoluteCalibrator, byte-identical.
        expected_ctx, _ = _pre_refactor_build_tactical_context(self.predictions)
        actual_ctx, _ = build_tactical_context(self.predictions)
        self.assertEqual(actual_ctx, expected_ctx)

    def test_summary_dict_matches_pre_refactor_oracle(self):
        _, expected_summary = _pre_refactor_build_tactical_context(self.predictions)
        _, actual_summary = build_tactical_context(self.predictions, calibrator=AbsoluteCalibrator())
        self.assertEqual(actual_summary, expected_summary)

    def test_summary_dict_matches_saved_ml_summary_in_demo_json(self):
        # Cross-check against real historical output actually saved to disk, not
        # just this test's own same-session oracle.
        _, actual_summary = build_tactical_context(self.predictions, calibrator=AbsoluteCalibrator())
        self.assertEqual(actual_summary, self.demo["ml_summary"])


if __name__ == "__main__":
    unittest.main()
