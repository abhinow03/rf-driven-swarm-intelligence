"""
Step 3 of the "vendor teammate's retrained STGT" session (AUDIT.md sec V):
smoke-tests the vendored, read-only src/swarm_intent/stgt/ subpackage against
the real swarm_data/best_model.pt checkpoint, and guards the deliberate
omissions documented in stgt/README.md (no build_tactical_context/
build_llm_prompt, no upstream "warning" character).

Skips (not fails) if swarm_data/best_model.pt isn't present on disk -- it's
gitignored data handed over out-of-band, not something CI/a fresh clone has.

Usage:
    python -m unittest tests.test_stgt_vendored -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

REPO = os.path.join(os.path.dirname(__file__), "..")
CHECKPOINT = os.path.join(REPO, "swarm_data", "best_model.pt")


class TestStgtVendoredScope(unittest.TestCase):
    """These run regardless of whether the checkpoint is present -- pure source checks."""

    def test_build_tactical_context_not_vendored(self):
        from swarm_intent.stgt import inference as stgt_inference
        self.assertFalse(hasattr(stgt_inference, "build_tactical_context"))
        self.assertFalse(hasattr(stgt_inference, "build_llm_prompt"))
        self.assertFalse(hasattr(stgt_inference, "infer_behavior_trend"))

    def test_predict_v2_and_sliding_window_inference_present(self):
        from swarm_intent.stgt import inference as stgt_inference
        self.assertTrue(callable(stgt_inference.predict_v2))
        self.assertTrue(callable(stgt_inference.sliding_window_inference))

    def test_no_warning_character_anywhere_in_vendored_source(self):
        stgt_dir = os.path.join(REPO, "src", "swarm_intent", "stgt")
        for fname in os.listdir(stgt_dir):
            if not fname.endswith(".py"):
                continue
            with open(os.path.join(stgt_dir, fname), encoding="utf-8") as f:
                text = f.read()
            self.assertNotIn("⚠", text, f"{fname} contains the excluded warning character")

    def test_formation_names_order_matches_this_repos_base_formations(self):
        from swarm_intent.config import BASE_FORMATIONS, TRANSITION_CLASS
        from swarm_intent.stgt.config import FORMATION_NAMES
        self.assertEqual(FORMATION_NAMES, list(BASE_FORMATIONS) + [TRANSITION_CLASS])


@unittest.skipUnless(os.path.exists(CHECKPOINT), "swarm_data/best_model.pt not present")
class TestStgtVendoredAgainstCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        from swarm_intent.stgt.config import device
        from swarm_intent.stgt.model import STGTModel

        cls.ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        cls.model = STGTModel(cls.ckpt["cfg"]).to(device)
        cls.model.load_state_dict(cls.ckpt["model_state_dict"])
        cls.model.eval()

        data_dir = os.path.join(REPO, "swarm_data")
        cls.X_val = np.load(os.path.join(data_dir, "X_val.npy"))
        cls.train_mean = np.load(os.path.join(data_dir, "train_mean.npy"))
        cls.train_std = np.load(os.path.join(data_dir, "train_std.npy"))
        cls.reg_mean = np.load(os.path.join(data_dir, "reg_mean.npy"))
        cls.reg_std = np.load(os.path.join(data_dir, "reg_std.npy"))

    def test_predict_v2_returns_expected_keys(self):
        from swarm_intent.stgt.inference import predict_v2
        out = predict_v2(self.model, self.X_val[0], self.ckpt["cfg"], self.reg_mean, self.reg_std)
        for key in ("formation_type", "formation_confidence", "centroid_velocity",
                   "approach_rate", "formation_stability", "role_differentiation",
                   "transition_from", "transition_to", "class_probabilities"):
            self.assertIn(key, out)

    def test_sliding_window_inference_over_a_longer_stream(self):
        from swarm_intent.stgt.inference import sliding_window_inference
        import numpy as np
        seq_raw = self.X_val[0] * self.train_std + self.train_mean
        long_seq = np.concatenate([seq_raw, seq_raw], axis=0)  # 100 timesteps
        preds = sliding_window_inference(self.model, long_seq, self.ckpt["cfg"],
                                         self.reg_mean, self.reg_std,
                                         self.train_mean, self.train_std)
        expected_n = (long_seq.shape[0] - 50) // 10 + 1
        self.assertEqual(len(preds), expected_n)
        self.assertIn("time_start_s", preds[0])


if __name__ == "__main__":
    unittest.main()
