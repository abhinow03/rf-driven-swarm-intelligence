"""
AUDIT.md sec AM (Rule-0 follow-up, finding 2b resolution): drift guard for
eval_data/LOCKED_seed999_FINAL.json. Regenerates the seed=999, n=1000 population from LIVE
src/swarm_intent/eval_trajectories.py code and fails if its sha256 ever diverges from the
locked file -- the exact silent-drift failure mode sec AM found (99% of trajectories differ
across 3 independently-evolved copies of this same generator, all under the same "seed=999"
label). CPU-only, no checkpoint needed -- this locks the TRAJECTORY population, not any
model's predictions on it.

Usage:
    python -m unittest tests.test_locked_seed999_population -v
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

REPO = os.path.join(os.path.dirname(__file__), "..")
LOCKED_PATH = os.path.join(REPO, "eval_data", "LOCKED_seed999_FINAL.json")


class TestLockedSeed999Population(unittest.TestCase):
    def test_locked_file_exists(self):
        self.assertTrue(os.path.isfile(LOCKED_PATH),
                        "eval_data/LOCKED_seed999_FINAL.json is missing -- the seed=999 "
                        "ceiling population lock (AUDIT.md sec AM) has been deleted")

    def test_live_generator_reproduces_locked_hash(self):
        from swarm_intent.eval_trajectories import sample_chain, build_long_sequence_labeled
        from rule0_2b_regenerate_populations import generate_population

        locked = json.loads(open(LOCKED_PATH).read())
        live_records = generate_population(sample_chain, build_long_sequence_labeled)

        live_sha = hashlib.sha256(json.dumps(live_records).encode()).hexdigest()
        locked_sha = hashlib.sha256(json.dumps(locked["records"]).encode()).hexdigest()

        self.assertEqual(
            live_sha, locked_sha,
            "src/swarm_intent/eval_trajectories.py's seed=999 population has DIVERGED from "
            "eval_data/LOCKED_seed999_FINAL.json. This is exactly the silent-drift failure "
            "mode AUDIT.md sec AM found (3 independently-evolved generator copies, all "
            "labeled 'seed=999', 99% divergent trajectories). If this divergence is "
            "intentional (a real, disclosed generator fix), re-lock deliberately: regenerate "
            "via scripts/rule0_2b_regenerate_populations.py, overwrite "
            "eval_data/LOCKED_seed999_FINAL.json, update the sha256 recorded in "
            "docs/PREREGISTRATION.md and docs/V5_STATE.json, and re-measure every ceiling "
            "figure that depends on it. Do not silently update this test's expectation.")


if __name__ == "__main__":
    unittest.main()
