"""
Phase 3a finalization step 4: drift guard for data/sft_train_v5_phase3a_merged.jsonl.
Same discipline as tests/test_locked_seed999_population.py -- re-derives the merged
corpus from its live source files (deterministic concatenation, no randomness
involved) and fails if the sha256 ever diverges from the locked value recorded in
docs/V5_STATE.json.

Usage:
    python -m unittest tests.test_phase3a_merged_corpus -v
"""
from __future__ import annotations

import hashlib
import json
import os
import unittest

REPO = os.path.join(os.path.dirname(__file__), "..")
MERGED_PATH = os.path.join(REPO, "data", "sft_train_v5_phase3a_merged.jsonl")
V5_STATE_PATH = os.path.join(REPO, "docs", "V5_STATE.json")
EXISTING_CORPUS_FILES = [
    os.path.join(REPO, "data", "sft_train_v5_phase1.jsonl"),
    os.path.join(REPO, "data", "sft_train_v5_phase1_val.jsonl"),
    os.path.join(REPO, "data", "sft_train_v5_phase1_mining.jsonl"),
]
NEW_CORPUS_FILE = os.path.join(REPO, "data", "abstention_corpus_teacher_trimmed900.jsonl")


def _sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


class TestPhase3aMergedCorpus(unittest.TestCase):
    def test_merged_file_exists(self):
        self.assertTrue(os.path.isfile(MERGED_PATH),
                        "data/sft_train_v5_phase3a_merged.jsonl is missing -- the Phase 3a "
                        "finalization merge output has been deleted")

    def test_row_count_is_12901(self):
        with open(MERGED_PATH) as f:
            n = sum(1 for line in f if line.strip())
        self.assertEqual(n, 12901)

    def test_sha256_matches_locked_v5_state(self):
        with open(V5_STATE_PATH) as f:
            state = json.load(f)
        locked_sha = state["phase3a_corpus_finalization"]["step4_lock"]["merged_corpus_sha256"]
        actual_sha = _sha256_of_file(MERGED_PATH)
        self.assertEqual(
            actual_sha, locked_sha,
            "data/sft_train_v5_phase3a_merged.jsonl has DIVERGED from the sha256 locked in "
            "docs/V5_STATE.json's phase3a_corpus_finalization.step4_lock.merged_corpus_sha256. If this "
            "is a deliberate, disclosed regeneration, re-lock explicitly: re-run "
            "llm_finetuning/phase3a_finalize_merge.py, update the recorded sha256 in "
            "V5_STATE.json, and re-verify every downstream artifact that depends on this "
            "corpus. Do not silently update this test's expectation.")

    def test_live_reconstruction_reproduces_locked_hash(self):
        """Deterministic re-derivation from the 4 source files, not a stored copy check."""
        rows = []
        for path in EXISTING_CORPUS_FILES:
            with open(path) as f:
                rows.extend(json.loads(l) for l in f if l.strip())
        with open(NEW_CORPUS_FILE) as f:
            rows.extend(json.loads(l) for l in f if l.strip())
        live_content = "\n".join(json.dumps(r) for r in rows) + "\n"
        live_sha = hashlib.sha256(live_content.encode()).hexdigest()
        actual_sha = _sha256_of_file(MERGED_PATH)
        self.assertEqual(live_sha, actual_sha)


if __name__ == "__main__":
    unittest.main()
